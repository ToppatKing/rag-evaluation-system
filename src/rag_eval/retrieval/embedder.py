"""Text embedding backends.

Two implementations share the :class:`BaseEmbedder` interface:

* :class:`SentenceTransformerEmbedder` — local inference, no API key required.
* :class:`OpenAIEmbedder`             — OpenAI ``text-embedding-3-*`` models.

Both return **L2-normalised** float vectors when *normalize=True* (default),
making dot-product and cosine similarity equivalent.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from google.genai import Client
import numpy as np
import os

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract interface for embedding models.

    Implementors must provide :meth:`embed_documents` and :meth:`embed_query`.
    A convenience :meth:`embed` alias is also provided for callers that expect
    a simple batch API.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of documents.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            Float32 array of shape ``(len(texts), dimension)``.
        """

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Args:
            query: The query text.

        Returns:
            Float32 array of shape ``(dimension,)``.
        """

    def embed(self, texts: list[str]) -> np.ndarray:
        """Compatibility wrapper used by the retriever stack.

        This mirrors a simpler batch-embedding API that some callers expect.
        """
        return self.embed_documents(texts)

    @staticmethod
    def _l2_normalise(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms


# ── SentenceTransformer backend ───────────────────────────────────────────────


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embeddings via the ``sentence-transformers`` library.

    Args:
        model_name: Any model from the HuggingFace Hub compatible with
            ``sentence_transformers.SentenceTransformer``.
            Default: ``"all-MiniLM-L6-v2"`` (fast, 384-dim).
        batch_size: Number of texts per encoding batch.
        normalize: L2-normalise output vectors.
        device: ``"cpu"`` or ``"cuda"``; inferred automatically if *None*.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        normalize: bool = True,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers: pip install sentence-transformers"
            ) from exc

        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize = normalize
        logger.info("Loading SentenceTransformer model: %s", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        self._dim: int = self._model.get_sentence_embedding_dimension()  # type: ignore[assignment]

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        logger.debug("Embedding %d documents with SentenceTransformer", len(texts))
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_documents([query])[0]

# ── Gemini backend ────────────────────────────────────────────────────────────

class GeminiEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "models/embedding-001",
        normalize: bool = True,
        fallback_embedder: BaseEmbedder | None = None,
    ):
        self._model_name = model_name
        self._normalize = normalize
        self._fallback_embedder = fallback_embedder
        if self._fallback_embedder is None:
            try:
                self._fallback_embedder = SentenceTransformerEmbedder(
                    model_name="all-MiniLM-L6-v2",
                    batch_size=64,
                    normalize=normalize,
                )
            except Exception as exc:  # pragma: no cover - best-effort fallback
                logger.warning("Unable to initialize local fallback embedder: %s", exc)
                self._fallback_embedder = None

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._client = Client(api_key=api_key) if api_key else None

        # Gemini embedding dimension. If the remote backend is unavailable, fall
        # back to the local embedder's dimension so the FAISS store is created
        # with matching vector shapes.
        self._dim = 768
        self._use_fallback = False
        if not api_key:
            self._use_fallback = True
            if self._fallback_embedder is not None:
                self._dim = self._fallback_embedder.dimension
        else:
            try:
                self._probe_remote()
            except Exception as exc:  # pragma: no cover - exercised at runtime
                logger.warning("Gemini embeddings unavailable (%s); using local fallback", exc)
                self._use_fallback = True
                if self._fallback_embedder is not None:
                    self._dim = self._fallback_embedder.dimension

    def _probe_remote(self) -> None:
        if self._client is None:
            raise RuntimeError("No Gemini client configured")
        self._client.models.embed_content(model=self._model_name, contents="probe")

    def _extract_embedding(self, response: Any) -> np.ndarray:
        if hasattr(response, "embeddings") and response.embeddings:
            embedding = response.embeddings[0]
            if hasattr(embedding, "values"):
                values = embedding.values
            elif isinstance(embedding, dict):
                values = embedding.get("values")
            else:
                values = None
        elif isinstance(response, dict):
            embedding = response.get("embedding") or response.get("embeddings", [{}])[0]
            if isinstance(embedding, dict):
                values = embedding.get("values") or embedding.get("embedding")
            else:
                values = embedding
        else:
            values = None

        if values is None:
            raise ValueError(f"Unable to parse Gemini embedding response: {response!r}")

        vec = np.array(values, dtype=np.float32)
        if self._normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec

    def _embed_single(self, text: str) -> np.ndarray:
        if self._use_fallback or self._client is None:
            if self._fallback_embedder is None:
                raise RuntimeError("Gemini embeddings unavailable and no local fallback embedder")
            return self._fallback_embedder.embed_query(text)

        try:
            response = self._client.models.embed_content(model=self._model_name, contents=text)
        except AttributeError:
            response = self._client.embed_content(model=self._model_name, content=text)
        except Exception as exc:  # pragma: no cover - exercised at runtime
            if self._fallback_embedder is None:
                raise
            logger.warning("Gemini embedding failed (%s); falling back to local embeddings", exc)
            self._use_fallback = True
            self._dim = self._fallback_embedder.dimension
            return self._fallback_embedder.embed_query(text)
        return self._extract_embedding(response)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs = [self._embed_single(t) for t in texts]
        return np.vstack(vecs)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed_single(query)

# ── OpenAI backend ────────────────────────────────────────────────────────────


class OpenAIEmbedder:
    """Embeddings via the OpenAI Embeddings API.

    Requires ``OPENAI_API_KEY`` to be set in the environment or passed
    explicitly via *api_key*.

    Args:
        model_name: OpenAI embedding model, e.g. ``"text-embedding-3-small"``.
        batch_size: Maximum texts per API request.
        normalize: L2-normalise output vectors.
        api_key: Override the ``OPENAI_API_KEY`` environment variable.
    """

    # Maps model name → known dimension (used when dimension must be known
    # before the first call, e.g. for FAISS index creation).
    _KNOWN_DIMS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        batch_size: int = 100,
        normalize: bool = True,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc

        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize = normalize
        self._client = OpenAI(api_key=api_key)
        self._dim = self._KNOWN_DIMS.get(model_name, 1536)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.embeddings.create(model=self._model_name, input=batch)
            all_vectors.extend(item.embedding for item in response.data)
        vecs = np.array(all_vectors, dtype=np.float32)
        return self._l2_normalise(vecs) if self._normalize else vecs

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_documents([query])[0]


# ── Factory ───────────────────────────────────────────────────────────────────


def build_embedder(cfg):
    provider = cfg.get("provider", "sentence_transformer")
    provider = provider.lower()

    if provider in {"sentence_transformer", "sentence-transformer", "local", "local_embedder"}:
        return SentenceTransformerEmbedder(
            model_name=cfg.get("model", "all-MiniLM-L6-v2"),
            batch_size=cfg.get("batch_size", 64),
            normalize=cfg.get("normalize", True),
            device=cfg.get("device")
        )

    elif provider == "openai":
        return OpenAIEmbedder(
            model_name=cfg.get("model", "text-embedding-3-small"),
            batch_size=cfg.get("batch_size", 100),
            normalize=cfg.get("normalize", True),
            api_key=cfg.get("api_key")
        )

    elif provider == "gemini":
        api_key = cfg.get("api_key") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning(
                "No Gemini API key found; falling back to local sentence-transformers embeddings"
            )
            return SentenceTransformerEmbedder(
                model_name=cfg.get("model", "all-MiniLM-L6-v2"),
                batch_size=cfg.get("batch_size", 64),
                normalize=cfg.get("normalize", True),
                device=cfg.get("device"),
            )
            try:
                return GeminiEmbedder(
                    model_name=cfg.get("model", "models/embedding-001"),
                    normalize=cfg.get("normalize", True),
                    fallback_embedder=None,
                )
            except Exception as exc:
                logger.warning(
                    "Gemini embedder unavailable (%s); falling back to local sentence-transformers embeddings",
                    exc,
                )
                return SentenceTransformerEmbedder(
                    model_name=cfg.get("model", "all-MiniLM-L6-v2"),
                    batch_size=cfg.get("batch_size", 64),
                    normalize=cfg.get("normalize", True),
                    device=cfg.get("device"),
                )
