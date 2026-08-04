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

import google.generativeai as genai
import numpy as np
import os

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract interface for embedding models.

    Implementors must provide :meth:`embed_documents` and :meth:`embed_query`.
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
    """
    Embeddings via Google Gemini API (free tier).
    Requires GEMINI_API_KEY in environment.
    """

    def __init__(self, model_name: str = "models/embedding-001", normalize: bool = True):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self._model_name = model_name
        self._normalize = normalize
        self._dim = 768  # Gemini embedding dimension

    @property
    def dimension(self) -> int:
        return self._dim

     def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            response = genai.embed_content(
                model=self._model_name,
                content=t,
                task_type="retrieval_document"
            )
            vec = np.array(response["embedding"], dtype=np.float32)
            if self._normalize:
                vec /= np.linalg.norm(vec)
            vecs.append(vec)
        return np.vstack(vecs)


    def embed_query(self, query: str) -> np.ndarray:
        response = genai.embed_content(
            model=self._model_name,
            content=query,
            task_type="retrieval_query"
        )
        vec = np.array(response["embedding"], dtype=np.float32)
        if self._normalize:
            vec /= np.linalg.norm(vec)
        return vec

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


def build_embedder(config: dict[str, Any]) -> BaseEmbedder:
    """Construct an embedder from a configuration dictionary.

    Args:
        config: Must contain ``provider`` key (``"sentence_transformers"`` or
            ``"openai"``) plus model-specific keys.

    Returns:
        A configured :class:`BaseEmbedder`.
    """
    provider = config.get("provider", "sentence_transformers")
    model_name = str(config.get("model_name", ""))
    batch_size = int(config.get("batch_size", 64))
    normalize = bool(config.get("normalize", True))

    if provider == "sentence_transformers":
        return SentenceTransformerEmbedder(
            model_name=model_name or "all-MiniLM-L6-v2",
            batch_size=batch_size,
            normalize=normalize,
        )
    if provider == "openai":
        return OpenAIEmbedder(
            model_name=model_name or "text-embedding-3-small",
            batch_size=batch_size,
            normalize=normalize,
        )
    raise ValueError(f"Unknown embedding provider: {provider!r}")
