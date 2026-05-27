"""Retrieval strategies: dense similarity search and Maximal Marginal Relevance.

* :class:`DenseRetriever` — standard nearest-neighbour lookup.
* :class:`MMRRetriever`   — re-ranks candidates for relevance *and* diversity.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from rag_eval.retrieval.embedder import BaseEmbedder
from rag_eval.retrieval.vector_store import FAISSVectorStore, StoredChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result.

    Attributes:
        chunk: The retrieved text chunk.
        score: Relevance score (higher = more relevant for cosine; lower for L2).
        rank: Zero-based rank in the result list.
    """

    chunk: StoredChunk
    score: float
    rank: int

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def source(self) -> str:
        return self.chunk.source


class BaseRetriever(ABC):
    """Abstract retriever interface."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: BaseEmbedder,
        top_k: int = 5,
    ) -> None:
        self._store = vector_store
        self._embedder = embedder
        self._top_k = top_k

    @abstractmethod
    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Return the top-k most relevant chunks for *query*.

        Args:
            query: Natural-language query string.

        Returns:
            Ordered list of :class:`RetrievalResult` objects.
        """


class DenseRetriever(BaseRetriever):
    """Retrieve by pure cosine / L2 similarity."""

    def retrieve(self, query: str) -> list[RetrievalResult]:
        qv = self._embedder.embed_query(query)
        raw = self._store.search(qv, top_k=self._top_k)
        return [
            RetrievalResult(chunk=chunk, score=score, rank=i)
            for i, (chunk, score) in enumerate(raw)
        ]


class MMRRetriever(BaseRetriever):
    """Maximal Marginal Relevance retrieval.

    Balances relevance to the query with diversity among selected chunks.

    The MMR score for candidate *c* at step *i* is::

        MMR(c) = λ · sim(q, c) - (1 - λ) · max_{s ∈ S} sim(s, c)

    where ``S`` is the set of already-selected chunks and ``q`` is the query.

    Args:
        vector_store: Populated FAISS store.
        embedder: Embedding backend.
        top_k: Number of chunks to return.
        mmr_lambda: Trade-off weight ∈ [0, 1].
            1.0 = pure relevance (equivalent to :class:`DenseRetriever`).
            0.0 = pure diversity.
        fetch_k: Candidate pool size before MMR re-ranking (should be ≥ top_k).
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: BaseEmbedder,
        top_k: int = 5,
        mmr_lambda: float = 0.6,
        fetch_k: int = 20,
    ) -> None:
        super().__init__(vector_store, embedder, top_k)
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError(f"mmr_lambda must be in [0, 1]; got {mmr_lambda}")
        self._lambda = mmr_lambda
        self._fetch_k = max(fetch_k, top_k)

    @staticmethod
    def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return cosine similarities between every pair (a[i], b[j]).

        Args:
            a: Shape ``(M, D)``.
            b: Shape ``(N, D)``.

        Returns:
            Shape ``(M, N)`` float32 similarity matrix.
        """
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return (a_n @ b_n.T).astype(np.float32)

    def retrieve(self, query: str) -> list[RetrievalResult]:
        qv = self._embedder.embed_query(query)
        candidates = self._store.search(qv, top_k=self._fetch_k)

        if not candidates:
            return []

        # Collect candidate embeddings by re-embedding their texts.
        # For large corpora, storing embeddings alongside chunks is faster;
        # this implementation keeps the vector store simple.
        candidate_texts = [c.text for c, _ in candidates]
        cand_vecs = self._embedder.embed_documents(candidate_texts)

        q_sims = np.array([score for _, score in candidates], dtype=np.float32)

        selected_indices: list[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(min(self._top_k, len(candidates))):
            if not remaining:
                break

            if not selected_indices:
                # First pick: purely relevance-based
                best = max(remaining, key=lambda i: q_sims[i])
            else:
                selected_vecs = cand_vecs[selected_indices]
                # (remaining × selected) similarity matrix
                r_vecs = cand_vecs[remaining]
                inter_sims = self._cosine_matrix(r_vecs, selected_vecs)
                max_inter = inter_sims.max(axis=1)

                mmr_scores = (
                    self._lambda * q_sims[remaining]
                    - (1 - self._lambda) * max_inter
                )
                best = remaining[int(np.argmax(mmr_scores))]

            selected_indices.append(best)
            remaining.remove(best)

        return [
            RetrievalResult(chunk=candidates[i][0], score=float(q_sims[i]), rank=rank)
            for rank, i in enumerate(selected_indices)
        ]


# ── Factory ───────────────────────────────────────────────────────────────────


def build_retriever(
    config: dict[str, object],
    vector_store: FAISSVectorStore,
    embedder: BaseEmbedder,
) -> BaseRetriever:
    """Construct a retriever from config.

    Args:
        config: Must contain ``method`` (``"dense"`` or ``"mmr"``),
            ``top_k``, and optionally ``mmr_lambda`` / ``mmr_fetch_k``.
        vector_store: Populated :class:`FAISSVectorStore`.
        embedder: :class:`BaseEmbedder` for query embedding.

    Returns:
        Configured :class:`BaseRetriever`.
    """
    method = str(config.get("method", "dense"))
    top_k = int(config.get("top_k", 5))

    if method == "dense":
        return DenseRetriever(vector_store, embedder, top_k=top_k)
    if method == "mmr":
        return MMRRetriever(
            vector_store,
            embedder,
            top_k=top_k,
            mmr_lambda=float(config.get("mmr_lambda", 0.6)),
            fetch_k=int(config.get("mmr_fetch_k", 20)),
        )
    raise ValueError(f"Unknown retrieval method: {method!r}")
