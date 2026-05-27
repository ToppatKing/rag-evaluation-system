"""FAISS-backed vector store for chunk storage and nearest-neighbour search.

Supports:
* Cosine similarity (inner-product on L2-normalised vectors) and L2 distance.
* Persistent serialisation (``save`` / ``load``) using FAISS's native
  binary format plus a companion JSON metadata file.
* Batch ``add`` for efficiency.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StoredChunk:
    """A chunk entry stored in the vector store.

    Attributes:
        text: The chunk text.
        source: Origin document path.
        chunk_index: Position within the source document.
        metadata: Arbitrary additional metadata.
    """

    text: str
    source: str
    chunk_index: int = 0
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class FAISSVectorStore:
    """In-memory FAISS index with persistent backing.

    Args:
        dimension: Embedding dimensionality.
        metric: ``"cosine"`` (inner product on normalised vecs) or ``"l2"``.

    Example::

        store = FAISSVectorStore(dimension=384)
        store.add(vectors, chunks)
        results = store.search(query_vec, top_k=5)
        store.save("data/faiss_index")

        store2 = FAISSVectorStore.load("data/faiss_index")
    """

    def __init__(self, dimension: int, metric: str = "cosine") -> None:
        try:
            import faiss  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Install faiss-cpu: pip install faiss-cpu") from exc

        self._dimension = dimension
        self._metric = metric
        self._chunks: list[StoredChunk] = []

        if metric == "cosine":
            self._index = faiss.IndexFlatIP(dimension)
        elif metric == "l2":
            self._index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported metric: {metric!r}. Use 'cosine' or 'l2'.")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        return self._index.ntotal

    @property
    def dimension(self) -> int:
        return self._dimension

    # ── Mutating operations ───────────────────────────────────────────────────

    def add(self, vectors: np.ndarray, chunks: list[StoredChunk]) -> None:
        """Add *vectors* and their corresponding *chunks* to the store.

        Args:
            vectors: Float32 array of shape ``(N, dimension)``.
            chunks: List of :class:`StoredChunk` objects of length N.

        Raises:
            ValueError: On shape or length mismatch.
        """
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Expected vectors of shape (N, {self._dimension}), "
                f"got {vectors.shape}"
            )
        if len(vectors) != len(chunks):
            raise ValueError(
                f"vectors ({len(vectors)}) and chunks ({len(chunks)}) must match"
            )

        self._index.add(vectors)
        self._chunks.extend(chunks)
        logger.debug("Added %d vectors; store size now %d", len(vectors), self.size)

    def clear(self) -> None:
        """Remove all entries from the store."""
        self._index.reset()
        self._chunks.clear()

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[StoredChunk, float]]:
        """Return the *top_k* most similar chunks and their scores.

        Args:
            query_vector: Float32 array of shape ``(dimension,)``.
            top_k: Number of results to return.

        Returns:
            List of ``(StoredChunk, score)`` tuples, descending by score.
            For cosine metric, scores are in ``[-1, 1]``.
            For L2 metric, scores are non-negative distances (lower = closer).
        """
        if self.size == 0:
            return []

        qv = query_vector.astype(np.float32)
        if qv.ndim == 1:
            qv = qv[np.newaxis, :]

        k = min(top_k, self.size)
        scores, indices = self._index.search(qv, k)

        results: list[tuple[StoredChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    def search_batch(
        self,
        query_vectors: np.ndarray,
        top_k: int = 5,
    ) -> list[list[tuple[StoredChunk, float]]]:
        """Batch version of :meth:`search`.

        Args:
            query_vectors: Float32 array of shape ``(Q, dimension)``.
            top_k: Results per query.

        Returns:
            List of Q result lists.
        """
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)
        if self.size == 0:
            return [[] for _ in range(len(query_vectors))]

        k = min(top_k, self.size)
        scores_batch, indices_batch = self._index.search(query_vectors, k)

        all_results: list[list[tuple[StoredChunk, float]]] = []
        for scores, indices in zip(scores_batch, indices_batch):
            row: list[tuple[StoredChunk, float]] = []
            for score, idx in zip(scores, indices):
                if idx != -1:
                    row.append((self._chunks[idx], float(score)))
            all_results.append(row)
        return all_results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist the index and metadata to *path* (without extension).

        Creates two files:
        * ``<path>.faiss`` — FAISS binary index
        * ``<path>.json``  — chunk metadata
        """
        import faiss  # type: ignore[import-untyped]

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        metadata = {
            "dimension": self._dimension,
            "metric": self._metric,
            "chunks": [asdict(c) for c in self._chunks],
        }
        path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )
        logger.info("Saved FAISS index (%d vectors) to %s", self.size, path)

    @classmethod
    def load(cls, path: str | Path) -> "FAISSVectorStore":
        """Load a previously saved index.

        Args:
            path: Base path (without extension) used when saving.

        Returns:
            Populated :class:`FAISSVectorStore`.
        """
        import faiss  # type: ignore[import-untyped]

        path = Path(path)
        metadata = json.loads(path.with_suffix(".json").read_text())

        store = cls(dimension=metadata["dimension"], metric=metadata["metric"])
        store._index = faiss.read_index(str(path.with_suffix(".faiss")))
        store._chunks = [StoredChunk(**c) for c in metadata["chunks"]]
        logger.info("Loaded FAISS index (%d vectors) from %s", store.size, path)
        return store
