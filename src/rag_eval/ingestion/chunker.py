"""Text chunking strategies for the RAG ingestion pipeline.

Three strategies are provided:

* :class:`FixedSizeChunker`    — character-level sliding window
* :class:`RecursiveChunker`   — hierarchical separator splitting (LangChain-inspired)
* :class:`SemanticChunker`    — embedding-similarity sentence grouping
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_eval.ingestion.loader import Document


# ── Chunk dataclass ──────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """A single text chunk with source provenance.

    Attributes:
        text: The chunk text content.
        source: Path to the originating document.
        chunk_index: Zero-based position within the document.
        metadata: Additional key-value metadata.
    """

    text: str
    source: str
    chunk_index: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        preview = self.text[:50].replace("\n", " ")
        return f"Chunk(idx={self.chunk_index}, len={len(self)}, preview={preview!r}...)"


# ── Abstract base ────────────────────────────────────────────────────────────


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        """Split *text* into :class:`Chunk` objects.

        Args:
            text: Input text to split.
            source: Provenance label (typically a file path).

        Returns:
            Ordered list of non-empty chunks.
        """

    def chunk_document(self, doc: "Document") -> list[Chunk]:
        """Convenience method to chunk a :class:`~rag_eval.ingestion.loader.Document`."""
        return self.chunk(doc.content, source=doc.source)


# ── Fixed-size chunker ───────────────────────────────────────────────────────


class FixedSizeChunker(BaseChunker):
    """Splits text into fixed-character windows with optional overlap.

    Args:
        chunk_size: Target chunk length in characters.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(text=piece, source=source, chunk_index=idx))
                idx += 1
            start += step
        return chunks


# ── Recursive chunker ────────────────────────────────────────────────────────


class RecursiveChunker(BaseChunker):
    """Hierarchical splitting using a priority list of separators.

    Tries each separator in order; if the resulting pieces are still larger
    than *chunk_size*, recursively splits them with the next separator.
    Adjacent pieces are greedily merged back up to *chunk_size*.

    This mirrors the approach popularised by LangChain's
    ``RecursiveCharacterTextSplitter`` but is implemented from scratch.

    Args:
        chunk_size: Target maximum chunk length in characters.
        chunk_overlap: Characters to include from the previous chunk.
        separators: Ordered list of separator strings to try.
    """

    _DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self._DEFAULT_SEPARATORS

    # ── internal helpers ─────────────────────────────────────────────────────

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split *text* until all pieces fit in *chunk_size*."""
        if len(text) <= self.chunk_size:
            return [text]

        sep, *rest_seps = separators

        if sep == "":
            # Character-level fallback
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        pieces = text.split(sep)
        result: list[str] = []
        for piece in pieces:
            if len(piece) <= self.chunk_size:
                result.append(piece)
            else:
                result.extend(self._split(piece, rest_seps or [""]))
        return [p for p in result if p.strip()]

    def _merge(self, pieces: list[str]) -> list[str]:
        """Greedily merge small pieces into chunks up to *chunk_size*."""
        merged: list[str] = []
        current: list[str] = []
        current_len = 0

        for piece in pieces:
            if current_len + len(piece) + 1 > self.chunk_size and current:
                merged.append(" ".join(current).strip())
                # Keep the overlap tail
                overlap_text = " ".join(current).strip()[-self.chunk_overlap :]
                current = [overlap_text, piece] if overlap_text else [piece]
                current_len = len(current[-1]) + len(current[0]) + 1
            else:
                current.append(piece)
                current_len += len(piece) + 1

        if current:
            merged.append(" ".join(current).strip())
        return [m for m in merged if m]

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        pieces = self._split(text, self.separators)
        merged = self._merge(pieces)
        return [
            Chunk(text=t, source=source, chunk_index=i)
            for i, t in enumerate(merged)
            if t.strip()
        ]


# ── Semantic chunker ─────────────────────────────────────────────────────────


class SemanticChunker(BaseChunker):
    """Groups sentences into chunks by embedding-space similarity.

    Sentences are split by terminal punctuation, then embedded with the
    provided *embed_fn*.  Consecutive sentences are merged into a chunk as
    long as their pairwise cosine similarity exceeds *threshold*.  When
    similarity drops below the threshold, a new chunk is started.

    Args:
        embed_fn: Callable that takes a list[str] and returns a 2-D float
            array of shape ``(N, D)``.
        threshold: Cosine similarity below which a boundary is inserted.
        min_chunk_size: Minimum characters to avoid trivially small chunks.
    """

    def __init__(
        self,
        embed_fn: object,  # Callable[[list[str]], np.ndarray]
        threshold: float = 0.85,
        min_chunk_size: int = 100,
    ) -> None:
        self.embed_fn = embed_fn
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Naively split text on sentence-ending punctuation."""
        pattern = r"(?<=[.!?])\s+"
        sentences = re.split(pattern, text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _cosine(a: "list[float]", b: "list[float]") -> float:
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        import numpy as np  # type: ignore[import-untyped]

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # Embed all sentences at once for efficiency
        embeddings: list[list[float]] = self.embed_fn(sentences)  # type: ignore[call-arg]

        groups: list[list[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            sim = self._cosine(embeddings[i - 1], embeddings[i])
            if sim >= self.threshold:
                groups[-1].append(sentences[i])
            else:
                groups.append([sentences[i]])

        # Merge tiny groups forward
        merged: list[str] = []
        buffer = ""
        for group in groups:
            text_piece = " ".join(group)
            if len(buffer) + len(text_piece) < self.min_chunk_size:
                buffer = (buffer + " " + text_piece).strip()
            else:
                if buffer:
                    merged.append(buffer)
                buffer = text_piece
        if buffer:
            merged.append(buffer)

        return [
            Chunk(text=t, source=source, chunk_index=i)
            for i, t in enumerate(merged)
            if t.strip()
        ]


# ── Factory ──────────────────────────────────────────────────────────────────


def build_chunker(config: dict[str, object]) -> BaseChunker:
    """Construct a :class:`BaseChunker` from a configuration dictionary.

    Args:
        config: Dictionary with at minimum a ``strategy`` key.
            For ``fixed`` and ``recursive`` strategies, ``chunk_size`` and
            ``chunk_overlap`` are also read.

    Returns:
        Configured chunker instance.

    Raises:
        ValueError: If *strategy* is unknown.
    """
    strategy = str(config.get("strategy", "recursive"))
    size = int(config.get("chunk_size", 512))
    overlap = int(config.get("chunk_overlap", 64))

    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=size, chunk_overlap=overlap)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size=size, chunk_overlap=overlap)
    if strategy == "semantic":
        # Caller must inject embed_fn separately after construction
        raise ValueError(
            "Use SemanticChunker(embed_fn=...) directly; "
            "embed_fn cannot be supplied via config dict."
        )
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")
