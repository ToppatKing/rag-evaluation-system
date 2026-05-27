"""Tests for rag_eval.ingestion.chunker."""

from __future__ import annotations

import pytest

from rag_eval.ingestion.chunker import (
    Chunk,
    FixedSizeChunker,
    RecursiveChunker,
    build_chunker,
)


class TestFixedSizeChunker:
    def test_single_chunk_short_text(self) -> None:
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk("Short text.", source="test.txt")
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."
        assert chunks[0].source == "test.txt"

    def test_multiple_chunks_produced(self) -> None:
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        text = "a" * 200
        chunks = chunker.chunk(text)
        # With size=50 and overlap=10, step=40 → ceil(200/40)=5 chunks
        assert len(chunks) >= 4

    def test_chunk_indices_sequential(self) -> None:
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk("x" * 300)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_overlap_creates_shared_content(self) -> None:
        text = "abcdefghijklmnopqrstuvwxyz" * 4  # 104 chars
        chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=5)
        chunks = chunker.chunk(text)
        if len(chunks) >= 2:
            # Last 5 chars of chunk N should appear at start of chunk N+1
            tail = chunks[0].text[-5:]
            head = chunks[1].text[:5]
            assert tail == head

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            FixedSizeChunker(chunk_size=50, chunk_overlap=50)

    def test_empty_text_returns_empty(self) -> None:
        chunker = FixedSizeChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []


class TestRecursiveChunker:
    def test_short_text_single_chunk(self) -> None:
        chunker = RecursiveChunker(chunk_size=500)
        chunks = chunker.chunk("Hello world.", source="f.txt")
        assert len(chunks) == 1

    def test_paragraph_boundaries_respected(self) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=0)
        chunks = chunker.chunk(text)
        # Each paragraph fits in 30 chars, so should stay separate
        assert len(chunks) >= 2
        for c in chunks:
            assert c.text.strip() != ""

    def test_no_empty_chunks(self, sample_text: str) -> None:
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(sample_text)
        for c in chunks:
            assert len(c.text.strip()) > 0

    def test_chunk_size_respected(self, sample_text: str) -> None:
        size = 100
        chunker = RecursiveChunker(chunk_size=size, chunk_overlap=0)
        chunks = chunker.chunk(sample_text)
        # Chunks may slightly exceed size due to sentence merging, but not
        # by more than ~2x (heuristic check)
        oversized = [c for c in chunks if len(c.text) > size * 2]
        assert len(oversized) == 0

    def test_document_chunking(self, sample_document) -> None:
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=20)
        chunks = chunker.chunk_document(sample_document)
        assert all(c.source == sample_document.source for c in chunks)


class TestBuildChunker:
    def test_build_fixed(self) -> None:
        chunker = build_chunker({"strategy": "fixed", "chunk_size": 200, "chunk_overlap": 20})
        assert isinstance(chunker, FixedSizeChunker)

    def test_build_recursive(self) -> None:
        chunker = build_chunker({"strategy": "recursive"})
        assert isinstance(chunker, RecursiveChunker)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            build_chunker({"strategy": "unknown_strategy"})

    def test_default_strategy_is_recursive(self) -> None:
        chunker = build_chunker({})
        assert isinstance(chunker, RecursiveChunker)
