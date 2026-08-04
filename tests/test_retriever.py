"""Tests for rag_eval.retrieval.vector_store and retriever."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rag_eval.retrieval.retriever import (
    DenseRetriever,
    EnsembleRetriever,
    HyDERetriever,
    MMRRetriever,
)
from rag_eval.retrieval.vector_store import FAISSVectorStore, StoredChunk


# ── FAISSVectorStore tests ────────────────────────────────────────────────────


@pytest.fixture
def small_store(stored_chunks) -> FAISSVectorStore:
    """A 4-dimensional store pre-populated with 5 chunks."""
    dim = 4
    store = FAISSVectorStore(dimension=dim, metric="cosine")
    rng = np.random.default_rng(seed=42)
    vecs = rng.standard_normal((len(stored_chunks), dim)).astype(np.float32)
    # L2-normalise so cosine = dot product
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    store.add(vecs, stored_chunks)
    return store


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls = []

    def embed_query(self, query: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class TestFAISSVectorStore:
    def test_size_after_add(self, stored_chunks) -> None:
        store = FAISSVectorStore(dimension=4)
        vecs = np.random.default_rng(0).standard_normal(
            (len(stored_chunks), 4)
        ).astype(np.float32)
        store.add(vecs, stored_chunks)
        assert store.size == len(stored_chunks)

    def test_search_returns_top_k(self, small_store) -> None:
        qv = np.random.default_rng(1).standard_normal(4).astype(np.float32)
        results = small_store.search(qv, top_k=3)
        assert len(results) == 3

    def test_search_sorted_by_score(self, small_store) -> None:
        qv = np.ones(4, dtype=np.float32)
        results = small_store.search(qv, top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_store_returns_empty(self) -> None:
        store = FAISSVectorStore(dimension=8)
        qv = np.zeros(8, dtype=np.float32)
        assert store.search(qv) == []

    def test_top_k_clamped_to_store_size(self, small_store) -> None:
        qv = np.ones(4, dtype=np.float32)
        results = small_store.search(qv, top_k=100)
        assert len(results) <= small_store.size

    def test_shape_mismatch_raises(self, stored_chunks) -> None:
        store = FAISSVectorStore(dimension=4)
        bad_vecs = np.ones((len(stored_chunks), 8), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            store.add(bad_vecs, stored_chunks)

    def test_length_mismatch_raises(self) -> None:
        store = FAISSVectorStore(dimension=4)
        vecs = np.ones((3, 4), dtype=np.float32)
        chunks = [StoredChunk(text="a", source="s")] * 5
        with pytest.raises(ValueError, match="must match"):
            store.add(vecs, chunks)

    def test_clear_resets_size(self, small_store) -> None:
        assert small_store.size > 0
        small_store.clear()
        assert small_store.size == 0

    def test_persist_and_reload(self, small_store) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index"
            small_store.save(path)
            assert path.with_suffix(".faiss").exists()
            assert path.with_suffix(".json").exists()

            loaded = FAISSVectorStore.load(path)
            assert loaded.size == small_store.size
            assert loaded.dimension == small_store.dimension

    def test_reload_search_gives_same_results(self, small_store) -> None:
        qv = np.ones(4, dtype=np.float32)
        original_results = small_store.search(qv, top_k=3)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index"
            small_store.save(path)
            loaded = FAISSVectorStore.load(path)
            loaded_results = loaded.search(qv, top_k=3)

        orig_texts = [c.text for c, _ in original_results]
        load_texts = [c.text for c, _ in loaded_results]
        assert orig_texts == load_texts

    def test_l2_metric(self) -> None:
        # Use exactly 4 chunks to match np.eye(4) which has 4 rows
        chunks = [
            StoredChunk(text=f"chunk {i}", source="test.txt", chunk_index=i)
            for i in range(4)
        ]
        store = FAISSVectorStore(dimension=4, metric="l2")
        vecs = np.eye(4, dtype=np.float32)
        store.add(vecs, chunks)
        qv = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(qv, top_k=1)
        assert len(results) == 1
        # Closest vector is the identical one (distance 0)
        assert results[0][1] == pytest.approx(0.0, abs=1e-5)

    def test_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported metric"):
            FAISSVectorStore(dimension=4, metric="dot")

    def test_mmr_retriever_handles_tuple_candidates(self) -> None:
        store = FAISSVectorStore(dimension=2)
        chunks = [
            StoredChunk(text="one", source="a"),
            StoredChunk(text="two", source="b"),
        ]
        store.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), chunks)

        retriever = MMRRetriever(store, _FakeEmbedder(), top_k=1, fetch_k=2, mmr_lambda=0.5)
        results = retriever.retrieve("hello")

        assert len(results) == 1
        assert results[0].text in {"one", "two"}

    def test_ensemble_retriever_runs_all_strategies_and_builds_comparison(self) -> None:
        store = FAISSVectorStore(dimension=2)
        chunks = [
            StoredChunk(text="one", source="a"),
            StoredChunk(text="two", source="b"),
        ]
        store.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), chunks)

        class _FakeGenerator:
            def generate_hypothetical_doc(self, query: str, instruction: str) -> str:
                return f"hypothesis for {query}"

        retriever = EnsembleRetriever(
            [
                DenseRetriever(store, _FakeEmbedder(), top_k=1),
                MMRRetriever(store, _FakeEmbedder(), top_k=1, fetch_k=2, mmr_lambda=0.5),
                HyDERetriever(store, _FakeEmbedder(), _FakeGenerator(), top_k=1),
            ],
            top_k=2,
        )

        results = retriever.retrieve("hello")

        assert len(results) >= 1
        assert retriever.last_breakdown["dense"]
        assert retriever.last_breakdown["mmr"]
        assert retriever.last_breakdown["hyde"]
        assert "dense" in retriever.last_comparison_summary.lower()
        assert "mmr" in retriever.last_comparison_summary.lower()
        assert "hyde" in retriever.last_comparison_summary.lower()
