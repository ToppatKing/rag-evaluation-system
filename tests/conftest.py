"""Shared pytest fixtures for the rag_eval test suite."""

from __future__ import annotations

import pytest

from rag_eval.ingestion.loader import Document
from rag_eval.retrieval.vector_store import StoredChunk


# ── Text fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_text() -> str:
    return (
        "Machine learning is a subset of artificial intelligence. "
        "It enables systems to learn from data without being explicitly programmed. "
        "Supervised learning uses labelled examples to train predictive models. "
        "Unsupervised learning discovers patterns in unlabelled data. "
        "Neural networks are inspired by the structure of the human brain. "
        "Deep learning uses multiple layers of neural networks. "
        "The attention mechanism allows models to focus on relevant parts of input. "
        "Transformers revolutionised natural language processing in 2017. "
        "BERT and GPT are pre-trained language models based on Transformers. "
        "Fine-tuning adapts a pre-trained model to a specific downstream task."
    )


@pytest.fixture
def sample_document(sample_text: str) -> Document:
    return Document(content=sample_text, source="tests/fixtures/sample.txt")


@pytest.fixture
def stored_chunks() -> list[StoredChunk]:
    return [
        StoredChunk(
            text="Machine learning enables systems to learn from data.",
            source="doc1.txt",
            chunk_index=0,
        ),
        StoredChunk(
            text="Transformers revolutionised natural language processing.",
            source="doc1.txt",
            chunk_index=1,
        ),
        StoredChunk(
            text="The attention mechanism focuses on relevant parts of the input.",
            source="doc1.txt",
            chunk_index=2,
        ),
        StoredChunk(
            text="Deep learning uses stacked neural network layers.",
            source="doc2.txt",
            chunk_index=0,
        ),
        StoredChunk(
            text="BERT is a pre-trained Transformer-based language model.",
            source="doc2.txt",
            chunk_index=1,
        ),
    ]
