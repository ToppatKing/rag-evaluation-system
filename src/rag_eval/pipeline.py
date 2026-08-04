"""End-to-end RAG pipeline.

The :class:`RAGPipeline` is the main entry point.  It wires together the
ingestion, retrieval, and generation components and exposes a single
``query`` method.

Example::

    from rag_eval.pipeline import RAGPipeline, PipelineConfig

    pipeline = RAGPipeline.from_config("config/config.yaml")
    pipeline.ingest_directory("data/sample_docs/")

    response = pipeline.query("What is self-attention?")
    print(response.answer)
    print(response.contexts)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rag_eval.generation.generator import BaseGenerator, GenerationResult, build_generator
from rag_eval.ingestion.chunker import BaseChunker, Chunk, build_chunker
from rag_eval.ingestion.loader import DocumentLoader
from rag_eval.ingestion.preprocessor import clean_text
from rag_eval.retrieval.embedder import BaseEmbedder, build_embedder
from rag_eval.retrieval.retriever import BaseRetriever, build_retriever
from rag_eval.retrieval.vector_store import FAISSVectorStore, StoredChunk

logger = logging.getLogger(__name__)


# ── Response dataclass ────────────────────────────────────────────────────────


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline.

    Attributes:
        answer: Generated answer text.
        query: The original query.
        contexts: Retrieved text passages used for generation.
        sources: Source document paths corresponding to each context.
        latency_s: Total wall-clock time (retrieval + generation).
        generation_result: Raw :class:`GenerationResult` from the LLM.
    """

    answer: str
    query: str
    contexts: list[str]
    sources: list[str]
    latency_s: float = 0.0
    generation_result: GenerationResult | None = None
    retrieval_breakdown: dict[str, list[str]] = field(default_factory=dict)
    retrieval_comparison: str = ""

    def __repr__(self) -> str:
        preview = self.answer[:80].replace("\n", " ")
        return (
            f"RAGResponse(query={self.query[:40]!r}, "
            f"contexts={len(self.contexts)}, "
            f"answer={preview!r}...)"
        )


# ── Pipeline config ───────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Typed configuration for the RAG pipeline.

    Loaded from YAML via :meth:`PipelineConfig.from_yaml`.
    """

    chunking: dict[str, Any] = field(default_factory=dict)
    embedding: dict[str, Any] = field(default_factory=dict)
    vector_store: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    ingestion: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        data = yaml.safe_load(Path(path).read_text())
        return cls(
            chunking=data.get("chunking", {}),
            embedding=data.get("embedding", {}),
            vector_store=data.get("vector_store", {}),
            retrieval=data.get("retrieval", {}),
            generation=data.get("generation", {}),
            evaluation=data.get("evaluation", {}),
            ingestion=data.get("ingestion", {}),
        )


# ── Pipeline ──────────────────────────────────────────────────────────────────


class RAGPipeline:
    """End-to-end retrieval-augmented generation pipeline.

    Components are constructed from a :class:`PipelineConfig` or injected
    directly for testing.

    Args:
        embedder: Embedding backend.
        chunker: Text chunking strategy.
        vector_store: FAISS vector store (may be empty on construction).
        retriever: Retrieval strategy.
        generator: LLM generation backend.
        config: Full pipeline configuration.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        chunker: BaseChunker,
        vector_store: FAISSVectorStore,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        config: PipelineConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._chunker = chunker
        self._store = vector_store
        self._retriever = retriever
        self._generator = generator
        self._config = config or PipelineConfig()
        self._loader = DocumentLoader()

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str | Path | dict) -> "RAGPipeline":
        """Construct a fully configured pipeline from a YAML file or a dictionary.

        If a FAISS index exists at the configured path, it is loaded
        automatically; otherwise an empty store is created.

        Args:
            config_path: Path to ``config.yaml``, or a pre-loaded configuration dict.

        Returns:
            Configured :class:`RAGPipeline`.
        """
        # Accept either a dictionary (for ablation overrides) or a file path
        if isinstance(config_path, dict):
            cfg = PipelineConfig(
                chunking=config_path.get("chunking", {}),
                embedding=config_path.get("embedding", {}),
                vector_store=config_path.get("vector_store", {}),
                retrieval=config_path.get("retrieval", {}),
                generation=config_path.get("generation", {}),
                evaluation=config_path.get("evaluation", {}),
                ingestion=config_path.get("ingestion", {})
            )
        else:
            cfg = PipelineConfig.from_yaml(config_path)

        embedder = build_embedder(cfg.embedding)
        chunker = build_chunker(cfg.chunking)

        index_path = cfg.vector_store.get("index_path", "data/faiss_index")
        metric = str(cfg.vector_store.get("metric", "cosine"))
        idx_file = Path(str(index_path)).with_suffix(".faiss")

        if idx_file.exists():
            logger.info("Loading existing FAISS index from %s", index_path)
            store = FAISSVectorStore.load(index_path)
        else:
            logger.info("Creating new FAISS index (dim=%d)", embedder.dimension)
            store = FAISSVectorStore(dimension=embedder.dimension, metric=metric)

        # Build generator FIRST so it can be passed to HyDE
        generator = build_generator(cfg.generation)
        
        # Extract the method string and remove it from the kwargs dictionary
        retriever_method = cfg.retrieval.get("method", "dense")
        retrieval_kwargs = {k: v for k, v in cfg.retrieval.items() if k != "method"}
        
        retriever = build_retriever(
            method=retriever_method, 
            vector_store=store, 
            embedder=embedder, 
            generator=generator,
            **retrieval_kwargs
        )

        return cls(embedder, chunker, store, retriever, generator, cfg)
    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_directory(self, directory: str | Path, *, save: bool = True) -> int:
        """Ingest all documents in *directory* into the vector store.

        Args:
            directory: Root directory to scan for documents.
            save: If *True* and an index path is configured, persist the
                updated index after ingestion.

        Returns:
            Number of chunks added.
        """
        directory = Path(directory)
        docs = list(self._loader.load_directory(directory))
        logger.info("Loaded %d documents from %s", len(docs), directory)
        return self._ingest_documents(docs, save=save)

    def ingest_texts(
        self,
        texts: list[str],
        sources: list[str] | None = None,
        *,
        save: bool = False,
    ) -> int:
        """Ingest raw text strings directly.

        Args:
            texts: List of text strings.
            sources: Optional source labels; defaults to ``["<inline_N>"]``.
            save: Whether to persist after ingestion.

        Returns:
            Number of chunks added.
        """
        from rag_eval.ingestion.loader import Document

        if sources is None:
            sources = [f"<inline_{i}>" for i in range(len(texts))]
        docs = [Document(content=t, source=s) for t, s in zip(texts, sources)]
        return self._ingest_documents(docs, save=save)

    def _ingest_documents(
        self,
        docs: list[Any],  # list[Document]
        *,
        save: bool = True,
    ) -> int:
        total_chunks = 0
        for doc in docs:
            cleaned = clean_text(doc.content)
            chunks: list[Chunk] = self._chunker.chunk(cleaned, source=doc.source)
            if not chunks:
                logger.warning("No chunks produced from %s", doc.source)
                continue

            texts = [c.text for c in chunks]
            vectors = self._embedder.embed_documents(texts)
            stored = [
                StoredChunk(
                    text=c.text,
                    source=c.source,
                    chunk_index=c.chunk_index,
                    metadata=c.metadata,
                )
                for c in chunks
            ]
            self._store.add(vectors, stored)
            total_chunks += len(chunks)
            logger.info(
                "Ingested %d chunks from %s (total: %d)",
                len(chunks),
                Path(doc.source).name,
                self._store.size,
            )

        if save:
            idx_path = self._config.vector_store.get("index_path")
            if idx_path:
                self._store.save(str(idx_path))

        return total_chunks

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str) -> RAGResponse:
        """Run a full RAG query: retrieve → generate.

        Args:
            question: Natural-language question.

        Returns:
            :class:`RAGResponse` with answer, contexts, and metadata.
        """
        t0 = time.perf_counter()

        # 1. Retrieve raw results (which might be objects or (item, score) tuples)
        raw_results = self._retriever.retrieve(question)

        # 2. Normalize them so we always have the underlying chunk/item object
        retrieval_results = []
        for r in raw_results:
            item = r[0] if isinstance(r, tuple) else r
            retrieval_results.append(item)

        # 3. Safely extract text and source attributes
        contexts = [
            getattr(item, "text", item.get("text", str(item))) if hasattr(item, "get") else getattr(item, "text", str(item))
            for item in retrieval_results
        ]
        sources = [
            getattr(item, "source", item.get("source", "unknown")) if hasattr(item, "get") else getattr(item, "source", "unknown")
            for item in retrieval_results
        ]
        if not contexts:
            logger.warning("No contexts retrieved for query: %s", question)
            return RAGResponse(
                answer="I could not find relevant information to answer this question.",
                query=question,
                contexts=[],
                sources=[],
                latency_s=time.perf_counter() - t0,
                retrieval_breakdown={},
                retrieval_comparison="",
            )

        breakdown: dict[str, list[str]] = {}
        comparison = ""
        if hasattr(self._retriever, "last_breakdown"):
            breakdown = {
                name: [self._serialize_chunk(chunk) for chunk in chunks]
                for name, chunks in getattr(self._retriever, "last_breakdown", {}).items()
            }
            comparison = getattr(self._retriever, "last_comparison_summary", "")

        gen_result = self._generator.generate(question, contexts)
        total_latency = time.perf_counter() - t0
        gen_result.latency_s = total_latency  # update with end-to-end time

        return RAGResponse(
            answer=gen_result.answer,
            query=question,
            contexts=contexts,
            sources=sources,
            latency_s=total_latency,
            generation_result=gen_result,
            retrieval_breakdown=breakdown,
            retrieval_comparison=comparison,
        )

    def _serialize_chunk(self, chunk: Any) -> str:
        if chunk is None:
            return ""
        if hasattr(chunk, "text"):
            return str(getattr(chunk, "text"))
        if isinstance(chunk, dict):
            return str(chunk.get("text", ""))
        return str(chunk)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_index(self, path: str | Path | None = None) -> None:
        """Save the FAISS index to *path* (or the configured path)."""
        p = str(path or self._config.vector_store.get("index_path", "data/faiss_index"))
        self._store.save(p)

    @property
    def index_size(self) -> int:
        """Number of vectors currently in the store."""
        return self._store.size