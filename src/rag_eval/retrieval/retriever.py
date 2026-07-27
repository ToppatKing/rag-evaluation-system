"""


Changes vs. baseline 
  - Added HyDERetriever (lines marked ← NEW)

The retriever is chosen in config.yaml:
    retrieval:
      method: dense | mmr | hyde      ← hyde is new
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rag_eval.retrieval.embedder import BaseEmbedder
    from rag_eval.retrieval.vector_store import FAISSVectorStore
    from rag_eval.generation.generator import BaseGenerator

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Shared data type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """Represents a single chunk returned by any retriever."""
    text: str
    doc_id: str
    score: float
    metadata: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Base class                         
# ──────────────────────────────────────────────────────────────────────────────

class BaseRetriever(ABC):
    """All retrievers implement this interface."""

    @abstractmethod
    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Return the top-k most relevant chunks for *query*."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short label used in evaluation reports."""


# ──────────────────────────────────────────────────────────────────────────────
# Dense retriever 
# ──────────────────────────────────────────────────────────────────────────────

class DenseRetriever(BaseRetriever):
    """Standard bi-encoder dense retrieval.

    Embeds the raw query string and does an exact cosine/inner-product search
    against the FAISS index.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: BaseEmbedder,
        top_k: int = 5,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    @property
    def name(self) -> str:
        return "dense"

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        query_vec = self.embedder.embed_query(query)
        return self.vector_store.search(query_vec, top_k=self.top_k)


# ──────────────────────────────────────────────────────────────────────────────
# MMR retriever                         
# ──────────────────────────────────────────────────────────────────────────────

class MMRRetriever(BaseRetriever):
    """Maximal Marginal Relevance retrieval.

    Retrieves a candidate pool and then iteratively selects chunks that
    balance relevance to the query with diversity from already-selected chunks.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: BaseEmbedder,
        top_k: int = 5,
        fetch_k: int = 20,
        mmr_lambda: float = 0.5,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.fetch_k = fetch_k          # candidate pool size before MMR re-ranking
        self.mmr_lambda = mmr_lambda    # λ=1.0 → pure relevance; λ=0.0 → pure diversity

    @property
    def name(self) -> str:
        return "mmr"

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        query_vec = self.embedder.embed_query(query)
        candidates = self.vector_store.search(query_vec, top_k=self.fetch_k)
        return self._mmr_rerank(query_vec, candidates)

    # ------------------------------------------------------------------
    def _mmr_rerank(
        self,
        query_vec: np.ndarray,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Greedy MMR selection from *candidates*."""
        if not candidates:
            return []

        # Build a matrix of candidate embeddings (re-embed the texts).
        # In practice the vector store may cache these; here we re-embed
        # to keep the implementation self-contained.
        candidate_texts = [c.text for c in candidates]
        candidate_vecs = self.embedder.embed(candidate_texts)  # (n, d)
        candidate_vecs = np.array(candidate_vecs)

        # Normalise for cosine similarity
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        c_norms = candidate_vecs / (
            np.linalg.norm(candidate_vecs, axis=1, keepdims=True) + 1e-10
        )

        selected_indices: list[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(min(self.top_k, len(candidates))):
            if not remaining:
                break

            # Relevance scores: cosine(query, candidate_i)
            rel_scores = c_norms[remaining] @ q_norm  # (|remaining|,)

            if not selected_indices:
                # First pick: purely most-relevant
                best_local = int(np.argmax(rel_scores))
            else:
                # Diversity term: max cosine(selected_j, candidate_i)
                sel_vecs = c_norms[selected_indices]  # (|selected|, d)
                div_scores = (c_norms[remaining] @ sel_vecs.T).max(axis=1)
                mmr_scores = self.mmr_lambda * rel_scores - (1 - self.mmr_lambda) * div_scores
                best_local = int(np.argmax(mmr_scores))

            chosen_global = remaining[best_local]
            selected_indices.append(chosen_global)
            remaining.pop(best_local)

        return [candidates[i] for i in selected_indices]


# ──────────────────────────────────────────────────────────────────────────────
# ← NEW: HyDE retriever
# ──────────────────────────────────────────────────────────────────────────────

# WHAT IS HyDE?
# ─────────────
# Gao et al. (2022) "Precise Zero-Shot Dense Retrieval without Relevance Labels."
# arXiv:2212.10496
#
# The key insight: instead of embedding the raw query (which may live in a very
# different part of the embedding space from the relevant passages), we ask an
# instruction-following LLM to *write a hypothetical document* that would answer
# the query.  We then embed *that hypothetical document* and search for real
# passages near it.
#
# The hypothetical document can be factually wrong — the paper explicitly shows
# this (their Korean fire-use example is off by ~6.5 million years) — because
# the embedding acts as a lossy compressor: factual errors are filtered out,
# leaving only the topical/semantic signal that navigates to the right
# neighbourhood in the vector space.
#
# Formally (from the paper, Eqs. 4–7):
#
#   d̂_k ~ g(q, INST)                   — sample a hypothetical doc from the LLM
#   v̂_q = (1/N) Σ f(d̂_k)              — average N embeddings
#   results = argmax_{d ∈ D} <v̂_q, f(d)>  — standard MIPS
#
# Here we use N=1 by default (single hypothetical doc) matching what the paper
# found to be the sweet spot for practical latency.  N is configurable.

class HyDERetriever(BaseRetriever):
    """Hypothetical Document Embeddings (HyDE) retriever.

    For each query:
        1. Prompt the LLM to write a *hypothetical* passage that would answer
           the query (the passage may be factually wrong — that is fine).
        2. Embed the hypothetical passage with the same encoder used for documents.
        3. Search FAISS with the hypothetical-passage embedding rather than the
           query embedding.

    Parameters
    ----------
    vector_store : FAISSVectorStore
        The same FAISS index used by Dense and MMR retrievers.
    embedder : BaseEmbedder
        The same text encoder used during ingestion.
    generator : BaseGenerator
        An LLM wrapper (OpenAI / Anthropic).  HyDERetriever only calls its
        `generate_hypothetical_doc(query, instruction)` helper (see generator.py).
    top_k : int
        Number of real passages to return.
    n_hypothetical : int
        Number of independent hypothetical docs to sample and average.
        Paper default: 1.  Higher values smooth out LLM variance at extra cost.
    domain_instruction : str | None
        Task-specific instruction prepended to the query.  If None, a generic
        instruction is used.  For CUAD (legal contracts) this should be set
        explicitly — see the default in config.yaml.
    """

    # Generic instruction used when no domain-specific one is configured.
    _DEFAULT_INSTRUCTION = (
        "Please write a passage that directly and fully answers the following "
        "question. The passage should be factual, dense, and written as if it "
        "were extracted from a relevant document."
    )

    # Domain-specific instruction for legal contract corpora (CUAD).
    LEGAL_INSTRUCTION = (
        "Please write a legal contract clause that directly answers the "
        "following question. Write it as if it were a clause appearing in an "
        "SEC-filed commercial agreement. Be specific about parties, dates, and "
        "obligations where relevant."
    )

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: BaseEmbedder,
        generator: BaseGenerator,
        top_k: int = 5,
        n_hypothetical: int = 1,
        domain_instruction: str | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.generator = generator
        self.top_k = top_k
        self.n_hypothetical = n_hypothetical
        self.instruction = domain_instruction or self._DEFAULT_INSTRUCTION

    @property
    def name(self) -> str:
        return "hyde"

    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Full HyDE pipeline: generate → embed → search."""
        t0 = time.perf_counter()

        # ── Step 1: generate N hypothetical documents ──────────────────
        hyp_docs = []
        for i in range(self.n_hypothetical):
            try:
                hyp_doc = self.generator.generate_hypothetical_doc(
                    query=query,
                    instruction=self.instruction,
                )
                hyp_docs.append(hyp_doc)
                logger.debug(
                    "HyDE [%d/%d]: generated %d-char hypothetical doc",
                    i + 1, self.n_hypothetical, len(hyp_doc),
                )
            except Exception as exc:
                logger.warning("HyDE: LLM call %d failed (%s); skipping.", i + 1, exc)

        if not hyp_docs:
            # Graceful degradation: fall back to dense retrieval if all LLM
            # calls failed (e.g. rate-limit during experiments).
            logger.warning(
                "HyDE: all hypothetical-doc generations failed; "
                "falling back to dense retrieval for this query."
            )
            query_vec = self.embedder.embed_query(query)
            return self.vector_store.search(query_vec, top_k=self.top_k)

        # ── Step 2: embed each hypothetical doc and average ────────────
        hyp_vecs = self.embedder.embed(hyp_docs)   # list of (d,) arrays
        hyp_vecs = np.array(hyp_vecs)              # (N, d)
        mean_vec = hyp_vecs.mean(axis=0)           # (d,) — Eq. 7 in HyDE paper

        # ── Step 3: MIPS against the real index ───────────────────────
        results = self.vector_store.search(mean_vec, top_k=self.top_k)

        elapsed = time.perf_counter() - t0
        logger.debug(
            "HyDE: retrieved %d chunks in %.3fs (incl. %d LLM call(s))",
            len(results), elapsed, len(hyp_docs),
        )
        return results

    # ------------------------------------------------------------------
    def retrieve_with_hypothesis(
        self, query: str
    ) -> tuple[list[RetrievedChunk], list[str]]:
        """Same as retrieve() but also returns the hypothetical docs.

        Useful for ablation logging: lets you inspect *what the LLM imagined*
        alongside the actual retrieved passages.

        Returns
        -------
        chunks : list[RetrievedChunk]
        hyp_docs : list[str]
            The raw hypothetical documents generated by the LLM.
        """
        hyp_docs = [
            self.generator.generate_hypothetical_doc(
                query=query,
                instruction=self.instruction,
            )
            for _ in range(self.n_hypothetical)
        ]
        hyp_vecs = np.array(self.embedder.embed(hyp_docs))
        mean_vec = hyp_vecs.mean(axis=0)
        chunks = self.vector_store.search(mean_vec, top_k=self.top_k)
        return chunks, hyp_docs


# ──────────────────────────────────────────────────────────────────────────────
# Factory  (updated to include hyde)
# ──────────────────────────────────────────────────────────────────────────────

def build_retriever(
    method: str,
    vector_store: FAISSVectorStore,
    embedder: BaseEmbedder,
    generator: BaseGenerator | None = None,   # ← only required for hyde
    top_k: int = 5,
    **kwargs,
) -> BaseRetriever:
    """Instantiate the right retriever from a config string.

    Parameters
    ----------
    method : {"dense", "mmr", "hyde"}
    vector_store, embedder : required for all methods
    generator : required only for "hyde"
    top_k : number of passages to return
    **kwargs : forwarded to the chosen retriever's __init__
        e.g. mmr_lambda=0.5, n_hypothetical=1, domain_instruction="..."
    """
    method = method.lower().strip()

    if method == "dense":
        return DenseRetriever(
            vector_store=vector_store,
            embedder=embedder,
            top_k=top_k,
        )

    if method == "mmr":
        return MMRRetriever(
            vector_store=vector_store,
            embedder=embedder,
            top_k=top_k,
            fetch_k=kwargs.get("fetch_k", top_k * 4),
            mmr_lambda=kwargs.get("mmr_lambda", 0.5),
        )

    if method == "hyde":                                            # ← NEW
        if generator is None:
            raise ValueError(
                "HyDERetriever requires a generator (LLM).  "
                "Pass generator= to build_retriever() or set "
                "retrieval.method: dense in config.yaml if you don't "
                "have an LLM configured."
            )
        return HyDERetriever(
            vector_store=vector_store,
            embedder=embedder,
            generator=generator,
            top_k=top_k,
            n_hypothetical=kwargs.get("n_hypothetical", 1),
            domain_instruction=kwargs.get("domain_instruction", None),
        )

    raise ValueError(
        f"Unknown retrieval method '{method}'. "
        "Choose from: dense | mmr | hyde"
    )
