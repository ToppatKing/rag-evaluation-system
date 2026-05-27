"""Evaluation metrics for the RAG pipeline.

Seven metrics are implemented:

+---------------------+------------------+---------------------------------------+
| Metric              | Type             | Description                           |
+=====================+==================+=======================================+
| Faithfulness        | LLM-as-judge     | Claims in answer supported by context |
| AnswerRelevancy     | Embedding cosine | Semantic sim(question, answer)        |
| ContextPrecision    | LLM-as-judge     | Retrieved chunks that are relevant    |
| ContextRecall       | LLM-as-judge     | Ground-truth info covered by context  |
| RougeL              | String overlap   | Longest-common-subsequence F1         |
| Latency             | Timing           | End-to-end query latency              |
| TokenEfficiency     | Ratio            | Answer tokens / context tokens        |
+---------------------+------------------+---------------------------------------+

All metrics share the :class:`BaseMetric` interface:
    ``metric.compute(sample, response) -> MetricResult``
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class MetricResult:
    """Result from a single metric computation.

    Attributes:
        name: Metric identifier.
        score: Scalar score in ``[0, 1]`` (or seconds for Latency).
        details: Optional per-component breakdown.
        error: Error message if computation failed.
    """

    name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


# ── Imports that may be needed ────────────────────────────────────────────────


def _lazy_openai():  # type: ignore[no-untyped-def]
    try:
        from openai import OpenAI  # type: ignore[import-untyped]
        return OpenAI
    except ImportError as exc:
        raise ImportError("Install openai: pip install openai") from exc


# ── Abstract base ─────────────────────────────────────────────────────────────


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric identifier."""

    @abstractmethod
    def compute(
        self,
        sample: "Any",        # EvalSample
        response: "Any",      # GenerationResult or RAGResponse
    ) -> MetricResult:
        """Compute the metric.

        Args:
            sample: :class:`~rag_eval.evaluation.dataset.EvalSample`
            response: :class:`~rag_eval.pipeline.RAGResponse`

        Returns:
            :class:`MetricResult`
        """


# ── LLM judge helper ──────────────────────────────────────────────────────────


class _LLMJudge:
    """Lightweight wrapper for LLM-as-judge calls (OpenAI only)."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        self._model = model
        self._temperature = temperature
        self._client = _lazy_openai()()

    def _call(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self._temperature,
            max_tokens=256,
        )
        return (resp.choices[0].message.content or "").strip()

    def score_0_to_1(self, prompt: str) -> float:
        """Call the judge and parse a float in [0, 1] from its reply."""
        reply = self._call(
            "You are an impartial evaluation judge. "
            "Respond ONLY with a single decimal number between 0 and 1.",
            prompt,
        )
        match = re.search(r"(0(\.\d+)?|1(\.0+)?)", reply)
        if match:
            return float(match.group())
        return 0.0

    def score_list(self, prompts: list[str]) -> list[float]:
        return [self.score_0_to_1(p) for p in prompts]


# ── 1. Faithfulness ───────────────────────────────────────────────────────────


class FaithfulnessMetric(BaseMetric):
    """Measures whether the answer is grounded in the retrieved context.

    Uses an LLM to decompose the answer into atomic claims, then check each
    claim against the context.  Score = supported_claims / total_claims.

    Args:
        judge_model: OpenAI model used as judge.
        judge_temperature: Sampling temperature for the judge.
    """

    def __init__(
        self, judge_model: str = "gpt-4o-mini", judge_temperature: float = 0.0
    ) -> None:
        self._judge = _LLMJudge(judge_model, judge_temperature)

    @property
    def name(self) -> str:
        return "faithfulness"

    def compute(self, sample: Any, response: Any) -> MetricResult:
        context = "\n\n".join(response.contexts)
        answer = response.answer

        if not answer.strip():
            return MetricResult(name=self.name, score=0.0, error="Empty answer")

        # Step 1: decompose answer into claims
        decompose_prompt = (
            f"Break the following answer into a list of independent, atomic claims. "
            f"Return ONLY a JSON array of strings, e.g. [\"claim1\", \"claim2\"].\n\n"
            f"Answer: {answer}"
        )
        raw = self._judge._call(
            "You extract atomic factual claims from text. Output only valid JSON.",
            decompose_prompt,
        )
        try:
            import json
            claims: list[str] = json.loads(raw)
            if not isinstance(claims, list):
                claims = [answer]
        except Exception:
            claims = [s.strip() for s in answer.split(".") if s.strip()]

        if not claims:
            return MetricResult(name=self.name, score=0.0, error="No claims extracted")

        # Step 2: verify each claim
        supported = 0
        details: dict[str, Any] = {"claims": []}
        for claim in claims:
            prompt = (
                f"Context:\n{context}\n\n"
                f"Claim: {claim}\n\n"
                f"Is this claim fully supported by the context? "
                f"Score 1.0 if yes, 0.0 if no."
            )
            s = self._judge.score_0_to_1(prompt)
            supported += s
            details["claims"].append({"claim": claim, "supported": s})

        score = supported / len(claims)
        details["num_claims"] = len(claims)
        details["supported"] = supported
        return MetricResult(name=self.name, score=score, details=details)


# ── 2. Answer Relevancy ───────────────────────────────────────────────────────


class AnswerRelevancyMetric(BaseMetric):
    """Semantic similarity between the question and the generated answer.

    Uses the cosine similarity between the embeddings of the query and
    the answer.  High scores indicate the answer stays on-topic.

    Args:
        embedder: :class:`~rag_eval.retrieval.embedder.BaseEmbedder`.
    """

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

    @property
    def name(self) -> str:
        return "answer_relevancy"

    def compute(self, sample: Any, response: Any) -> MetricResult:
        query = sample.query
        answer = response.answer
        if not answer.strip():
            return MetricResult(name=self.name, score=0.0, error="Empty answer")

        qv = self._embedder.embed_query(query)
        av = self._embedder.embed_query(answer)

        dot = float(np.dot(qv, av))
        norm = float(np.linalg.norm(qv) * np.linalg.norm(av))
        score = dot / norm if norm > 0 else 0.0
        # Clamp to [0, 1] (negative similarity → 0)
        score = max(0.0, min(1.0, score))
        return MetricResult(name=self.name, score=score)


# ── 3. Context Precision ──────────────────────────────────────────────────────


class ContextPrecisionMetric(BaseMetric):
    """Proportion of retrieved context chunks that are relevant to the query.

    Score = num_relevant_chunks / total_chunks_retrieved.

    Args:
        judge_model: OpenAI model used as judge.
    """

    def __init__(self, judge_model: str = "gpt-4o-mini") -> None:
        self._judge = _LLMJudge(judge_model)

    @property
    def name(self) -> str:
        return "context_precision"

    def compute(self, sample: Any, response: Any) -> MetricResult:
        contexts = response.contexts
        if not contexts:
            return MetricResult(name=self.name, score=0.0, error="No contexts")

        prompts = [
            f"Query: {sample.query}\n\nContext: {ctx}\n\n"
            f"Is this context relevant to answering the query? "
            f"Score 1.0 if yes, 0.0 if no."
            for ctx in contexts
        ]
        scores = self._judge.score_list(prompts)
        precision = float(np.mean(scores))
        return MetricResult(
            name=self.name,
            score=precision,
            details={"per_chunk_scores": scores, "num_chunks": len(contexts)},
        )


# ── 4. Context Recall ─────────────────────────────────────────────────────────


class ContextRecallMetric(BaseMetric):
    """Coverage of ground-truth information in the retrieved context.

    Requires ``sample.reference_answer``.  Prompts the judge to check how
    much of the reference answer is supported by the retrieved context.

    Args:
        judge_model: OpenAI model used as judge.
    """

    def __init__(self, judge_model: str = "gpt-4o-mini") -> None:
        self._judge = _LLMJudge(judge_model)

    @property
    def name(self) -> str:
        return "context_recall"

    def compute(self, sample: Any, response: Any) -> MetricResult:
        if not sample.reference_answer:
            return MetricResult(
                name=self.name, score=0.0, error="No reference answer provided"
            )

        context = "\n\n".join(response.contexts)
        prompt = (
            f"Reference Answer: {sample.reference_answer}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            f"What fraction of the information needed to produce the reference answer "
            f"is present in the retrieved context? "
            f"Score 1.0 if all information is present, 0.0 if none."
        )
        score = self._judge.score_0_to_1(prompt)
        return MetricResult(name=self.name, score=score)


# ── 5. ROUGE-L ────────────────────────────────────────────────────────────────


class RougeLMetric(BaseMetric):
    """ROUGE-L F1 score between the generated and reference answers.

    Measures the longest common subsequence overlap.  Requires
    ``sample.reference_answer``.
    """

    def __init__(self) -> None:
        try:
            from rouge_score import rouge_scorer  # type: ignore[import-untyped]
            self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        except ImportError as exc:
            raise ImportError("Install rouge-score: pip install rouge-score") from exc

    @property
    def name(self) -> str:
        return "rouge_l"

    def compute(self, sample: Any, response: Any) -> MetricResult:
        if not sample.reference_answer:
            return MetricResult(name=self.name, score=0.0, error="No reference answer")

        scores = self._scorer.score(sample.reference_answer, response.answer)
        f1 = scores["rougeL"].fmeasure
        return MetricResult(
            name=self.name,
            score=float(f1),
            details={
                "precision": scores["rougeL"].precision,
                "recall": scores["rougeL"].recall,
            },
        )


# ── 6. Latency ────────────────────────────────────────────────────────────────


class LatencyMetric(BaseMetric):
    """Records the wall-clock latency of the generation call (seconds).

    Note: this reads ``response.latency_s`` so the RAG pipeline must set it.
    """

    @property
    def name(self) -> str:
        return "latency"

    def compute(self, sample: Any, response: Any) -> MetricResult:
        latency = getattr(response, "latency_s", 0.0)
        return MetricResult(name=self.name, score=latency)


# ── 7. Token Efficiency ───────────────────────────────────────────────────────


class TokenEfficiencyMetric(BaseMetric):
    """Ratio of answer tokens to total context tokens.

    Higher values indicate more concise answers relative to context size.
    Useful for detecting overly verbose or padded answers.
    """

    @property
    def name(self) -> str:
        return "token_efficiency"

    @staticmethod
    def _rough_token_count(text: str) -> int:
        """Approximate token count (word-level split)."""
        return max(1, len(text.split()))

    def compute(self, sample: Any, response: Any) -> MetricResult:
        answer_tokens = self._rough_token_count(response.answer)
        context_tokens = sum(
            self._rough_token_count(c) for c in response.contexts
        )
        score = answer_tokens / context_tokens if context_tokens > 0 else 0.0
        return MetricResult(
            name=self.name,
            score=round(score, 4),
            details={"answer_tokens": answer_tokens, "context_tokens": context_tokens},
        )


# ── Registry ──────────────────────────────────────────────────────────────────


def build_metrics(
    config: dict[str, Any],
    embedder: Any | None = None,
) -> list[BaseMetric]:
    """Construct the list of metrics specified in *config*.

    Args:
        config: Evaluation config dict with ``metrics`` list and
            ``judge_model`` key.
        embedder: Required for :class:`AnswerRelevancyMetric`.

    Returns:
        List of configured :class:`BaseMetric` instances.
    """
    requested: list[str] = list(config.get("metrics", []))
    judge_model = str(config.get("judge_model", "gpt-4o-mini"))

    mapping: dict[str, Any] = {
        "faithfulness": lambda: FaithfulnessMetric(judge_model=judge_model),
        "answer_relevancy": lambda: AnswerRelevancyMetric(embedder),
        "context_precision": lambda: ContextPrecisionMetric(judge_model=judge_model),
        "context_recall": lambda: ContextRecallMetric(judge_model=judge_model),
        "rouge_l": RougeLMetric,
        "latency": LatencyMetric,
        "token_efficiency": TokenEfficiencyMetric,
    }

    metrics: list[BaseMetric] = []
    for name in requested:
        if name not in mapping:
            raise ValueError(f"Unknown metric: {name!r}. Available: {list(mapping)}")
        if name == "answer_relevancy" and embedder is None:
            raise ValueError("answer_relevancy metric requires an embedder")
        metrics.append(mapping[name]())
    return metrics
