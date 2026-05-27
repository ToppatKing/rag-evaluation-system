"""Tests for rag_eval.evaluation.metrics — API-free metrics only."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_eval.evaluation.dataset import EvalSample
from rag_eval.evaluation.metrics import (
    LatencyMetric,
    RougeLMetric,
    TokenEfficiencyMetric,
)


def _make_response(
    answer: str,
    contexts: list[str] | None = None,
    latency_s: float = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        answer=answer,
        contexts=contexts or ["Some context text about machine learning."],
        latency_s=latency_s,
    )


def _make_sample(
    query: str = "What is ML?",
    reference: str = "",
) -> EvalSample:
    return EvalSample(query=query, reference_answer=reference)


# ── ROUGE-L ───────────────────────────────────────────────────────────────────


class TestRougeLMetric:
    def setup_method(self) -> None:
        self.metric = RougeLMetric()

    def test_perfect_match(self) -> None:
        answer = "Machine learning enables systems to learn from data."
        sample = _make_sample(reference=answer)
        response = _make_response(answer=answer)
        result = self.metric.compute(sample, response)
        assert result.score == pytest.approx(1.0, abs=1e-3)

    def test_no_overlap_gives_low_score(self) -> None:
        sample = _make_sample(reference="The cat sat on the mat.")
        response = _make_response(answer="Quantum physics studies subatomic particles.")
        result = self.metric.compute(sample, response)
        assert result.score < 0.2

    def test_partial_overlap(self) -> None:
        sample = _make_sample(
            reference="Machine learning is a subset of artificial intelligence."
        )
        response = _make_response(
            answer="Machine learning helps build intelligent systems."
        )
        result = self.metric.compute(sample, response)
        assert 0.0 < result.score < 1.0

    def test_no_reference_returns_error(self) -> None:
        sample = _make_sample(reference="")
        response = _make_response(answer="Some answer.")
        result = self.metric.compute(sample, response)
        assert not result.ok
        assert result.score == 0.0

    def test_details_contain_precision_recall(self) -> None:
        answer = "Neural networks are used in deep learning."
        sample = _make_sample(reference=answer)
        response = _make_response(answer=answer)
        result = self.metric.compute(sample, response)
        assert "precision" in result.details
        assert "recall" in result.details

    def test_name(self) -> None:
        assert self.metric.name == "rouge_l"


# ── Latency ───────────────────────────────────────────────────────────────────


class TestLatencyMetric:
    def setup_method(self) -> None:
        self.metric = LatencyMetric()

    def test_reads_latency_from_response(self) -> None:
        response = _make_response(answer="answer", latency_s=2.35)
        result = self.metric.compute(_make_sample(), response)
        assert result.score == pytest.approx(2.35)

    def test_zero_latency(self) -> None:
        response = _make_response(answer="answer", latency_s=0.0)
        result = self.metric.compute(_make_sample(), response)
        assert result.score == 0.0

    def test_missing_latency_defaults_to_zero(self) -> None:
        response = SimpleNamespace(answer="a", contexts=["c"])
        # No latency_s attribute
        result = self.metric.compute(_make_sample(), response)
        assert result.score == 0.0

    def test_name(self) -> None:
        assert self.metric.name == "latency"


# ── Token Efficiency ──────────────────────────────────────────────────────────


class TestTokenEfficiencyMetric:
    def setup_method(self) -> None:
        self.metric = TokenEfficiencyMetric()

    def test_efficient_answer(self) -> None:
        # Short answer, long context → low ratio
        response = _make_response(
            answer="ML learns from data.",
            contexts=["A very long context " * 50],
        )
        result = self.metric.compute(_make_sample(), response)
        assert result.score < 0.1

    def test_verbose_answer(self) -> None:
        # Long answer, short context → high ratio (can exceed 1.0)
        response = _make_response(
            answer="word " * 100,
            contexts=["short"],
        )
        result = self.metric.compute(_make_sample(), response)
        assert result.score > 1.0

    def test_empty_context_gives_zero(self) -> None:
        response = _make_response(answer="answer", contexts=[])
        result = self.metric.compute(_make_sample(), response)
        assert result.score == 0.0

    def test_details_contain_token_counts(self) -> None:
        response = _make_response(answer="hello world", contexts=["a b c d"])
        result = self.metric.compute(_make_sample(), response)
        assert "answer_tokens" in result.details
        assert "context_tokens" in result.details
        assert result.details["answer_tokens"] == 2
        assert result.details["context_tokens"] == 4

    def test_name(self) -> None:
        assert self.metric.name == "token_efficiency"


# ── MetricResult ──────────────────────────────────────────────────────────────


class TestMetricResult:
    def test_ok_true_when_no_error(self) -> None:
        from rag_eval.evaluation.metrics import MetricResult

        r = MetricResult(name="test", score=0.9)
        assert r.ok is True

    def test_ok_false_when_error(self) -> None:
        from rag_eval.evaluation.metrics import MetricResult

        r = MetricResult(name="test", score=0.0, error="Something broke")
        assert r.ok is False
