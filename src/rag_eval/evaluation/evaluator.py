"""Evaluation orchestrator: runs metrics across a dataset and produces reports.

Usage example::

    evaluator = RAGEvaluator(metrics, pipeline)
    report = evaluator.evaluate(dataset)
    report.print_summary()
    report.to_csv("results/eval.csv")
"""

from __future__ import annotations

import google.generativeai as genai
import os


import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from rag_eval.evaluation.dataset import EvalDataset, EvalSample
from rag_eval.evaluation.metrics import BaseMetric, MetricResult

logger = logging.getLogger(__name__)
_console = Console()

import re

def extract_score(text: str) -> float:
    """
    Extract a float score (0–1) from Gemini judge output.
    Falls back to 0.0 if no score is found.
    """
    match = re.search(r"([01](?:\.\d+)?)", text)
    return float(match.group(1)) if match else 0.0


class GeminiJudge:
    """
    LLM-as-judge using Gemini 1.5 Flash (free).
    """

    _FALLBACK_MODELS = ("gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash")

    def __init__(self, model_name="gemini-1.5-flash"):
        self.model_name = model_name
        self.model = None
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model_name)
            except Exception as exc:  # pragma: no cover - runtime fallback
                logger.warning("Gemini judge initialization failed (%s); using fallback", exc)

    def _call_model(self, prompt: str):
        if self.model is None:
            raise RuntimeError("Gemini judge model not available")

        last_exc: Exception | None = None
        candidates = [self.model_name] + [model for model in self._FALLBACK_MODELS if model != self.model_name]
        for candidate in candidates:
            try:
                return genai.GenerativeModel(candidate).generate_content(prompt)
            except Exception as exc:  # pragma: no cover - runtime fallback
                last_exc = exc
                logger.debug("Gemini judge model %s failed: %s", candidate, exc)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No Gemini judge models available")

    def judge(self, question: str, answer: str, context: str) -> str:
        prompt = f"""
You are an evaluation judge.
Question: {question}
Answer: {answer}
Context: {context}

Evaluate correctness, faithfulness, and relevance.
Respond with a score from 0 to 1 and a short explanation.
"""
        if self.model is None:
            return "0.0 No judge available."

        try:
            response = self._call_model(prompt)
            return getattr(response, "text", "") or ""
        except Exception as exc:  # pragma: no cover - runtime fallback
            logger.info("Gemini judge unavailable (%s); using fallback", exc)
            return "0.0 No judge available."

# ── Per-query result ──────────────────────────────────────────────────────────


@dataclass
class QueryResult:
    """Evaluation results for a single query.

    Attributes:
        sample: The source :class:`EvalSample`.
        response: The RAG system's response object.
        metric_results: Computed :class:`MetricResult` objects keyed by name.
    """

    sample: EvalSample
    response: Any  # RAGResponse
    metric_results: dict[str, MetricResult] = field(default_factory=dict)

    def score(self, metric_name: str) -> float | None:
        r = self.metric_results.get(metric_name)
        return r.score if r and r.ok else None

    def to_dict(self) -> dict[str, Any]:
        scores = {k: v.score for k, v in self.metric_results.items() if v.ok}
        return {
            "sample_id": self.sample.sample_id,
            "query": self.sample.query,
            "answer": self.response.answer,
            "contexts_used": len(self.response.contexts),
            "latency_s": getattr(self.response, "latency_s", 0.0),
            **scores,
        }


# ── Aggregate report ──────────────────────────────────────────────────────────


@dataclass
class EvaluationReport:
    """Aggregate results across all queries.

    Attributes:
        query_results: Per-query breakdown.
        dataset_name: Name of the evaluated dataset.
        elapsed_s: Wall-clock time for the full evaluation run.
    """

    query_results: list[QueryResult]
    dataset_name: str = ""
    elapsed_s: float = 0.0

    def _scores_for(self, metric_name: str) -> list[float]:
        return [
            qr.score(metric_name)
            for qr in self.query_results
            if qr.score(metric_name) is not None
        ]

    def aggregate(self) -> dict[str, dict[str, float]]:
        """Return mean ± std for each metric."""
        all_metrics: set[str] = set()
        for qr in self.query_results:
            all_metrics.update(qr.metric_results.keys())

        agg: dict[str, dict[str, float]] = {}
        for metric in sorted(all_metrics):
            scores = self._scores_for(metric)
            if scores:
                agg[metric] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "n": len(scores),
                }
        return agg

    # ── Output methods ────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a Rich-formatted summary table to the console."""
        agg = self.aggregate()
        n = len(self.query_results)

        title = f"RAG Evaluation Report — {n} queries"
        if self.dataset_name:
            title += f" | dataset: {self.dataset_name}"

        table = Table(
            title=title,
            show_header=True,
            header_style="bold cyan",
            border_style="bright_blue",
        )
        table.add_column("Metric", style="bold white", min_width=22)
        table.add_column("Mean", justify="right", style="green")
        table.add_column("Std", justify="right", style="yellow")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("N", justify="right")

        for metric, stats in agg.items():
            bar = _make_bar(stats["mean"])
            display_mean = (
                f"{stats['mean']:.3f}"
                if metric != "latency"
                else f"{stats['mean']:.2f}s"
            )
            table.add_row(
                f"{metric}  {bar}",
                display_mean,
                f"{stats['std']:.3f}",
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}",
                str(int(stats["n"])),
            )

        _console.print()
        _console.print(table)
        _console.print(f"\n[dim]Total evaluation time: {self.elapsed_s:.1f}s[/dim]")

    def to_csv(self, path: str | Path) -> None:
        """Write per-query results to a CSV file."""
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Install pandas: pip install pandas") from exc

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [qr.to_dict() for qr in self.query_results]
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        logger.info("Saved per-query CSV to %s", path)

    def to_json(self, path: str | Path) -> None:
        """Write full results (per-query + aggregate) to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dataset": self.dataset_name,
            "elapsed_s": self.elapsed_s,
            "aggregate": self.aggregate(),
            "per_query": [qr.to_dict() for qr in self.query_results],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info("Saved JSON report to %s", path)

    def save_all(self, output_dir: str | Path, formats: list[str] | None = None) -> None:
        """Save reports in all requested formats.

        Args:
            output_dir: Directory to write files into.
            formats: Subset of ``["console", "csv", "json"]``.
                Defaults to all three.
        """
        formats = formats or ["console", "csv", "json"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if "console" in formats:
            self.print_summary()
        if "csv" in formats:
            self.to_csv(output_dir / "results.csv")
        if "json" in formats:
            self.to_json(output_dir / "results.json")


# ── Evaluator ─────────────────────────────────────────────────────────────────


class RAGEvaluator:
    """Runs a RAG pipeline over an evaluation dataset and computes metrics.

    Args:
        metrics: List of :class:`~rag_eval.evaluation.metrics.BaseMetric` instances.
        pipeline: The :class:`~rag_eval.pipeline.RAGPipeline` to evaluate.
        verbose: Whether to print per-query progress.
    """

    def __init__(
        self,
        metrics: list[BaseMetric],
        pipeline: Any,  # RAGPipeline
        *,
        verbose: bool = True,
    ) -> None:
        self._metrics = metrics
        self._pipeline = pipeline
        self._verbose = verbose

    def evaluate(
        self,
        dataset: EvalDataset,
        *,
        max_samples: int | None = None,
    ) -> EvaluationReport:
        """Evaluate the pipeline on *dataset*.

        Args:
            dataset: :class:`EvalDataset` to evaluate over.
            max_samples: Truncate the dataset to this many samples (for quick
                smoke-tests).

        Returns:
            :class:`EvaluationReport` with per-query and aggregate results.
        """
        samples = list(dataset)
        if max_samples is not None:
            samples = samples[:max_samples]

        query_results: list[QueryResult] = []
        t_start = time.perf_counter()

        for i, sample in enumerate(samples, 1):
            if self._verbose:
                _console.print(
                    f"[dim][{i}/{len(samples)}][/dim] Evaluating: "
                    f"[italic]{sample.query[:70]}...[/italic]"
                )

            response = self._pipeline.query(sample.query)
            metric_results: dict[str, MetricResult] = {}
            # Run Gemini judge
            judge = GeminiJudge()
            judge_output = judge.judge(
                sample.query,
                response.answer,
                "\n\n".join(response.contexts)
            )
            
            # Parse score from Gemini output
            score = extract_score(judge_output)
            
            metric_results["gemini_judge"] = MetricResult(
                name="gemini_judge",
                score=score,
                error="" if score is not None else "No judge available",
            )
            for metric in self._metrics:
                try:
                    result = metric.compute(sample, response)
                    metric_results[metric.name] = result
                    if self._verbose:
                        status = f"{result.score:.3f}" if result.ok else f"ERR: {result.error}"
                        _console.print(
                            f"    [dim]{metric.name}[/dim]: [cyan]{status}[/cyan]"
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Metric %s failed: %s", metric.name, exc)
                    metric_results[metric.name] = MetricResult(
                        name=metric.name, score=0.0, error=str(exc)
                    )

            query_results.append(
                QueryResult(sample=sample, response=response, metric_results=metric_results)
            )

        elapsed = time.perf_counter() - t_start
        return EvaluationReport(
            query_results=query_results,
            dataset_name=dataset.name,
            elapsed_s=elapsed,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_bar(score: float, width: int = 10) -> str:
    """Return a Unicode progress bar string for *score* ∈ [0, 1]."""
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)
