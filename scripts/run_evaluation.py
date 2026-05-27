#!/usr/bin/env python
"""Run a full evaluation benchmark and save results.

Usage::

    python scripts/run_evaluation.py \\
        --config config/config.yaml \\
        --dataset data/eval_dataset.json \\
        --output results/ \\
        --max-samples 20

If --dataset is omitted, the built-in ML fundamentals sample dataset is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.rule import Rule

from rag_eval.evaluation.dataset import EvalDataset, load_sample_dataset
from rag_eval.evaluation.evaluator import RAGEvaluator
from rag_eval.evaluation.metrics import build_metrics
from rag_eval.pipeline import RAGPipeline

_console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation benchmark")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to eval dataset JSON; omit to use built-in sample dataset",
    )
    parser.add_argument("--docs", default="data/sample_docs")
    parser.add_argument("--output", default="results")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of evaluation samples",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["console", "csv", "json"],
        choices=["console", "csv", "json"],
    )
    args = parser.parse_args()

    _console.print(Rule("[bold blue]RAG Evaluation[/bold blue]"))

    # ── Build pipeline ────────────────────────────────────────────────────────
    _console.print("[bold]Loading pipeline...[/bold]")
    pipeline = RAGPipeline.from_config(args.config)

    if pipeline.index_size == 0:
        _console.print(f"[yellow]Ingesting documents from {args.docs}...[/yellow]")
        n = pipeline.ingest_directory(args.docs)
        _console.print(f"[green]Ingested {n} chunks.[/green]")

    _console.print(f"[dim]Index size: {pipeline.index_size} vectors[/dim]\n")

    # ── Load dataset ──────────────────────────────────────────────────────────
    if args.dataset:
        _console.print(f"[bold]Loading dataset from {args.dataset}...[/bold]")
        dataset = EvalDataset.from_json(args.dataset)
    else:
        _console.print("[bold]Using built-in ML fundamentals sample dataset.[/bold]")
        dataset = load_sample_dataset()

    _console.print(f"[dim]Samples: {len(dataset)}[/dim]\n")

    # ── Build metrics ─────────────────────────────────────────────────────────
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    eval_cfg = cfg.get("evaluation", {})

    try:
        # Attempt to build embedder for answer relevancy
        from rag_eval.retrieval.embedder import build_embedder
        embedder = build_embedder(cfg.get("embedding", {}))
    except Exception:
        embedder = None

    metrics = build_metrics(eval_cfg, embedder=embedder)
    metric_names = [m.name for m in metrics]
    _console.print(f"[bold]Metrics:[/bold] {', '.join(metric_names)}\n")

    # ── Run evaluation ────────────────────────────────────────────────────────
    evaluator = RAGEvaluator(metrics, pipeline, verbose=True)
    report = evaluator.evaluate(dataset, max_samples=args.max_samples)

    # ── Save results ──────────────────────────────────────────────────────────
    report.save_all(args.output, formats=args.formats)
    _console.print(Rule("[dim]Evaluation complete[/dim]"))


if __name__ == "__main__":
    main()
