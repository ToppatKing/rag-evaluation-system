#!/usr/bin/env python
"""Interactive query demo for the RAG pipeline.

Usage::

    python scripts/run_demo.py --config config/config.yaml \\
                                --query "What is the attention mechanism?"

If --query is omitted, a few example queries are run automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the src package importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from rag_eval.pipeline import RAGPipeline

_console = Console()

_DEMO_QUERIES = [
    "What is the attention mechanism in Transformers?",
    "How does dropout prevent overfitting?",
    "What distinguishes supervised from unsupervised learning?",
]


def run_query(pipeline: RAGPipeline, query: str) -> None:
    _console.print(Rule(f"[bold cyan]Query[/bold cyan]"))
    _console.print(f"[bold white]{query}[/bold white]\n")

    response = pipeline.query(query)

    _console.print("[bold green]Answer:[/bold green]")
    _console.print(Panel(response.answer, border_style="green"))

    _console.print(f"[bold yellow]Retrieved {len(response.contexts)} context(s):[/bold yellow]")
    for i, (ctx, src) in enumerate(zip(response.contexts, response.sources), 1):
        preview = ctx[:120].replace("\n", " ") + ("..." if len(ctx) > 120 else "")
        _console.print(
            f"  [dim][{i}][/dim] [italic]{Path(src).name}[/italic]: {preview}"
        )

    _console.print(f"\n[dim]Latency: {response.latency_s:.2f}s[/dim]\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline demo")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to pipeline configuration YAML",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Single query to run; omit for built-in demo queries",
    )
    parser.add_argument(
        "--docs",
        default="data/sample_docs",
        help="Document directory to ingest (skipped if index already exists)",
    )
    args = parser.parse_args()

    _console.print(Rule("[bold blue]RAG Demo[/bold blue]"))

    pipeline = RAGPipeline.from_config(args.config)

    if pipeline.index_size == 0:
        _console.print(f"[yellow]Index is empty. Ingesting documents from {args.docs}...[/yellow]")
        n = pipeline.ingest_directory(args.docs)
        _console.print(f"[green]Ingested {n} chunks.[/green]\n")

    queries = [args.query] if args.query else _DEMO_QUERIES
    for q in queries:
        run_query(pipeline, q)

    _console.print(Rule("[dim]Done[/dim]"))


if __name__ == "__main__":
    main()
