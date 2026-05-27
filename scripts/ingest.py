#!/usr/bin/env python
"""Ingest documents into the RAG pipeline vector index.

Loads all supported documents from a directory, chunks them, embeds them,
and writes a persistent FAISS index ready for querying.

Usage::

    python scripts/ingest.py --docs data/sample_docs/ --config config/config.yaml
    python scripts/ingest.py --docs data/sample_docs/ --clear   # wipe and re-index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the src package importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.rule import Rule

_console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index")
    parser.add_argument(
        "--docs",
        default="data/sample_docs",
        help="Directory of documents to ingest (default: data/sample_docs)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to pipeline configuration YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete the existing index before ingesting (full re-index)",
    )
    args = parser.parse_args()

    _console.print(Rule("[bold blue]RAG Document Ingestion[/bold blue]"))

    # ── Load config ───────────────────────────────────────────────────────────
    import yaml

    config_path = Path(args.config)
    if not config_path.exists():
        _console.print(f"[red]Config file not found:[/red] {config_path}")
        sys.exit(1)

    cfg = yaml.safe_load(config_path.read_text())
    index_path = Path(cfg.get("vector_store", {}).get("index_path", "data/faiss_index"))

    # ── Optionally clear existing index ───────────────────────────────────────
    if args.clear:
        for suffix in (".faiss", ".json"):
            p = index_path.with_suffix(suffix)
            if p.exists():
                p.unlink()
                _console.print(f"[yellow]Deleted existing index file:[/yellow] {p}")

    # ── Build pipeline ────────────────────────────────────────────────────────
    from rag_eval.pipeline import RAGPipeline

    _console.print(f"[bold]Loading pipeline from[/bold] {args.config}...")
    pipeline = RAGPipeline.from_config(args.config)
    _console.print(
        f"[dim]Existing index size: {pipeline.index_size} vectors[/dim]"
    )

    # ── Ingest ────────────────────────────────────────────────────────────────
    docs_dir = Path(args.docs)
    if not docs_dir.exists():
        _console.print(f"[red]Documents directory not found:[/red] {docs_dir}")
        _console.print(
            "[dim]Tip: create sample documents in data/sample_docs/ "
            "or point --docs at an existing directory.[/dim]"
        )
        sys.exit(1)

    _console.print(f"[bold]Ingesting documents from[/bold] {docs_dir}...")
    n_chunks = pipeline.ingest_directory(docs_dir, save=True)

    _console.print(
        f"\n[green]✓ Ingested {n_chunks} chunks[/green] "
        f"(index now has {pipeline.index_size} vectors)"
    )
    _console.print(
        f"[dim]Index saved to: "
        f"{index_path.parent}/[/dim]"
    )
    _console.print(Rule("[dim]Done[/dim]"))


if __name__ == "__main__":
    main()
