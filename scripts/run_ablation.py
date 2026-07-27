"""
scripts/run_ablation.py
========================
Runs the full evaluation pipeline three times — once per retrieval method
(dense, mmr, hyde) — on the same eval dataset and same FAISS index, then
produces a side-by-side comparison table.
 
This is the core experiment of Proposal A ("HyDE vs Dense vs MMR Retrieval —
A Domain-Specific Ablation Study").
 
Usage
-----
    # Run all three methods:
    python scripts/run_ablation.py --config config/config.yaml
 
    # Run specific methods only:
    python scripts/run_ablation.py --config config/config.yaml --methods dense hyde
 
    # Use CUAD eval dataset:
    python scripts/run_ablation.py --config config/config.yaml \
        --dataset data/cuad_eval.json --output results/cuad_ablation/
 
Output
------
  results/ablation/
    dense_results.json         per-query breakdown for dense retrieval
    mmr_results.json           per-query breakdown for MMR retrieval
    hyde_results.json          per-query breakdown for HyDE retrieval
    ablation_summary.json      aggregated metrics for all three
    ablation_report.txt        formatted comparison table (thesis-ready)
"""
 
from __future__ import annotations
 
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from rag_eval.evaluation.dataset import EvalDataset
from rag_eval.evaluation.evaluator import RAGEvaluator
from rag_eval.evaluation.metrics import build_metrics
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
 
METHODS = ["dense", "mmr", "hyde"]
 
# Metrics reported in the comparison table
REPORTED_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "rouge_l",
    "latency_s",        # mean end-to-end latency in seconds
    "token_efficiency",
]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _load_config(path: str) -> dict:
    import yaml  # type: ignore
    with open(path) as f:
        return yaml.safe_load(f)
 
 
def _load_pipeline(config: dict, method: str):
    """Instantiate a RAGPipeline with the requested retrieval method.
 
    This injects the method at runtime so we never touch config.yaml on disk —
    all three runs share the same in-memory config object, with only
    config["retrieval"]["method"] swapped.
    """
    # Local import keeps startup fast when the library isn't installed
    from rag_eval.pipeline import RAGPipeline  # type: ignore
 
    cfg = dict(config)  # shallow copy
    cfg["retrieval"] = dict(config.get("retrieval", {}))
    
    # Override the retrieval method
    cfg["retrieval"]["method"] = method
 
    # Pass the overridden dictionary, NOT a file path
    return RAGPipeline.from_config(cfg)
 
def _run_evaluation(pipeline, dataset: EvalDataset, config: dict) -> tuple[list[dict], dict]:
    """Run the evaluation using the official RAGEvaluator."""
    eval_cfg = config.get("evaluation", {})

    # Re-use pipeline's embedder for answer relevancy to avoid loading the model twice
    metrics = build_metrics(eval_cfg, embedder=pipeline._embedder)
    evaluator = RAGEvaluator(metrics, pipeline, verbose=False)

    # Run the official evaluation framework
    report = evaluator.evaluate(dataset)

    # Extract the dictionaries from the report object
    # (Checking standard property names: usually 'results' and 'aggregate')
    per_query = getattr(report, "results", getattr(report, "samples", []))
    aggregate = getattr(report, "aggregate", getattr(report, "metrics", {}))

    # Safety conversion in case the report stores objects instead of raw dicts
    if per_query and not isinstance(per_query[0], dict):
        per_query = [vars(q) if hasattr(q, "__dict__") else q for q in per_query]

    aggregate["n_samples"] = len(per_query)
    return per_query, aggregate
 
def _format_table(results: dict[str, dict]) -> str:
    """Render a plain-text comparison table suitable for a thesis appendix."""
    col_w = 22
    metric_w = 24
 
    header = f"{'Metric':<{metric_w}}" + "".join(
        f"{m:^{col_w}}" for m in results
    )
    divider = "─" * len(header)
 
    rows = [header, divider]
    for metric in REPORTED_METRICS:
        label = metric.replace("_", " ").title()
        row = f"{label:<{metric_w}}"
        for method_results in results.values():
            val = method_results.get(metric, 0.0)
            row += f"{val:^{col_w}.4f}"
        rows.append(row)
 
    rows.append(divider)
 
    # n_samples row
    row = f"{'N Samples':<{metric_w}}"
    for method_results in results.values():
        row += f"{method_results.get('n_samples', '?'):^{col_w}}"
    rows.append(row)
 
    return "\n".join(rows)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
 
def main() -> None:
    parser = argparse.ArgumentParser(description="HyDE vs Dense vs MMR ablation.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--dataset", default=None,
        help="Path to eval JSON.  Defaults to evaluation.eval_dataset in config.",
    )
    parser.add_argument("--output", default="results/ablation/")
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=METHODS,
        help="Which retrieval methods to evaluate (default: all three).",
    )
    args = parser.parse_args()
 
    config = _load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    # Load eval dataset
    dataset_path = args.dataset or config.get("evaluation", {}).get("eval_dataset")
    if not dataset_path or not Path(dataset_path).exists():
        logger.error(
            "Eval dataset not found at '%s'.  "
            "Run setup_cuad.py first, or set evaluation.eval_dataset in config.",
            dataset_path,
        )
        sys.exit(1)
 
    dataset = EvalDataset.from_json(dataset_path)
    logger.info("Loaded %d eval samples from %s", len(dataset), dataset_path)
 
    # ── Run each method ────────────────────────────────────────────────
    all_results: dict[str, dict] = {}
 
    for method in args.methods:
        logger.info("=" * 60)
        logger.info("Running method: %s", method.upper())
        logger.info("=" * 60)
 
        pipeline = _load_pipeline(config, method)
        per_query, aggregate = _run_evaluation(pipeline, dataset, config)
 
        # Save per-query results
        per_query_path = output_dir / f"{method}_results.json"
        with open(per_query_path, "w") as f:
            json.dump(per_query, f, indent=2)
        logger.info("Per-query results → %s", per_query_path)
 
        all_results[method] = aggregate
        logger.info("Aggregate for %s: %s", method, aggregate)
 
    # ── Summary & report ───────────────────────────────────────────────
    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Summary → %s", summary_path)
 
    table = _format_table(all_results)
    report_path = output_dir / "ablation_report.txt"
    report_content = (
        "HyDE vs Dense vs MMR — Retrieval Ablation\n"
        "==========================================\n"
        f"Dataset : {dataset_path}\n"
        f"Config  : {args.config}\n\n"
        f"{table}\n\n"
        "Metrics:\n"
        "  Faithfulness      fraction of answer claims supported by context (LLM-as-judge)\n"
        "  Answer Relevancy  cosine similarity between question and answer embeddings\n"
        "  Context Precision fraction of retrieved chunks that are relevant (LLM-as-judge)\n"
        "  Context Recall    coverage of gold answer in retrieved context (LLM-as-judge)\n"
        "  Rouge L           longest-common-subsequence F1 vs. reference answer\n"
        "  Latency S         mean end-to-end wall-clock time (seconds)\n"
        "  Token Efficiency  answer tokens / context tokens (conciseness)\n"
    )
    report_path.write_text(report_content, encoding="utf-8")
 
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION COMPLETE")
    logger.info("=" * 60)
    print("\n" + table + "\n")
    logger.info("Full report → %s", report_path)
 
 
if __name__ == "__main__":
    main()
 