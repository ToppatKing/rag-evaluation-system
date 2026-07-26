"""
scripts/setup_cuad.py
=====================
One-shot script that:
  1. Downloads the CUAD dataset from HuggingFace (no account required).
  2. Saves each contract as a .txt file under data/cuad_contracts/.
  3. Converts CUAD's SQuAD-2.0-style QA annotations into the eval dataset
     format your pipeline already understands (data/cuad_eval.json).
  4. Runs the existing ingestion pipeline to embed and index the contracts.
 
Usage
-----
    python scripts/setup_cuad.py --config config/config.yaml
 
Requirements (add to pyproject.toml if not already present):
    datasets>=2.14          # HuggingFace datasets library
    tqdm
 
What CUAD looks like
--------------------
CUAD is released in SQuAD 2.0 format.  Each entry is:
 
  {
    "title": "MasterAgreement_Verizon_...",
    "paragraphs": [
      {
        "context": "<full contract paragraph text>",
        "qas": [
          {
            "question": "Highlight the parts (if any) of this clause related
                         to 'Governing Law'. Details: Which state/country's
                         law governs the interpretation of the contract?",
            "id": "...",
            "answers": [
              {"text": "This Agreement shall be governed by the laws of
                         the State of Delaware.", "answer_start": 4521}
            ],
            "is_impossible": false
          }
        ]
      }
    ]
  }
 
We convert each answerable QA pair into an EvalSample:
  - question:          natural-language question derived from the CUAD question
  - reference_answer:  the extracted clause text (the gold answer span)
  - doc_id:            the contract's title (links back to the indexed document)
  - metadata:          cuad category label, is_impossible flag, etc.
 
Note on "training"
------------------
RAG doesn't train a model — there are no gradient updates here.  "Training the
RAG on CUAD" means:
  (a) populating the FAISS vector store with CUAD contract text (knowledge base)
  (b) building a ground-truth eval set from CUAD's expert annotations
 
If you later want to fine-tune the *embedding model* on CUAD's QA pairs as hard
negatives (a natural extension of this thesis), that is a separate step not
covered by this script.
"""
 
from __future__ import annotations
 
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _parse_category(cuad_question: str) -> str:
    """Extract the CUAD category name from the verbose question string.
 
    CUAD question format:
      'Highlight the parts (if any) of this clause related to "Governing Law".
       Details: Which state/country's law governs ...'
 
    Returns 'Governing Law'.
    """
    match = re.search(r'related to ["\u201c]([^"\u201d]+)["\u201d]', cuad_question)
    if match:
        return match.group(1).strip()
    # Fallback: use first sentence
    return cuad_question.split(".")[0].strip()
 
 
def _naturalise_question(cuad_question: str, category: str) -> str:
    """Convert CUAD's verbose QA question into a concise natural-language query.
 
    The 'Details:' portion already contains a natural question, so we extract
    and use that directly.
 
    Example:
      Input:  'Highlight the parts (if any) related to "Governing Law".
               Details: Which state/country's law governs the contract?'
      Output: 'Which state/country's law governs the contract?'
    """
    if "Details:" in cuad_question:
        return cuad_question.split("Details:")[-1].strip()
    return f"What does the contract say about {category.lower()}?"
 
 
def _load_config(config_path: str) -> dict:
    """Load config.yaml — requires PyYAML."""
    try:
        import yaml  # type: ignore
    except ImportError:
        logger.error("PyYAML not installed.  Run: pip install pyyaml")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Download CUAD via HuggingFace datasets
# ─────────────────────────────────────────────────────────────────────────────
 
def download_cuad(split: str = "test") -> tuple[list[dict], list[dict]]:
    """Download CUAD and return (train_data, test_data) as raw dicts.
 
    HuggingFace caches the download locally in ~/.cache/huggingface/datasets/
    so subsequent runs are instant.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        logger.error(
            "HuggingFace 'datasets' not installed.\n"
            "  Run: pip install datasets"
        )
        sys.exit(1)
 
    logger.info("Downloading CUAD from HuggingFace (first run may take a few minutes)…")
    ds = load_dataset("cuad", trust_remote_code=True)
    logger.info(
        "CUAD downloaded. Train: %d contracts, Test: %d contracts.",
        len(ds["train"]),
        len(ds["test"]),
    )
    return ds["train"].to_list(), ds["test"].to_list()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Save contracts as .txt files
# ─────────────────────────────────────────────────────────────────────────────
 
def save_contracts(
    raw_data: list[dict],
    contracts_dir: Path,
) -> dict[str, Path]:
    """Extract contract texts and write one .txt per contract.
 
    CUAD groups paragraphs by title (contract name).  We reconstruct the full
    contract text by joining all paragraph contexts for that title.
 
    Returns a mapping {title: file_path} for later indexing.
    """
    contracts_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
 
    # Group paragraphs by contract title
    by_title: dict[str, list[str]] = {}
    for item in raw_data:
        title = item.get("title", "unknown")
        for para in item.get("paragraphs", []):
            by_title.setdefault(title, []).append(para["context"])
 
    for title, paragraphs in by_title.items():
        # Sanitise filename
        safe_name = re.sub(r'[^\w\-.]', '_', title)[:120]
        file_path = contracts_dir / f"{safe_name}.txt"
        full_text = "\n\n".join(paragraphs)
        file_path.write_text(full_text, encoding="utf-8")
        written[title] = file_path
 
    logger.info("Saved %d contracts to %s", len(written), contracts_dir)
    return written
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Build eval dataset JSON
# ─────────────────────────────────────────────────────────────────────────────
 
def build_eval_dataset(
    raw_data: list[dict],
    output_path: Path,
    categories: list[str] | None = None,
    max_per_category: int = 5,
) -> list[dict]:
    """Convert CUAD QA pairs to the EvalSample JSON format.
 
    Only keeps:
    - Answerable pairs (is_impossible == False) with at least one gold span.
    - Categories listed in *categories* (or all 41 if None).
    - At most *max_per_category* samples per category (to keep eval tractable).
 
    Output JSON schema (matches your existing EvalSample dataclass):
    [
      {
        "question": "Which state governs this contract?",
        "reference_answer": "This Agreement shall be governed by ...",
        "doc_id": "VerizonTransferAgreement_2020",
        "metadata": {
          "cuad_category": "Governing Law",
          "contract_title": "VerizonTransferAgreement_2020",
          "is_impossible": false
        }
      },
      ...
    ]
    """
    category_counts: dict[str, int] = {}
    samples: list[dict] = []
 
    for item in raw_data:
        title = item.get("title", "unknown")
        for para in item.get("paragraphs", []):
            for qa in para.get("qas", []):
                is_impossible = qa.get("is_impossible", True)
                if is_impossible:
                    continue  # skip unanswerable
 
                answers = qa.get("answers", [])
                if not answers:
                    continue
 
                category = _parse_category(qa["question"])
 
                # Filter by requested categories
                if categories and category not in categories:
                    continue
 
                # Cap per-category
                if category_counts.get(category, 0) >= max_per_category:
                    continue
 
                # Use the first gold answer span
                reference_answer = answers[0]["text"].strip()
                if not reference_answer:
                    continue
 
                natural_q = _naturalise_question(qa["question"], category)
 
                samples.append({
                    "question": natural_q,
                    "reference_answer": reference_answer,
                    "doc_id": title,
                    "metadata": {
                        "cuad_category": category,
                        "contract_title": title,
                        "is_impossible": False,
                        "cuad_qa_id": qa.get("id", ""),
                    },
                })
                category_counts[category] = category_counts.get(category, 0) + 1
 
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
 
    logger.info(
        "Built eval dataset: %d samples across %d categories → %s",
        len(samples),
        len(category_counts),
        output_path,
    )
 
    # Print per-category breakdown
    logger.info("Per-category counts:")
    for cat, count in sorted(category_counts.items()):
        logger.info("  %-40s %d", cat, count)
 
    return samples
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Ingest contracts into FAISS
# ─────────────────────────────────────────────────────────────────────────────
 
def ingest_contracts(contracts_dir: Path, config_path: str) -> None:
    """Call the existing ingest.py CLI to embed and index the saved contracts.
 
    This reuses 100% of your existing ingestion pipeline (loader → preprocessor
    → chunker → embedder → FAISSVectorStore) without duplicating any code.
    """
    import subprocess
    logger.info("Ingesting contracts into FAISS index…")
    cmd = [
        sys.executable, "scripts/ingest.py",
        "--docs", str(contracts_dir),
        "--config", config_path,
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.warning(
            "Ingestion returned non-zero exit code %d. "
            "Check the output above for errors.",
            result.returncode,
        )
    else:
        logger.info("Ingestion complete.")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CUAD, save contracts, build eval dataset, ingest."
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Skip the FAISS ingestion step (useful if already ingested).",
    )
    parser.add_argument(
        "--split", default=None,
        help="Override config cuad.split (train | test).",
    )
    args = parser.parse_args()
 
    cfg = _load_config(args.config)
    cuad_cfg = cfg.get("cuad", {})
 
    split = args.split or cuad_cfg.get("split", "test")
    contracts_dir = Path(cuad_cfg.get("contracts_dir", "data/cuad_contracts/"))
    eval_path = Path(cuad_cfg.get("eval_dataset_path", "data/cuad_eval.json"))
    max_per_cat = cuad_cfg.get("max_samples_per_category", 5)
    categories = cuad_cfg.get("categories", None)   # None → all 41
 
    # ── 1. Download ────────────────────────────────────────────────────
    train_data, test_data = download_cuad(split=split)
    eval_raw = test_data if split == "test" else train_data
    # Use the OTHER split for the FAISS knowledge base to avoid leakage.
    index_raw = train_data if split == "test" else test_data
 
    # ── 2. Save contracts ──────────────────────────────────────────────
    save_contracts(index_raw, contracts_dir)
 
    # ── 3. Build eval dataset ──────────────────────────────────────────
    build_eval_dataset(
        raw_data=eval_raw,
        output_path=eval_path,
        categories=categories,
        max_per_category=max_per_cat,
    )
 
    logger.info(
        "\nEval dataset written to: %s\n"
        "To use it with your pipeline, either:\n"
        "  (a) set evaluation.eval_dataset: %s in config.yaml, or\n"
        "  (b) pass --dataset %s to scripts/run_evaluation.py",
        eval_path, eval_path, eval_path,
    )
 
    # ── 4. Ingest ──────────────────────────────────────────────────────
    if not args.skip_ingest:
        ingest_contracts(contracts_dir, args.config)
    else:
        logger.info(
            "Skipping ingestion (--skip-ingest).  "
            "Run manually:  python scripts/ingest.py --docs %s --config %s",
            contracts_dir, args.config,
        )
 
    logger.info(
        "\n✓  CUAD setup complete.\n"
        "   Next steps:\n"
        "   1. Run the ablation:  python scripts/run_ablation.py --config %s\n"
        "   2. Or test one mode:  python scripts/run_evaluation.py "
        "--dataset %s --config %s",
        args.config, eval_path, args.config,
    )
 
 
if __name__ == "__main__":
    main()
