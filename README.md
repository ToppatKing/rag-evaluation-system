# Domain-Specific RAG System with Rigorous Evaluation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/ToppatKing/rag-evaluation-system/actions/workflows/tests.yml/badge.svg)](https://github.com/ToppatKing/rag-evaluation-system/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline with a multi-dimensional evaluation framework. Built to benchmark chunking strategies, retrieval methods, and generation quality on domain-specific corpora.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                                  │
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌────────────────────────┐    │
│  │  INGESTION  │    │  RETRIEVAL  │    │      GENERATION        │    │
│  │             │    │             │    │                        │    │
│  │ DocumentLoad│──▶│  Embedder   │    │  PromptBuilder          │   │
│  │ Preprocessor│    │  (ST / OAI) │    │  Generator             │    │
│  │ Chunker     │──▶│  FAISSStore │───▶│  (OpenAI / Anthropic)  │    │
│  │ (Fixed /    │    │  Retriever  │    │                        │    │
│  │  Recursive /│    │  (Dense/MMR/│    └────────────────────────┘    │
|  |  Semantic)  |    │   HyDE)     │                                  | 
│  │             │    └─────────────┘               │                  │
│  └─────────────┘                                  │                  │
│                                                   ▼                  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                     EVALUATION FRAMEWORK                       │  │
│  │                                                                │  │
│  │  Faithfulness │ AnswerRelevancy │ ContextPrecision │ ROUGE-L   │  │
│  │  ContextRecall │ Latency │ TokenEfficiency │ BERTScore (opt.)  │  │
│  │                                                                │  │
│  │  EvaluationReport ──▶ CSV / JSON / Console                    │   │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Features

| Category | Capability |
|---|---|
| **Ingestion** | PDF, TXT, Markdown, DOCX loading |
| **Chunking** | Fixed-size, Recursive, Semantic (embedding-based) |
| **Embedding** | `sentence-transformers` (local) or OpenAI `text-embedding-3-small` |
| **Vector Store** | FAISS with cosine similarity; persistent index |
| **Retrieval** | Dense similarity search; MMR for diversity; HyDE zero-shot retrieval |
| **Generation** | OpenAI GPT-4o or Anthropic Claude; structured prompting |
| **Evaluation** | 7 metrics; LLM-as-judge + classical NLP |
| **Reporting** | Console, CSV, JSON; per-query breakdown |

---

## Evaluation Metrics

| Metric | Type | Description |
|---|---|---|
| **Faithfulness** | LLM-as-judge | Fraction of answer claims supported by retrieved context |
| **Answer Relevancy** | Embedding cosine | Semantic similarity between question and answer |
| **Context Precision** | LLM-as-judge | Proportion of retrieved chunks that are relevant |
| **Context Recall** | LLM-as-judge | Coverage of ground-truth information in retrieved context |
| **ROUGE-L** | String overlap | Longest common subsequence F1 vs. reference answer |
| **Latency** | Timing | End-to-end query latency (retrieval + generation) |
| **Token Efficiency** | Ratio | Answer tokens / context tokens (conciseness proxy) |

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/ToppatKing/rag-evaluation-system.git
cd rag-evaluation-system
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp config/config.yaml config/local.yaml
# Edit config/local.yaml — set your API keys or choose local embeddings
```

Set environment variables:
```bash
export OPENAI_API_KEY="sk-..."        # for OpenAI embeddings + generation
export ANTHROPIC_API_KEY="sk-ant-..." # for Anthropic generation (optional)
```

### 3. Ingest documents

```bash
python scripts/ingest.py --docs data/sample_docs/ --config config/config.yaml
```

### 4. Run the demo

```bash
python scripts/run_demo.py --query "What is attention mechanism in transformers?"
```

### 5. Run evaluation

```bash
python scripts/run_evaluation.py \
  --dataset data/eval_dataset.json \
  --config config/config.yaml \
  --output results/
```

---

## Sample Evaluation Results

```
╔══════════════════════════════════════════════════════════════╗
║              RAG EVALUATION REPORT — 50 queries             ║
╠══════════════════════════════════════════════════════════════╣
║  Faithfulness       │ ████████████████░░░░ │  0.81 ± 0.09   ║
║                     |                      |                ║
║  Answer Relevancy   │ █████████████████░░░ │  0.84 ± 0.07   ║
║                     |                      |                ║
║  Context Precision  │ ███████████████░░░░░ │  0.76 ± 0.13   ║
║                     |                      |                ║
║  Context Recall     │ ██████████████░░░░░░ │  0.72 ± 0.15   ║
║                     |                      |                ║
║  ROUGE-L            │ ████████████░░░░░░░░ │  0.61 ± 0.18   ║
║  Avg Latency        │                      │  1.34s ± 0.41s ║
║  Token Efficiency   │                      │  0.23 ± 0.11   ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Project Structure

```
rag-evaluation-system/
├── config/
│   └── config.yaml              # All pipeline hyperparameters
├── src/rag_eval/
│   ├── ingestion/
│   │   ├── loader.py            # Multi-format document loading
│   │   ├── preprocessor.py      # Text cleaning & normalization
│   │   └── chunker.py           # Fixed / Recursive / Semantic chunking
│   ├── retrieval/
│   │   ├── embedder.py          # SentenceTransformer & OpenAI backends
│   │   ├── vector_store.py      # FAISS index wrapper
│   │   └── retriever.py         # Dense & MMR retrieval
│   ├── generation/
│   │   └── generator.py         # OpenAI / Anthropic generation
│   ├── evaluation/
│   │   ├── metrics.py           # Individual metric implementations
│   │   ├── evaluator.py         # Metric orchestration & reporting
│   │   └── dataset.py           # EvalSample dataclass & loaders
│   └── pipeline.py              # End-to-end RAGPipeline
├── tests/
│   ├── conftest.py
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_metrics.py
├── scripts/
│   ├── ingest.py
│   ├── setup_cuad.py
│   ├── run_demo.py
│   ├── run_ablation.py
│   └── run_evaluation.py
└── data/
    └── sample_docs/
```

---

## Running Tests

```bash
pytest tests/ -v --cov=src/rag_eval --cov-report=term-missing
```

---

## Configuration

All pipeline knobs live in `config/config.yaml`:

```yaml
chunking:
  strategy: recursive      # fixed | recursive | semantic
  chunk_size: 512
  chunk_overlap: 64

retrieval:
  top_k: 5
  method: mmr              # dense | mmr | hyde
  mmr_lambda: 0.5

generation:
  provider: openai         # openai | anthropic
  model: gpt-4o
  temperature: 0.1
  max_tokens: 512
```
## CUAD Legal Corpus Setup

To run experiments on the [CUAD dataset](https://arxiv.org/abs/2103.06268) (510 SEC-filed contracts, 41 clause categories):

```bash
# Install the extra dependency
pip install datasets

# Download contracts, build eval dataset, and ingest into FAISS
python scripts/setup_cuad.py --config config/config.yaml
```

Then run the three-way retrieval ablation:

```bash
python scripts/run_ablation.py \
  --config config/config.yaml \
  --dataset data/cuad_eval.json \
  --output results/cuad_ablation/
```

This evaluates **Dense vs. MMR vs. HyDE** across all 7 metrics on the same corpus and produces `results/cuad_ablation/ablation_report.txt`.

---

## Extending the System

**Add a new chunker:**
```python
from rag_eval.ingestion.chunker import BaseChunker

class SlidingWindowChunker(BaseChunker):
    def chunk(self, text: str) -> list[Chunk]:
        ...
```

**Add a new metric:**
```python
from rag_eval.evaluation.metrics import BaseMetric, MetricResult

class ExactMatchMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "exact_match"

    def compute(self, sample: EvalSample, response: RAGResponse) -> MetricResult:
        score = float(sample.reference_answer.strip().lower()
                      == response.answer.strip().lower())
        return MetricResult(name=self.name, score=score)
```


---
## 💡 Troubleshooting & API Notes

* **OpenAI Account Balance:** Ensure your OpenAI account has active prepaid credits. Brand-new accounts with a $0 balance will trigger an `insufficient_quota` (429) error.
* **API Key Management:** Always set your full, unmasked secret key via environment variables (`OPENAI_API_KEY`). Avoid leaving placeholder strings (like `sk-...`) in your configuration files.
* **HyDE Token Consumption:** When running the ablation script with HyDE enabled, keep in mind that it makes an extra LLM generation call for every single query prior to retrieval, which consumes additional API tokens and budget.
## License

MIT — see [LICENSE](LICENSE).
