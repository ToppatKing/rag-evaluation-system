"""Evaluation dataset structures and loaders.

An evaluation dataset is a collection of :class:`EvalSample` objects.
Each sample contains a query, optional reference answer, and optional
ground-truth context.  Datasets can be loaded from JSON or constructed
programmatically.

JSON schema (array of objects):

.. code-block:: json

    [
      {
        "query": "What is the attention mechanism?",
        "reference_answer": "The attention mechanism ...",
        "ground_truth_context": ["Transformers use attention ..."],
        "metadata": {}
      }
    ]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class EvalSample:
    """A single evaluation sample.

    Attributes:
        query: The question to be answered by the RAG system.
        reference_answer: Gold-standard answer (optional; required for
            ROUGE-L and context-recall metrics).
        ground_truth_context: Text passages that *should* contain the
            answer (optional; used for context-recall evaluation).
        sample_id: Unique identifier for tracing results.
        metadata: Arbitrary labels (e.g. difficulty, topic).
    """

    query: str
    reference_answer: str = ""
    ground_truth_context: list[str] = field(default_factory=list)
    sample_id: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            # Stable hash of the query as a fallback ID
            self.sample_id = f"sample_{abs(hash(self.query)) % 10**8:08d}"

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "query": self.query,
            "reference_answer": self.reference_answer,
            "ground_truth_context": self.ground_truth_context,
            "metadata": self.metadata,
        }


@dataclass
class EvalDataset:
    """An ordered collection of :class:`EvalSample` objects.

    Attributes:
        samples: The evaluation samples.
        name: Human-readable dataset name.
    """

    samples: list[EvalSample]
    name: str = "unnamed"

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[EvalSample]:
        return iter(self.samples)

    def __getitem__(self, idx: int) -> EvalSample:
        return self.samples[idx]

    @classmethod
    def from_json(cls, path: str | Path, name: str = "") -> "EvalDataset":
        """Load an :class:`EvalDataset` from a JSON file.

        Args:
            path: Path to a JSON file containing an array of sample objects.
            name: Optional dataset name; defaults to the file stem.

        Returns:
            Populated :class:`EvalDataset`.
        """
        path = Path(path)
        raw: list[dict[str, object]] = json.loads(path.read_text(encoding="utf-8"))
        samples = [
            EvalSample(
                query=str(item.get("query") or item.get("question") or ""),
                reference_answer=str(item.get("reference_answer") or item.get("answer") or ""),
                ground_truth_context=list(
                    item.get("ground_truth_context") or item.get("contexts") or item.get("context") or []
                ),
                sample_id=str(item.get("sample_id") or item.get("id") or ""),
                metadata=dict(item.get("metadata", {})),  # type: ignore[arg-type]
            )
            for item in raw
        ]
        return cls(samples=samples, name=name or path.stem)

    def to_json(self, path: str | Path) -> None:
        """Serialise the dataset to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([s.to_dict() for s in self.samples], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

def load_sample_dataset() -> EvalDataset:
    """Return a small built-in evaluation dataset for quick testing.

    Covers machine learning fundamentals to be paired with the sample
    documents in ``data/sample_docs/``.
    """
    samples = [
        EvalSample(
            query="What is the attention mechanism in Transformers?",
            reference_answer=(
                "The attention mechanism allows a model to weigh the importance "
                "of different input tokens when producing each output token. "
                "In Transformers, scaled dot-product attention computes queries, "
                "keys, and values from the input, then uses softmax-normalised "
                "dot products to produce a weighted sum of the values."
            ),
            sample_id="s001",
        ),
        EvalSample(
            query="What problem does the vanishing gradient problem cause in RNNs?",
            reference_answer=(
                "In deep RNNs, gradients shrink exponentially as they backpropagate "
                "through many time steps, making it difficult for the network to "
                "learn long-range dependencies."
            ),
            sample_id="s002",
        ),
        EvalSample(
            query="How does dropout regularisation work?",
            reference_answer=(
                "Dropout randomly sets a fraction of neuron activations to zero "
                "during training, preventing co-adaptation and acting as an implicit "
                "ensemble of sub-networks."
            ),
            sample_id="s003",
        ),
        EvalSample(
            query="What distinguishes supervised from unsupervised learning?",
            reference_answer=(
                "Supervised learning trains on labelled examples where the correct "
                "output is provided; unsupervised learning finds patterns in "
                "unlabelled data without explicit target labels."
            ),
            sample_id="s004",
        ),
        EvalSample(
            query="What is the purpose of the softmax function in classification?",
            reference_answer=(
                "Softmax converts raw logits into a probability distribution over "
                "classes by exponentiating each logit and normalising so all "
                "probabilities sum to one."
            ),
            sample_id="s005",
        ),
    ]
    return EvalDataset(samples=samples, name="ml_fundamentals_sample")
