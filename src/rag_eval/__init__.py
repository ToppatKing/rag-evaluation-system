"""rag_eval — Domain-Specific RAG System with Rigorous Evaluation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rag-eval")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__all__ = ["__version__"]
