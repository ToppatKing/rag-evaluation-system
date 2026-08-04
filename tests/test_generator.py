from __future__ import annotations

from types import SimpleNamespace

import rag_eval.generation.generator as generator_module
from rag_eval.generation.generator import GeminiGenerator


def test_gemini_generator_retries_supported_model_names(monkeypatch):
    calls: list[str] = []

    class FakeModel:
        def __init__(self, model_name: str):
            self.model_name = model_name
            calls.append(model_name)

        def generate_content(self, **_: object) -> SimpleNamespace:
            if self.model_name == "gemini-1.5-flash":
                raise RuntimeError("404 NOT_FOUND")
            return SimpleNamespace(text="fallback success")

    monkeypatch.setattr(generator_module.genai, "GenerativeModel", FakeModel)

    generator = GeminiGenerator(model_name="gemini-1.5-flash")
    result = generator.generate("question", ["context"])

    assert calls[0] == "gemini-1.5-flash"
    assert calls[1] == "gemini-2.0-flash"
    assert result.answer == "fallback success"
