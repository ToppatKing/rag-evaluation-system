"""LLM generation backends.

Supports:
* :class:`OpenAIGenerator`     — OpenAI Chat Completions API
* :class:`AnthropicGenerator`  — Anthropic Messages API

Both implement :class:`BaseGenerator` which accepts a natural-language query
plus a list of context strings, builds a structured prompt, and returns a
:class:`GenerationResult`.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, factual assistant. Answer the user's question using ONLY "
    "the provided context. If the context does not contain sufficient information, "
    "state that clearly rather than speculating. Be concise and accurate."
)

_CONTEXT_TEMPLATE = """\
<context>
{context}
</context>

Question: {query}

Answer:"""


@dataclass
class GenerationResult:
    """Result from a generation call.

    Attributes:
        answer: The generated answer text.
        query: The original query.
        contexts: Context strings used.
        model: Model identifier.
        prompt_tokens: Tokens in the prompt (if reported by API).
        completion_tokens: Tokens in the completion.
        latency_s: Wall-clock time for the API call.
        metadata: Additional provider-specific data.
    """

    answer: str
    query: str
    contexts: list[str]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def context_text(self) -> str:
        return "\n\n".join(
            f"[{i + 1}] {ctx}" for i, ctx in enumerate(self.contexts)
        )


class BaseGenerator(ABC):
    """Abstract interface for generation backends."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    @abstractmethod
    def generate(self, query: str, contexts: list[str]) -> GenerationResult:
        """Generate an answer for *query* grounded in *contexts*.

        Args:
            query: The user question.
            contexts: Retrieved text passages.

        Returns:
            :class:`GenerationResult` with answer and usage metadata.
        """

    def _build_user_message(self, query: str, contexts: list[str]) -> str:
        context_text = "\n\n".join(
            f"[Context {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )
        return _CONTEXT_TEMPLATE.format(context=context_text, query=query)


# ── OpenAI ────────────────────────────────────────────────────────────────────


class OpenAIGenerator(BaseGenerator):
    """OpenAI Chat Completions generator.

    Args:
        model: OpenAI model identifier (e.g. ``"gpt-4o-mini"``).
        temperature: Sampling temperature.
        max_tokens: Max completion tokens.
        system_prompt: System role message.
        api_key: Overrides ``OPENAI_API_KEY`` environment variable.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 512,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        api_key: str | None = None,
    ) -> None:
        super().__init__(model, temperature, max_tokens, system_prompt)
        try:
            from openai import OpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc
        self._client = OpenAI(api_key=api_key)

    def generate(self, query: str, contexts: list[str]) -> GenerationResult:
        user_msg = self._build_user_message(query, contexts)
        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency = time.perf_counter() - t0

        answer = response.choices[0].message.content or ""
        usage = response.usage
        logger.debug(
            "OpenAI (%s) | tokens: %d prompt + %d completion | %.2fs",
            self.model,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            latency,
        )
        return GenerationResult(
            answer=answer,
            query=query,
            contexts=contexts,
            model=self.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_s=latency,
        )


# ── Anthropic ─────────────────────────────────────────────────────────────────


class AnthropicGenerator(BaseGenerator):
    """Anthropic Messages API generator.

    Args:
        model: Anthropic model identifier (e.g. ``"claude-haiku-4-5-20251001"``).
        temperature: Sampling temperature.
        max_tokens: Max completion tokens.
        system_prompt: System role message.
        api_key: Overrides ``ANTHROPIC_API_KEY`` environment variable.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        temperature: float = 0.1,
        max_tokens: int = 512,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        api_key: str | None = None,
    ) -> None:
        super().__init__(model, temperature, max_tokens, system_prompt)
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Install anthropic: pip install anthropic") from exc
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(self, query: str, contexts: list[str]) -> GenerationResult:
        user_msg = self._build_user_message(query, contexts)
        t0 = time.perf_counter()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_msg}],
            temperature=self.temperature,
        )
        latency = time.perf_counter() - t0

        answer = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        usage = response.usage
        return GenerationResult(
            answer=answer,
            query=query,
            contexts=contexts,
            model=self.model,
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
            latency_s=latency,
        )


# ── Factory ───────────────────────────────────────────────────────────────────


def build_generator(config: dict[str, Any]) -> BaseGenerator:
    """Construct a generator from configuration.

    Args:
        config: Must contain ``provider`` (``"openai"`` or ``"anthropic"``),
            ``model``, ``temperature``, ``max_tokens``, and optionally
            ``system_prompt``.

    Returns:
        Configured :class:`BaseGenerator`.
    """
    provider = config.get("provider", "openai")
    model = str(config.get("model", ""))
    temperature = float(config.get("temperature", 0.1))
    max_tokens = int(config.get("max_tokens", 512))
    system_prompt = str(config.get("system_prompt", _DEFAULT_SYSTEM_PROMPT))

    if provider == "openai":
        return OpenAIGenerator(
            model=model or "gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
    if provider == "anthropic":
        return AnthropicGenerator(
            model=model or "claude-haiku-4-5-20251001",
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
    raise ValueError(f"Unknown generation provider: {provider!r}")
