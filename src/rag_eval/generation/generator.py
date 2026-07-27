"""LLM generation backends.

Supports:
* :class:`OpenAIGenerator`     — OpenAI Chat Completions API
* :class:`AnthropicGenerator`  — Anthropic Messages API

Both implement :class:`BaseGenerator` which accepts a natural-language query
plus a list of context strings, builds a structured prompt, and returns a
:class:`GenerationResult`.
"""


from __future__ import annotations

import os
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
        
    @abstractmethod
    def generate_hypothetical_doc(self, query: str, instruction: str) -> str:
        """Generate a *hypothetical* passage that would answer *query*.
 
        The passage is used only as a query vector — it is never shown to users.
        It may be factually incorrect; that is acceptable by design (HyDE §2.3).
 
        Parameters
        ----------
        query : str
            The user's search query.
        instruction : str
            Domain-specific instruction, e.g.:
            "Write a legal contract clause that answers the question."
 
        Returns
        -------
        str
            The generated hypothetical passage.
        """
 
 


# ── OpenAI ────────────────────────────────────────────────────────────────────


class OpenAIGenerator(BaseGenerator):
    """OpenAI Chat Completions generator."""

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

        # Safely resolve API key: ignore placeholders or empty config strings
        resolved_key = api_key
        if not resolved_key or "..." in resolved_key or resolved_key == "sk-...":
            resolved_key = os.getenv("OPENAI_API_KEY")

        self._client = OpenAI(api_key=resolved_key)

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

    def generate_hypothetical_doc(self, query: str, instruction: str) -> str:
        prompt = (
            f"{instruction}\n\n"
            f"Question: {query}\n\n"
            "Passage:"
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a document writer.  Write only the requested "
                            "passage, with no preamble, no caveats, and no markdown."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,        # slight randomness helps HyDE
                max_tokens=256,         # hypothetical doc should be concise
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("OpenAI hypothetical-doc generation failed: %s", exc)
            raise
 
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

    def generate_hypothetical_doc(self, query: str, instruction: str) -> str:
        prompt = (
            f"{instruction}\n\n"
            f"Question: {query}\n\n"
            "Passage:"
        )
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.7,
                system=(
                    "You are a document writer.  Write only the requested "
                    "passage, with no preamble, no caveats, and no markdown."
                ),
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as exc:
            logger.error("Anthropic hypothetical-doc generation failed: %s", exc)
            raise
 
# ── Factory ───────────────────────────────────────────────────────────────────


def build_generator(config: dict[str, Any]) -> BaseGenerator:
    provider = config.get("provider", "openai")
    model = str(config.get("model", ""))
    temperature = float(config.get("temperature", 0.1))
    max_tokens = int(config.get("max_tokens", 512))
    system_prompt = str(config.get("system_prompt", _DEFAULT_SYSTEM_PROMPT))
    api_key = config.get("api_key")

    if provider == "openai":
        return OpenAIGenerator(
            model=model or "gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            api_key=api_key,
        )
    if provider == "anthropic":
        return AnthropicGenerator(
            model=model or "claude-haiku-4-5-20251001",
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            api_key=api_key,
        )
    raise ValueError(f"Unknown generation provider: {provider!r}")