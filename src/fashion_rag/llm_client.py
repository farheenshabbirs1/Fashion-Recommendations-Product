"""Provider-agnostic LLM client for generating the final recommendation text.

Resolution order (explicit ``provider`` argument, or the ``LLM_PROVIDER`` env var):
  - "anthropic": uses the Anthropic API (requires ANTHROPIC_API_KEY)
  - "openai": uses the OpenAI API (requires OPENAI_API_KEY)
  - "mock" (default): deterministic, template-based generation with no
    network calls -- lets the whole pipeline run out of the box with no
    API keys, which is what the demo and tests use by default.
"""
from __future__ import annotations

import abc
import os
import re
from typing import Optional


class LLMClient(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        ...


class MockLLMClient(LLMClient):
    """Deterministic stand-in for a real LLM, built from the prompt's own context block.

    It parses the "- Name ($price, category, ...): description Review
    highlight: ..." lines that PromptBuilder produces and turns each into a
    short recommendation sentence, so the demo is reproducible, free, and
    requires no API key.
    """

    _LINE_RE = re.compile(r"^- (?P<name>[^(]+) \((?P<meta>[^)]+)\): (?P<rest>.+)$")

    def generate(self, prompt: str) -> str:
        candidate_lines = [line for line in prompt.splitlines() if line.startswith("- ")]
        if not candidate_lines:
            return "I couldn't find a matching product in the provided context."

        recommendations = []
        for line in candidate_lines[:3]:
            match = self._LINE_RE.match(line)
            if not match:
                continue
            name = match.group("name").strip()
            rest = match.group("rest").strip()
            reason = rest.split("Review highlight:")[0].strip().rstrip(".")
            recommendations.append(f"- {name}: {reason}.")

        header = "Based on your request, here are my top picks:"
        return header + "\n" + "\n".join(recommendations)


class OpenAILLMClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only when installed
            raise ImportError("openai is not installed. `pip install openai`.") from exc
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


class AnthropicLLMClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-5") -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only when installed
            raise ImportError("anthropic is not installed. `pip install anthropic`.") from exc
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text"))


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    """Factory: choose an LLM backend by explicit name, the ``LLM_PROVIDER`` env var, or "mock"."""
    choice = (provider or os.getenv("LLM_PROVIDER") or "mock").lower()
    if choice == "mock":
        return MockLLMClient()
    if choice == "openai":
        return OpenAILLMClient()
    if choice == "anthropic":
        return AnthropicLLMClient()
    raise ValueError(f"Unknown LLM provider '{choice}'. Use 'mock', 'openai', or 'anthropic'.")
