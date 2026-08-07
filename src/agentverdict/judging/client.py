"""Minimal Groq chat-completions client (OpenAI-compatible wire format).

Hand-rolled on httpx rather than an SDK so retries, timeouts, latency, and token
accounting stay explicit. Tests inject an ``httpx`` transport — no network.

``chat`` is the general entry point (supports tool calling); ``chat_json`` is the
judge-facing wrapper that forces a JSON-object response.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from agentverdict.config import get_settings


class JudgeClientError(RuntimeError):
    """A model API call failed after retries (or was misconfigured)."""


@dataclass
class ToolCall:
    """One tool call requested by the model.

    ``raw_arguments`` is the wire string; ``parsed_arguments`` decodes it and
    raises ``ValueError`` when the model emitted malformed JSON.
    """

    id: str
    name: str
    raw_arguments: str

    def parsed_arguments(self) -> dict[str, Any]:
        if not self.raw_arguments.strip():
            return {}
        try:
            arguments = json.loads(self.raw_arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool call arguments were not valid JSON: {exc}") from exc
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments must be a JSON object")
        return arguments


@dataclass
class ChatResult:
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None


class GroqChatClient:
    """Synchronous chat client with bounded retries."""

    MAX_ATTEMPTS = 4
    RETRY_STATUS = {429, 500, 502, 503}

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        key = api_key or settings.groq_api_key
        if not key:
            raise JudgeClientError(
                "No Groq API key configured. Set AGENTVERDICT_GROQ_API_KEY or GROQ_API_KEY."
            )
        self._client = httpx.Client(
            base_url=base_url or settings.groq_base_url,
            timeout=timeout_s or settings.judge_timeout_s,
            headers={"Authorization": f"Bearer {key}"},
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GroqChatClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        response_format: dict[str, str] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """POST /chat/completions and return the assistant message."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        last_error = "exhausted retries"
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            start = time.perf_counter()
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                last_error = f"transport error: {exc}"
                if attempt < self.MAX_ATTEMPTS:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise JudgeClientError(last_error) from exc
            if response.status_code in self.RETRY_STATUS and attempt < self.MAX_ATTEMPTS:
                time.sleep(_retry_delay(response, attempt))
                continue
            if response.status_code != 200:
                raise JudgeClientError(
                    f"Groq API error {response.status_code}: {response.text[:300]}"
                )
            latency_ms = (time.perf_counter() - start) * 1000
            data = response.json()
            try:
                choice = data["choices"][0]
                message = choice["message"]
            except (KeyError, IndexError, TypeError) as exc:
                raise JudgeClientError(f"Unexpected Groq response shape: {data!r:.300}") from exc
            usage = data.get("usage") or {}
            return ChatResult(
                content=message.get("content") or "",
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                latency_ms=latency_ms,
                tool_calls=_parse_tool_calls(message.get("tool_calls")),
                finish_reason=choice.get("finish_reason"),
            )
        raise JudgeClientError(last_error)

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
    ) -> ChatResult:
        """Ask for a JSON-object response (used by the judge)."""
        return self.chat(
            model,
            messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        function = entry.get("function") or {}
        name = function.get("name")
        if not name:
            continue
        calls.append(
            ToolCall(
                id=str(entry.get("id") or f"call_{index}"),
                name=str(name),
                raw_arguments=function.get("arguments") or "",
            )
        )
    return calls


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("retry-after")
    if header is not None:
        try:
            return max(float(header), 0.0)
        except ValueError:
            pass
    return float(min(2**attempt, 8))
