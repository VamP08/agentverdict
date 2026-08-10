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
            if response.status_code in self.RETRY_STATUS and _server_forbids_retry(response):
                # The server can say the request will not succeed on a retry -- a daily
                # token budget rather than a momentary burst. Three more attempts spread
                # over a minute change nothing except how long the caller waits to find out.
                raise JudgeClientError(
                    f"Groq API error {response.status_code} (server advised against retrying): "
                    f"{response.text[:300]}"
                )
            if response.status_code in self.RETRY_STATUS:
                requested = _requested_delay(response)
                throttled = requested is not None and requested > MAX_RETRY_DELAY_S
                if throttled:
                    # Name the cause. A caller told to come back in five minutes is being
                    # rate limited, not served slowly, and "Groq API error 429" does not
                    # tell them to slow the suite down or raise the account's ceiling.
                    last_error = (
                        f"rate limited by the server ({response.status_code}); it asked for "
                        f"{requested:.0f}s and this client waits at most "
                        f"{MAX_RETRY_DELAY_S:.0f}s per attempt"
                    )
                else:
                    last_error = f"retryable status {response.status_code}"
                if attempt < self.MAX_ATTEMPTS:
                    time.sleep(_retry_delay(response, attempt))
                    continue
                if throttled:
                    raise JudgeClientError(last_error)
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


#: Longest wait this client will honor from a ``retry-after`` header, in seconds.
#:
#: The server is allowed to ask for any delay it likes. Groq answers a tokens-per-minute
#: squeeze with a wait long enough to clear the window, and a suite that makes hundreds of
#: calls -- an eval sweep, a bias probe -- will meet one. Sleeping for exactly as long as
#: asked is indistinguishable from a hang: no output, no error, a process that looks alive
#: and is doing nothing. Bounded instead, so an exhausted run reports a rate limit rather
#: than stalling behind one.
MAX_RETRY_DELAY_S = 30.0


def _server_forbids_retry(response: httpx.Response) -> bool:
    """True when the server sent ``x-should-retry: false``.

    Groq sets it on a rate limit that a retry cannot clear -- a per-day token budget, as
    opposed to the per-minute window a short back-off does clear. Taking the server at its
    word turns four futile attempts into one immediate, accurate error.
    """
    return response.headers.get("x-should-retry", "").strip().lower() == "false"


def _requested_delay(response: httpx.Response) -> float | None:
    """The server's own ``retry-after``, in seconds, or None when it did not say."""
    header = response.headers.get("retry-after")
    if header is None:
        return None
    try:
        return max(float(header), 0.0)
    except ValueError:
        # Retry-After also permits an HTTP date. Falling back to the exponential
        # schedule is better than parsing dates against an unsynchronised clock.
        return None


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    requested = _requested_delay(response)
    if requested is not None:
        return min(requested, MAX_RETRY_DELAY_S)
    return float(min(2**attempt, 8))
