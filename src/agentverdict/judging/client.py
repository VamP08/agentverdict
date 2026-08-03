"""Minimal Groq chat-completions client (OpenAI-compatible wire format).

Hand-rolled on httpx rather than an SDK so retries, timeouts, latency, and token
accounting stay explicit. Tests inject an ``httpx`` transport — no network.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from agentverdict.config import get_settings


class JudgeClientError(RuntimeError):
    """A judge API call failed after retries (or was misconfigured)."""


@dataclass
class ChatResult:
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class GroqChatClient:
    """Synchronous chat client with bounded retries and JSON response format."""

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

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> ChatResult:
        """POST /chat/completions asking for a JSON object response."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
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
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise JudgeClientError(f"Unexpected Groq response shape: {data!r:.300}") from exc
            usage = data.get("usage") or {}
            return ChatResult(
                content=content,
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                latency_ms=latency_ms,
            )
        raise JudgeClientError(last_error)


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("retry-after")
    if header is not None:
        try:
            return max(float(header), 0.0)
        except ValueError:
            pass
    return float(min(2**attempt, 8))
