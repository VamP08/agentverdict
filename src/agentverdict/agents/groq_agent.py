"""The reference agent-under-test: a Groq tool-calling loop over the mock tools.

This is the agent AgentVerdict replays against stored tasks to produce fresh
trajectories. It is deliberately ordinary — a system persona, the task's tools,
and a bounded call loop — because the interesting object under measurement is the
judge, not this agent. Adapters never touch the database: everything the replay
runner needs comes back in ``AgentRunResult``.

The persona never mentions the task's expected outcome or that a run is being
evaluated; leaking either would make every replay trivially correct.
"""

from __future__ import annotations

import json
from typing import Any

from agentverdict.agents.base import AgentRunResult, opening_message
from agentverdict.agents.mock_tools import MockToolRegistry, ToolOutcome
from agentverdict.config import get_settings
from agentverdict.judging.client import ChatResult, GroqChatClient, JudgeClientError, ToolCall
from agentverdict.models import Task, utcnow
from agentverdict.schemas import StepIn, TrajectoryStatus

DEFAULT_SYSTEM_PROMPT = """\
You are Aria, a customer-support agent for Northwind Gadgets, an online store selling small \
consumer electronics. You are chatting with a customer who has just messaged support.

You have tools for looking up orders, issuing refunds, and emailing customers. Call a tool \
whenever you need real information or need to act on an order, and never state order details, \
amounts, or delivery dates that a tool has not given you.

How to handle a conversation well:
- Work out which order the customer means before you do anything to it. If they have not given \
you an order id and you cannot identify the order from what they said, ask them for it instead \
of guessing.
- Check that the order on file matches the customer you are talking to before you change it.
- Refunds and emails cannot be taken back. Say what you intend to do and get the customer's \
agreement before you do it.
- If a tool returns an error, say what you found and what you will do next; never describe a \
failed action as if it had succeeded.
- Keep the customer informed: acknowledge how they feel, explain what you are doing, and close \
with a short summary of what happened and what to expect next.

Write like a person: warm, direct, plain language, no filler.\
"""


class GroqReferenceAgent:
    """Runs a task through Groq with mock tools and records every step."""

    def __init__(
        self,
        model: str | None = None,
        registry: MockToolRegistry | None = None,
        client: GroqChatClient | None = None,
        max_turns: int = 8,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        name: str = "groq-agent",
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.name = name
        self.model = model or get_settings().judge_model
        self.registry = registry if registry is not None else MockToolRegistry.from_file()
        self.max_turns = max_turns
        self.temperature = temperature
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._client = client
        self._owns_client = client is None

    # --- client lifecycle ----------------------------------------------------

    def _ensure_client(self) -> GroqChatClient:
        """Build the HTTP client on first use, so constructing an agent is cheap."""
        if self._client is None:
            self._client = GroqChatClient()
            self._owns_client = True
        return self._client

    def close(self) -> None:
        """Close the chat client, but only when this agent created it."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> GroqReferenceAgent:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- the run loop --------------------------------------------------------

    def run(self, task: Task) -> AgentRunResult:
        """Attempt ``task`` once and return the recorded steps."""
        started_at = utcnow()
        opening = opening_message(task)
        tools = wrap_tools_spec(task.tools_spec)

        steps: list[StepIn] = [
            StepIn(type="system", content={"text": self.system_prompt}),
            StepIn(type="user_message", content={"text": opening}),
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": opening},
        ]

        status: TrajectoryStatus = "truncated"
        turn_detail: list[dict[str, Any]] = []
        tool_call_counts: dict[str, int] = {}
        input_tokens = 0
        output_tokens = 0
        tool_calls_made = 0
        tool_errors = 0
        finish_reason: str | None = None
        error_message: str | None = None

        try:
            client = self._ensure_client()
            for turn in range(1, self.max_turns + 1):
                result = client.chat(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    tools=tools or None,
                )
                input_tokens += result.input_tokens
                output_tokens += result.output_tokens
                finish_reason = result.finish_reason
                turn_detail.append(
                    {
                        "turn": turn,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "latency_ms": round(result.latency_ms, 2),
                        "tool_calls": len(result.tool_calls),
                        "finish_reason": result.finish_reason,
                    }
                )

                text = (result.content or "").strip()
                if text:
                    steps.append(StepIn(type="assistant_message", content={"text": text}))

                if not result.tool_calls:
                    status = "completed"
                    break

                messages.append(_assistant_message(result))
                for call in result.tool_calls:
                    tool_calls_made += 1
                    tool_call_counts[call.name] = tool_call_counts.get(call.name, 0) + 1
                    arguments, outcome = self._execute(call)
                    if outcome.is_error:
                        tool_errors += 1
                    steps.append(
                        StepIn(
                            type="tool_call",
                            content={"name": call.name, "arguments": arguments},
                        )
                    )
                    steps.append(
                        StepIn(
                            type="tool_result",
                            content={
                                "name": call.name,
                                "result": outcome.result,
                                "is_error": outcome.is_error,
                            },
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": _dumps(outcome.result),
                        }
                    )
        except JudgeClientError as exc:
            status = "error"
            error_message = str(exc)

        meta: dict[str, Any] = {
            "adapter": self.name,
            "model": self.model,
            "turns": len(turn_detail),
            "max_turns": self.max_turns,
            "turn_detail": turn_detail,
            "tool_calls": tool_calls_made,
            "tool_errors": tool_errors,
            "tool_call_counts": tool_call_counts,
            "finish_reason": finish_reason,
        }
        if error_message is not None:
            meta["error"] = error_message

        return AgentRunResult(
            steps=steps,
            status=status,
            agent_config={
                "adapter": self.name,
                "model": self.model,
                "max_turns": self.max_turns,
                "temperature": self.temperature,
            },
            meta=meta,
            started_at=started_at,
            completed_at=utcnow(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _execute(self, call: ToolCall) -> tuple[dict[str, Any], ToolOutcome]:
        """Run one requested call, turning malformed arguments into a tool error.

        A model that emits invalid JSON arguments gets the failure back as a tool
        result and can correct itself on the next turn — a recoverable mistake,
        not a dead run.
        """
        try:
            arguments = call.parsed_arguments()
        except ValueError as exc:
            outcome = ToolOutcome(
                result={
                    "error": "invalid_arguments",
                    "detail": str(exc),
                    "received": call.raw_arguments,
                },
                is_error=True,
            )
            return {}, outcome
        return arguments, self.registry.call(call.name, arguments)


def wrap_tools_spec(tools_spec: Any) -> list[dict[str, Any]]:
    """Wrap a task's tool specs as the chat API expects.

    Stored specs are bare function schemas (``{"name": ..., "parameters": ...}``);
    the wire format needs ``{"type": "function", "function": spec}``. Entries that
    are already wrapped pass through, and entries without a name are skipped
    rather than sent as an invalid tool definition.
    """
    wrapped: list[dict[str, Any]] = []
    if not isinstance(tools_spec, list):
        return wrapped
    for spec in tools_spec:
        if not isinstance(spec, dict):
            continue
        inner = spec.get("function") if spec.get("type") == "function" else spec
        if not isinstance(inner, dict):
            continue
        name = inner.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        wrapped.append({"type": "function", "function": dict(inner)})
    return wrapped


def _assistant_message(result: ChatResult) -> dict[str, Any]:
    """The assistant turn to append when the model asked for tools."""
    return {
        "role": "assistant",
        "content": result.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.raw_arguments},
            }
            for call in result.tool_calls
        ],
    }


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
