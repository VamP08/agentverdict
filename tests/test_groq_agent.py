"""The Groq reference agent under test: tool loop, recorded steps, and failure modes.

Every model call goes through an ``httpx.MockTransport``, so the suite needs no
network and no API key. Responses are scripted turn by turn, which also lets the
tests assert on the exact payloads the agent puts on the wire.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import pytest

from agentverdict.agents.base import AgentRunResult, opening_message
from agentverdict.agents.groq_agent import (
    DEFAULT_SYSTEM_PROMPT,
    GroqReferenceAgent,
    wrap_tools_spec,
)
from agentverdict.agents.mock_tools import MockToolRegistry
from agentverdict.judging.client import GroqChatClient
from agentverdict.models import Task

#: Sentinel expected outcome. It must never leave the database: an agent told what
#: the right answer is makes every replay trivially correct.
EXPECTED_OUTCOME = "SENTINEL-the-agent-refunds-NW-1001-after-verifying-it"

ORDER_RESULT = {"order_id": "NW-1001", "status": "delivered", "total": 34.99}

LOOKUP_SPEC: dict[str, Any] = {
    "name": "lookup_order",
    "description": "Fetch an order by id.",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
}


def _task(**overrides: Any) -> Task:
    """An unsaved task; adapters never touch the database."""
    fields: dict[str, Any] = {
        "key": "replay-refund-01",
        "prompt": "Grader view: the agent should look up NW-1001 and refund it.",
        "user_message": "Hi, the mouse from order NW-1001 turned up cracked.",
        "tools_spec": [LOOKUP_SPEC],
        "expected_outcome": EXPECTED_OUTCOME,
        "tags": ["refunds"],
    }
    fields.update(overrides)
    return Task(**fields)


def _body(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class _ScriptedGroq:
    """Serves a fixed list of ``(status, body)`` responses and records the requests."""

    def __init__(
        self,
        script: list[tuple[int, dict[str, Any]]],
        repeat_last: bool = False,
    ) -> None:
        self.script = list(script)
        self.repeat_last = repeat_last
        self.payloads: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        self.payloads.append(json.loads(request.content))
        index = len(self.payloads) - 1
        if index >= len(self.script):
            if not self.repeat_last:
                raise AssertionError(
                    f"agent made {len(self.payloads)} model calls but the script has "
                    f"{len(self.script)}"
                )
            index = len(self.script) - 1
        status, body = self.script[index]
        return httpx.Response(status, json=body)


@pytest.fixture
def registry() -> MockToolRegistry:
    return MockToolRegistry.from_mapping(
        {
            "tools": {
                "lookup_order": {
                    "rules": [
                        {"when": {"order_id": "NW-1001"}, "result": ORDER_RESULT},
                    ],
                    "default": {"result": {"error": "order_not_found"}, "is_error": True},
                }
            }
        }
    )


@contextmanager
def _scripted_agent(
    script: list[tuple[int, dict[str, Any]]],
    registry: MockToolRegistry,
    repeat_last: bool = False,
    **agent_kwargs: Any,
) -> Iterator[tuple[GroqReferenceAgent, _ScriptedGroq]]:
    handler = _ScriptedGroq(script, repeat_last=repeat_last)
    client = GroqChatClient(
        api_key="test-key",
        base_url="https://groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )
    agent = GroqReferenceAgent(
        model="test-model", registry=registry, client=client, **agent_kwargs
    )
    try:
        yield agent, handler
    finally:
        client.close()


def _types(result: AgentRunResult) -> list[str]:
    return [step.type for step in result.steps]


# --- the happy path ----------------------------------------------------------


def test_tool_call_then_answer_is_recorded_in_order(registry: MockToolRegistry) -> None:
    task = _task()
    answer = "I've checked order NW-1001 — it was delivered on the 18th. Shall I refund it?"
    script = [
        (
            200,
            _body(
                tool_calls=[_tool_call("call_1", "lookup_order", '{"order_id": "NW-1001"}')],
                prompt_tokens=100,
                completion_tokens=20,
                finish_reason="tool_calls",
            ),
        ),
        (200, _body(content=answer, prompt_tokens=150, completion_tokens=30)),
    ]

    with _scripted_agent(script, registry) as (agent, handler):
        result = agent.run(task)

    assert result.status == "completed"
    assert _types(result) == [
        "system",
        "user_message",
        "tool_call",
        "tool_result",
        "assistant_message",
    ]
    assert result.steps[0].content == {"text": DEFAULT_SYSTEM_PROMPT}
    assert result.steps[1].content == {"text": task.user_message}
    assert result.steps[2].content == {
        "name": "lookup_order",
        "arguments": {"order_id": "NW-1001"},
    }
    assert result.steps[3].content == {
        "name": "lookup_order",
        "result": ORDER_RESULT,
        "is_error": False,
    }
    assert result.steps[4].content == {"text": answer}

    # Tokens are summed across every turn, not just the last one.
    assert result.input_tokens == 250
    assert result.output_tokens == 50
    assert result.agent_config == {
        "adapter": "groq-agent",
        "model": "test-model",
        "max_turns": 8,
        "temperature": 0.0,
    }
    assert result.meta["turns"] == 2
    assert result.meta["tool_calls"] == 1
    assert result.meta["tool_errors"] == 0
    assert result.meta["tool_call_counts"] == {"lookup_order": 1}
    assert result.meta["finish_reason"] == "stop"
    assert result.started_at <= result.completed_at

    # The second turn feeds the tool output back as a role="tool" message.
    assert len(handler.payloads) == 2
    tool_message = handler.payloads[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert json.loads(tool_message["content"]) == ORDER_RESULT


def test_tools_are_wrapped_and_the_expected_outcome_never_leaves_the_task(
    registry: MockToolRegistry,
) -> None:
    refund_spec = {"name": "refund_order", "parameters": {"type": "object"}}
    task = _task(
        tools_spec=[
            LOOKUP_SPEC,
            {"type": "function", "function": refund_spec},  # already wrapped: passes through
            {"description": "nameless specs are dropped rather than sent"},
            "not-a-spec",
        ]
    )

    with _scripted_agent([(200, _body(content="How can I help?"))], registry) as (agent, handler):
        result = agent.run(task)

    assert result.status == "completed"
    payload = handler.payloads[0]
    assert payload["tools"] == [
        {"type": "function", "function": LOOKUP_SPEC},
        {"type": "function", "function": refund_spec},
    ]
    assert payload["model"] == "test-model"
    assert payload["temperature"] == 0.0
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1] == {"role": "user", "content": task.user_message}

    assert EXPECTED_OUTCOME not in DEFAULT_SYSTEM_PROMPT
    assert EXPECTED_OUTCOME not in payload["messages"][0]["content"]
    assert EXPECTED_OUTCOME not in json.dumps(payload)


def test_wrap_tools_spec_tolerates_junk() -> None:
    assert wrap_tools_spec(None) == []
    assert wrap_tools_spec({"name": "not-a-list"}) == []
    assert wrap_tools_spec([{"name": "  "}, {"type": "function", "function": "junk"}]) == []
    assert wrap_tools_spec([{"name": "ping"}]) == [
        {"type": "function", "function": {"name": "ping"}}
    ]


# --- failure modes -----------------------------------------------------------


def test_exhausting_max_turns_yields_truncated(registry: MockToolRegistry) -> None:
    script = [
        (
            200,
            _body(
                tool_calls=[_tool_call("call_1", "lookup_order", '{"order_id": "NW-1001"}')],
                finish_reason="tool_calls",
            ),
        )
    ]

    with _scripted_agent(script, registry, repeat_last=True, max_turns=2) as (agent, handler):
        result = agent.run(_task())

    assert result.status == "truncated"
    assert len(handler.payloads) == 2
    assert _types(result) == [
        "system",
        "user_message",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
    ]
    assert result.meta["turns"] == 2
    assert result.meta["max_turns"] == 2
    assert result.meta["tool_call_counts"] == {"lookup_order": 2}
    assert "error" not in result.meta


def test_client_error_yields_error_status_and_keeps_partial_steps(
    registry: MockToolRegistry,
) -> None:
    script = [
        (
            200,
            _body(
                tool_calls=[_tool_call("call_1", "lookup_order", '{"order_id": "NW-1001"}')],
                prompt_tokens=100,
                completion_tokens=20,
                finish_reason="tool_calls",
            ),
        ),
        (400, {"error": {"message": "context length exceeded"}}),
    ]

    with _scripted_agent(script, registry) as (agent, handler):
        result = agent.run(_task())

    assert result.status == "error"
    # The work done before the failure is still recorded and still worth judging.
    assert _types(result) == ["system", "user_message", "tool_call", "tool_result"]
    assert result.steps[3].content["result"] == ORDER_RESULT
    assert "400" in result.meta["error"]
    assert result.input_tokens == 100
    assert result.output_tokens == 20
    assert len(handler.payloads) == 2


def test_malformed_tool_arguments_come_back_as_a_tool_error(
    registry: MockToolRegistry,
) -> None:
    script = [
        (
            200,
            _body(
                tool_calls=[_tool_call("call_1", "lookup_order", "{order_id: NW-1001")],
                finish_reason="tool_calls",
            ),
        ),
        (200, _body(content="Sorry, could you confirm the order id?")),
    ]

    with _scripted_agent(script, registry) as (agent, handler):
        result = agent.run(_task())

    # A model that emits invalid JSON gets a correctable tool error, not a dead run.
    assert result.status == "completed"
    assert _types(result) == [
        "system",
        "user_message",
        "tool_call",
        "tool_result",
        "assistant_message",
    ]
    assert result.steps[2].content == {"name": "lookup_order", "arguments": {}}
    tool_result = result.steps[3].content
    assert tool_result["is_error"] is True
    assert tool_result["result"]["error"] == "invalid_arguments"
    assert tool_result["result"]["received"] == "{order_id: NW-1001"
    assert result.meta["tool_errors"] == 1
    assert "invalid_arguments" in handler.payloads[1]["messages"][-1]["content"]


def test_unknown_tool_is_reported_back_to_the_model(registry: MockToolRegistry) -> None:
    script = [
        (
            200,
            _body(
                tool_calls=[_tool_call("call_1", "cancel_order", '{"order_id": "NW-1001"}')],
                finish_reason="tool_calls",
            ),
        ),
        (200, _body(content="I can't cancel that, but I can refund it.")),
    ]

    with _scripted_agent(script, registry) as (agent, _handler):
        result = agent.run(_task())

    assert result.status == "completed"
    assert result.steps[3].content["is_error"] is True
    assert result.steps[3].content["result"]["error"] == "unknown tool: cancel_order"
    assert result.meta["tool_errors"] == 1


def test_max_turns_must_be_positive(registry: MockToolRegistry) -> None:
    with pytest.raises(ValueError, match="max_turns"):
        GroqReferenceAgent(model="test-model", registry=registry, max_turns=0)


# --- opening_message ---------------------------------------------------------


def test_opening_message_prefers_the_user_message() -> None:
    task = _task(user_message="  Hi, my order never turned up.  ")
    assert opening_message(task) == "Hi, my order never turned up."


def test_opening_message_falls_back_to_the_prompt() -> None:
    prompt = "Grader view: the agent should apologise and refund."
    assert opening_message(_task(prompt=prompt, user_message=None)) == prompt
    assert opening_message(_task(prompt=prompt, user_message="   ")) == prompt
