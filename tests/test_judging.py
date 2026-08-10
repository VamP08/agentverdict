"""Judging tests: Groq client retries, prompt rendering, and the eval runner (no network)."""

import json
from collections.abc import Callable

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agentverdict.judging.client import GroqChatClient, JudgeClientError
from agentverdict.judging.prompts import PROMPT_VERSION, build_messages, render_transcript
from agentverdict.judging.runner import parse_decision, run_eval
from agentverdict.models import HumanLabel, Judge, JudgeVerdict, Task, Trajectory


def _chat_body(content: str, prompt_tokens: int = 100, completion_tokens: int = 20) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _decision(verdict: str = "pass", rationale: str = "Correct tool use throughout.") -> str:
    return json.dumps({"verdict": verdict, "rationale": rationale, "rubric_scores": {}})


def _make_client(handler: Callable[[httpx.Request], httpx.Response]) -> GroqChatClient:
    return GroqChatClient(
        api_key="test-key",
        base_url="https://groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )


MESSAGES = [{"role": "user", "content": "judge this"}]


# --- client ------------------------------------------------------------------


def test_chat_json_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        payload = json.loads(request.content)
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["model"] == "test-model"
        return httpx.Response(200, json=_chat_body(_decision()))

    with _make_client(handler) as client:
        result = client.chat_json("test-model", MESSAGES)
    assert json.loads(result.content)["verdict"] == "pass"
    assert result.input_tokens == 100
    assert result.output_tokens == 20
    assert result.latency_ms >= 0


def test_chat_json_retries_on_429_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "rate"})
        return httpx.Response(200, json=_chat_body(_decision()))

    with _make_client(handler) as client:
        result = client.chat_json("test-model", MESSAGES)
    assert calls["n"] == 2
    assert json.loads(result.content)["verdict"] == "pass"


def test_chat_json_does_not_sleep_out_a_long_retry_after() -> None:
    """A server that asks for five minutes must not turn the client into a hang.

    Sleeping for exactly as long as asked is indistinguishable from a stall from the
    outside: no output, no error, a process that looks alive. The wait is capped and the
    exhausted run says it was rate limited.
    """
    from agentverdict.judging.client import MAX_RETRY_DELAY_S

    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "300"}, json={"error": "rate"})

    with _make_client(handler) as client:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("agentverdict.judging.client.time.sleep", slept.append)
            with pytest.raises(JudgeClientError) as raised:
                client.chat_json("test-model", MESSAGES)

    assert slept, "the client should still back off"
    assert max(slept) <= MAX_RETRY_DELAY_S
    message = str(raised.value)
    assert "rate limited" in message
    assert "300s" in message  # the server's own figure, so the cause is not guesswork


def test_chat_json_stops_immediately_when_the_server_forbids_retrying() -> None:
    """A daily token budget is not a burst: retrying only delays the bad news."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            429,
            headers={"retry-after": "374", "x-should-retry": "false"},
            json={"error": {"message": "tokens per day (TPD): Limit 100000, Used 99290"}},
        )

    with _make_client(handler) as client:
        with pytest.raises(JudgeClientError) as raised:
            client.chat_json("test-model", MESSAGES)

    assert calls["n"] == 1, "the server said a retry would not help"
    assert "tokens per day" in str(raised.value)  # the operator needs the actual limit


def test_chat_json_client_error_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    with _make_client(handler) as client:
        with pytest.raises(JudgeClientError, match="400"):
            client.chat_json("test-model", MESSAGES)
    assert calls["n"] == 1


def test_chat_json_gives_up_after_max_attempts() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, headers={"retry-after": "0"}, json={"error": "boom"})

    with _make_client(handler) as client:
        with pytest.raises(JudgeClientError, match="500"):
            client.chat_json("test-model", MESSAGES)
    assert calls["n"] == GroqChatClient.MAX_ATTEMPTS


# --- prompts and decision parsing -------------------------------------------


def test_build_messages_includes_task_and_transcript(
    session: Session,
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    task = make_task(
        key="judging-prompt-task",
        prompt="Refund the customer politely.",
        expected_outcome="Refund issued after verifying the order.",
        tools_spec=[{"name": "refund_order"}],
    )
    trajectory = make_trajectory(task=task)

    messages = build_messages(task, trajectory)
    assert messages[0]["role"] == "system"
    assert '"verdict"' in messages[0]["content"]
    body = messages[1]["content"]
    assert "Refund the customer politely." in body
    assert "Refund issued after verifying the order." in body
    assert "refund_order" in body
    assert "[0] USER:" in body
    assert "TOOL CALL lookup_order" in body
    assert "TOOL RESULT lookup_order" in body


def test_render_transcript_marks_tool_errors(
    make_trajectory: Callable[..., Trajectory],
) -> None:
    trajectory = make_trajectory(
        steps=[
            ("tool_call", {"name": "lookup_order", "arguments": {"order_id": "X"}}),
            ("tool_result", {"name": "lookup_order", "result": "timeout", "is_error": True}),
        ]
    )
    transcript = render_transcript(trajectory)
    assert "TOOL RESULT ERROR lookup_order" in transcript


def test_parse_decision_valid_and_invalid() -> None:
    decision = parse_decision(_decision("borderline"))
    assert decision.verdict == "borderline"
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_decision("not json at all")
    with pytest.raises(ValueError, match="decision contract"):
        parse_decision(json.dumps({"verdict": "excellent", "rationale": "nope"}))


# --- runner ------------------------------------------------------------------


@pytest.fixture
def judge(session: Session) -> Judge:
    row = Judge(name="test-judge", model="test-model")
    session.add(row)
    session.commit()
    return row


def test_run_eval_happy_path(
    session: Session,
    judge: Judge,
    make_trajectory: Callable[..., Trajectory],
) -> None:
    for _ in range(3):
        make_trajectory()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_body(_decision()))

    with _make_client(handler) as client:
        run = run_eval(session, judge, client=client)

    assert run.status == "completed"
    assert run.completed_at is not None
    assert run.trajectory_count == 3
    assert run.error_count == 0
    assert run.verdict_counts == {"pass": 3}
    assert run.input_tokens == 300
    assert run.output_tokens == 60
    # Provenance: which rubric produced these verdicts. The regression gate refuses to
    # compare two runs stamped differently, so an unstamped run silently disables it.
    assert run.judge_prompt_version == PROMPT_VERSION
    verdicts = list(session.scalars(select(JudgeVerdict)))
    assert len(verdicts) == 3
    assert all(v.verdict == "pass" and v.rationale for v in verdicts)


def test_prompt_version_tracks_the_prompt_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Editing the rubric must move the fingerprint, or the gate compares across rubrics."""
    from agentverdict.judging import prompts

    monkeypatch.setattr(prompts, "SYSTEM_PROMPT", prompts.SYSTEM_PROMPT + "\nAlso be lenient.")
    assert prompts._recompute_version() != PROMPT_VERSION


def test_prompt_version_covers_every_section_the_judge_reads(
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """The fingerprint's copy of the user-message headers must match the real ones.

    ``_USER_PROMPT_SHAPE`` is a hand-written transcription of what ``build_messages``
    emits. If someone renames a section there and not here, the rubric changes and the
    fingerprint does not -- which is the exact failure the fingerprint exists to stop.
    """
    from agentverdict.judging.prompts import _USER_PROMPT_SHAPE

    task = make_task(expected_outcome="The refund is issued.", tools_spec=[{"name": "refund"}])
    messages = build_messages(task, make_trajectory(task=task))
    user_message = messages[1]["content"]

    for section in _USER_PROMPT_SHAPE.split("|"):
        assert section in user_message


def test_run_eval_retries_malformed_judge_output(
    session: Session,
    judge: Judge,
    make_trajectory: Callable[..., Trajectory],
) -> None:
    make_trajectory()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json=_chat_body("sorry, no JSON today"))
        payload = json.loads(request.content)
        assert payload["messages"][-1]["content"].startswith("Your previous answer")
        return httpx.Response(200, json=_chat_body(_decision("fail", "Wrong tool order.")))

    with _make_client(handler) as client:
        run = run_eval(session, judge, client=client)

    assert calls["n"] == 2
    assert run.error_count == 0
    assert run.verdict_counts == {"fail": 1}
    verdict = session.scalars(select(JudgeVerdict)).one()
    assert verdict.verdict == "fail"
    assert verdict.input_tokens == 200  # both calls accounted
    assert verdict.output_tokens == 40


def test_run_eval_records_errors_and_keeps_going(
    session: Session,
    judge: Judge,
    make_trajectory: Callable[..., Trajectory],
) -> None:
    make_trajectory()
    make_trajectory()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(400, json={"error": {"message": "bad"}})
        return httpx.Response(200, json=_chat_body(_decision()))

    with _make_client(handler) as client:
        run = run_eval(session, judge, client=client)

    assert run.trajectory_count == 2
    assert run.error_count == 1
    assert run.verdict_counts == {"pass": 1}
    errored = session.scalars(select(JudgeVerdict).where(JudgeVerdict.error.is_not(None))).one()
    assert errored.verdict is None
    assert "400" in errored.error


def test_run_eval_computes_human_agreement(
    session: Session,
    judge: Judge,
    make_trajectory: Callable[..., Trajectory],
) -> None:
    agreeing = make_trajectory()
    disagreeing = make_trajectory()
    unlabeled = make_trajectory()
    session.add_all(
        [
            HumanLabel(trajectory_id=agreeing.id, annotator="alice", verdict="pass"),
            HumanLabel(trajectory_id=disagreeing.id, annotator="alice", verdict="fail"),
        ]
    )
    session.commit()
    assert unlabeled.labels == []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_body(_decision("pass")))

    with _make_client(handler) as client:
        run = run_eval(session, judge, client=client)

    assert run.meta["human_agreement"] == {"agree": 1, "total": 2, "rate": 0.5}


def test_run_eval_respects_task_key_and_limit(
    session: Session,
    judge: Judge,
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    target = make_task(key="target-task")
    other = make_task(key="other-task")
    make_trajectory(task=target)
    make_trajectory(task=target)
    make_trajectory(task=other)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_body(_decision()))

    with _make_client(handler) as client:
        filtered = run_eval(session, judge, task_key="target-task", client=client)
        limited = run_eval(session, judge, limit=1, client=client)

    assert filtered.trajectory_count == 2
    assert filtered.task_key == "target-task"
    assert limited.trajectory_count == 1
    total_verdicts = session.scalar(select(func.count()).select_from(JudgeVerdict))
    assert total_verdicts == 3
