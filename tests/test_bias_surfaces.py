"""The probe's three surfaces: the JSON endpoint, the calibration page, and the CLI.

The endpoint and the page read stored rows and nothing else — a page refresh must
never start a sweep that costs tokens — so their tests seed a probe run first and
then assert what a caller receives. The empty states get their own tests for the
reason the calibration surfaces do: rendering zeros for a probe nobody ran reads as
a judge that passed a test that was never administered.

The command is driven end to end against a throwaway SQLite file, because a command
that works only against an injected session is not the command anybody runs. Its
model client is replaced with an ``httpx`` mock transport, so the sweep is real and
the network is not. One assertion there is a house rule rather than a behaviour:
everything the command prints must be ASCII, because Windows consoles mojibake the
typography that reads well in a docstring.
"""

import itertools
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner, Result

from agentverdict.bias.perturb import FILLER, LENGTH_VARIANTS, ORDER_VARIANTS
from agentverdict.bias.runner import run_probe
from agentverdict.cli import app as cli_app
from agentverdict.config import get_settings
from agentverdict.db import get_session_factory, reset_engine
from agentverdict.judging.client import GroqChatClient
from agentverdict.migrate import upgrade_to_head
from agentverdict.models import (
    BiasProbeResult,
    BiasProbeRun,
    Judge,
    Step,
    Task,
    Trajectory,
)

runner = CliRunner()

#: The corpus every CLI test probes: three scenarios, one recorded run each, each
#: with an agent turn so the length probe has something to pad.
SCENARIOS: tuple[str, ...] = ("refund-simple", "order-status", "damaged-item")


def _decision(verdict: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "verdict": verdict,
                            "rationale": "Scripted: the agent verified the order first.",
                            "rubric_scores": {},
                        }
                    )
                }
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def _scripted_client(answer: Callable[[str], str]) -> GroqChatClient:
    """A judge whose verdict is a function of the prompt it was handed."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(200, json=_decision(answer(payload["messages"][1]["content"])))

    return GroqChatClient(
        api_key="test-key",
        base_url="https://groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )


def _grades_padding_up(user_message: str) -> str:
    """Borderline normally, pass once filler is present: a length-sensitive judge."""
    return "pass" if any(sentence in user_message for sentence in FILLER) else "borderline"


def _steady(user_message: str) -> str:
    return "pass"


def _dies_on_call(number: int) -> Callable[[str], str]:
    """A judge whose transport drops partway through the sweep.

    ``RuntimeError`` and deliberately not ``JudgeClientError``: the client's own
    errors are caught per call and recorded as error rows, so only something the
    runner does not expect reaches the handler that closes the run as failed.
    """
    calls = itertools.count(1)

    def answer(user_message: str) -> str:
        if next(calls) == number:
            raise RuntimeError("the provider dropped the connection")
        return "pass"

    return answer


def _seed_corpus(session: Session, *, judge_name: str = "probe-judge") -> Judge:
    """One judge and the SCENARIOS corpus, committed."""
    judge = Judge(name=judge_name, model="llama-test-model")
    session.add(judge)
    for position, key in enumerate(SCENARIOS):
        task = Task(
            key=key,
            prompt=f"Scenario {position}: help the customer with {key.replace('-', ' ')}.",
            expected_outcome="The customer's problem is resolved.",
            tools_spec=[{"name": "lookup_order"}],
        )
        session.add(task)
        session.flush()
        session.add(
            Trajectory(
                task_id=task.id,
                source="import",
                steps=[
                    Step(index=0, type="user_message", content={"text": f"Hi, about {key}."}),
                    Step(
                        index=1,
                        type="tool_call",
                        content={"name": "lookup_order", "arguments": {"order_id": key}},
                    ),
                    Step(
                        index=2,
                        type="tool_result",
                        content={
                            "name": "lookup_order",
                            "result": {"status": "delivered"},
                            "is_error": False,
                        },
                    ),
                    Step(
                        index=3,
                        type="assistant_message",
                        content={"text": "I have checked the order and it was delivered."},
                    ),
                ],
            )
        )
    session.commit()
    return judge


# --- GET /api/judges/{id}/bias -------------------------------------------------


@pytest.fixture
def probed_judge(session: Session) -> Judge:
    """A judge with one stored length probe: three scenarios, all graded up by padding."""
    judge = _seed_corpus(session)
    with _scripted_client(_grades_padding_up) as client:
        run_probe(session, judge, "length", repeats=2, client=client)
    session.commit()
    return judge


def test_the_bias_endpoint_reports_a_stored_probe(
    client: TestClient, probed_judge: Judge
) -> None:
    """Everything the intervals rest on travels with the report.

    A caller must be able to check the finding rather than trust it, which means the
    cluster count, the repeat count, the mover threshold and the judge's own noise
    floor all come back beside the shift.
    """
    response = client.get(f"/api/judges/{probed_judge.id}/bias", params={"probe": "length"})

    assert response.status_code == 200
    report = response.json()
    assert report["judge_id"] == probed_judge.id
    assert report["probe"] == "length"
    assert report["trajectories"] == len(SCENARIOS)
    assert report["tasks"] == len(SCENARIOS)  # the n behind every interval
    assert report["repeats"] == 2
    assert report["control_disagreement"] == 0.0
    assert report["min_movers"] == 3
    assert report["adjusted_confidence"] > report["confidence"]
    arms = {arm["variant"]: arm for arm in report["arms"]}
    assert sorted(arms) == ["padded_1x", "padded_2x", "padded_3x"]
    # Three of three trajectories moved, against a floor of three, and the interval
    # collapsed onto the shift because every trajectory moved by the same amount.
    assert arms["padded_2x"]["signed_shift"] == 0.5
    assert arms["padded_2x"]["movers_observed"] == len(SCENARIOS)
    assert arms["padded_2x"]["outcome"] == "biased"


def test_the_bias_endpoint_keeps_the_two_probes_apart(
    client: TestClient, session: Session, probed_judge: Judge
) -> None:
    """One report describes one probe: their arms are not comparable.

    The length arms would look like missing order arms and the order arms like
    unrun length arms, and a reader would have no way to tell which.
    """
    with _scripted_client(_steady) as scripted:
        run_probe(session, probed_judge, "order", repeats=2, client=scripted)
    session.commit()

    order = client.get(f"/api/judges/{probed_judge.id}/bias", params={"probe": "order"}).json()
    length = client.get(f"/api/judges/{probed_judge.id}/bias", params={"probe": "length"}).json()

    assert [arm["variant"] for arm in order["arms"]] == [
        variant for variant in ORDER_VARIANTS if variant != "identity"
    ]
    assert [arm["variant"] for arm in length["arms"]] == [
        variant for variant in LENGTH_VARIANTS if variant != "identity"
    ]
    assert all(arm["outcome"] == "inconclusive" for arm in order["arms"])


def test_the_bias_endpoint_defaults_to_the_order_probe(
    client: TestClient, probed_judge: Judge
) -> None:
    response = client.get(f"/api/judges/{probed_judge.id}/bias")

    assert response.status_code == 200
    assert response.json()["probe"] == "order"


def test_the_bias_endpoint_empty_state_is_a_200_with_nulls(
    client: TestClient, make_judge: Callable[..., Judge]
) -> None:
    """A judge nobody has probed has no findings, which is a state and not a fault.

    Zeros here would render as a judge that held steady under every perturbation.
    """
    judge = make_judge(name="unprobed-judge", model="llama-test-model")

    report = client.get(f"/api/judges/{judge.id}/bias", params={"probe": "length"}).json()

    assert report["repeats"] == 0
    assert report["trajectories"] == 0
    assert report["arms"] == []
    assert report["control_disagreement"] is None
    assert report["min_movers"] is None


def test_the_bias_endpoint_unknown_judge_returns_404(client: TestClient) -> None:
    response = client.get("/api/judges/no-such-judge/bias")

    assert response.status_code == 404
    assert "no-such-judge" in response.json()["detail"]


def test_the_bias_endpoint_rejects_an_unknown_probe_name(
    client: TestClient, make_judge: Callable[..., Judge]
) -> None:
    """A typo must not be answered with an empty report.

    "No bias detected" and "you asked for a probe that does not exist" are different
    answers, and only one of them is about the judge.
    """
    judge = make_judge(name="probe-typo-judge", model="llama-test-model")

    response = client.get(f"/api/judges/{judge.id}/bias", params={"probe": "verbosity"})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "verbosity" in detail
    assert "order" in detail and "length" in detail


# --- the calibration page ------------------------------------------------------


def test_the_calibration_page_carries_the_bias_section(
    client: TestClient, probed_judge: Judge
) -> None:
    """The page links to the stored probe rather than computing one.

    Opening a page must never call a model. The panel fetches the endpoint above in
    the browser, so the server-rendered HTML carries the links and the explanation,
    and the command that records a probe is named for a reader who has none.
    """
    response = client.get(f"/calibration/{probed_judge.id}")

    assert response.status_code == 200
    body = response.text
    assert "Bias probes" in body
    assert f"/api/judges/{probed_judge.id}/bias?probe=order" in body
    assert f"/api/judges/{probed_judge.id}/bias?probe=length" in body
    assert f"agentverdict probe {probed_judge.name}" in body


# --- agentverdict probe --------------------------------------------------------


@pytest.fixture
def cli_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[sessionmaker[Session]]:
    """Point the application's own settings at a scratch SQLite file, then restore.

    The CLI resolves its database through ``get_settings``/``get_engine``, so the only
    honest way to drive it is to move those globals and put them back.
    """
    url = f"sqlite:///{(tmp_path / 'probe.db').as_posix()}"
    monkeypatch.setenv("AGENTVERDICT_DATABASE_URL", url)
    get_settings.cache_clear()
    reset_engine()
    try:
        upgrade_to_head(url)
        yield get_session_factory()
    finally:
        reset_engine()
        get_settings.cache_clear()


@pytest.fixture
def cli_corpus(cli_db: sessionmaker[Session]) -> Judge:
    session = cli_db()
    try:
        return _seed_corpus(session)
    finally:
        session.close()


def _install_client(monkeypatch: pytest.MonkeyPatch, answer: Callable[[str], str]) -> None:
    """Hand the runner a mock-transport client instead of a real one.

    ``run_probe`` builds its own client when the caller passes none, which is exactly
    what the command does — so this is the seam that keeps the command under test and
    the network out of it.
    """
    monkeypatch.setattr(
        "agentverdict.bias.runner.GroqChatClient",
        lambda *args, **kwargs: _scripted_client(answer),
    )


def _install_rejecting_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider that rejects the padded prompts and answers every other one.

    The length probe's expected failure: the padded arms carry the longest transcripts,
    so a context-length rejection takes exactly one side of every pair and leaves the
    control untouched.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        user = json.loads(request.content)["messages"][1]["content"]
        if any(sentence in user for sentence in FILLER):
            return httpx.Response(400, json={"error": {"message": "context_length_exceeded"}})
        return httpx.Response(200, json=_decision("pass"))

    monkeypatch.setattr(
        "agentverdict.bias.runner.GroqChatClient",
        lambda *args, **kwargs: GroqChatClient(
            api_key="test-key",
            base_url="https://groq.test/openai/v1",
            transport=httpx.MockTransport(handler),
        ),
    )


def _probe(*args: str) -> Result:
    return runner.invoke(cli_app, ["probe", *args])


def test_probe_refuses_one_repeat_with_a_message_and_not_a_traceback(
    cli_corpus: Judge,
) -> None:
    """The guard runs before the database is opened, so a typo costs nothing.

    At one draw per arm the identity arm has no within-arm pair, so the judge's own
    noise floor is undefined and every shift would be measured against nothing.
    """
    result = _probe(cli_corpus.name, "--repeats", "1")

    assert result.exit_code == 1
    assert "--repeats must be at least 2" in result.output
    assert "noise floor" in result.output
    assert "Traceback" not in result.output


def test_probe_refuses_an_unknown_probe_name(cli_corpus: Judge) -> None:
    result = _probe(cli_corpus.name, "--probe", "self-preference")

    assert result.exit_code == 1
    assert "Unknown probe 'self-preference'" in result.output
    assert "order" in result.output and "length" in result.output
    assert "Traceback" not in result.output


def test_probe_names_the_command_that_registers_a_missing_judge(
    cli_db: sessionmaker[Session],
) -> None:
    """The empty state names the command that fixes it, as calibrate does."""
    result = _probe("nobody")

    assert result.exit_code == 1
    assert "agentverdict judge add nobody" in result.output


def test_probe_prints_an_ascii_report_a_reader_can_check(
    cli_corpus: Judge, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole text report, on a judge that grades padded transcripts higher.

    Everything the finding rests on is printed beside it: the cluster count, the
    judge's own test-retest floor, the movers observed against the movers needed, and
    the multiplicity correction the intervals were read at.

    ASCII because the console this runs on is often a Windows one, where the em-dashes
    and curly quotes that read well in a docstring arrive as mojibake.
    """
    _install_client(monkeypatch, _grades_padding_up)

    result = _probe(cli_corpus.name, "--probe", "length", "--repeats", "2")

    assert result.exit_code == 0, result.output
    assert result.output.isascii()
    assert f"Bias probe: judge '{cli_corpus.name}'  [length sensitivity]" in result.output
    assert f"sample     {len(SCENARIOS)} trajectories over {len(SCENARIOS)} tasks" in result.output
    assert "intervals resample tasks" in result.output
    assert "Judge test-retest (2 byte-identical calls per trajectory)" in result.output
    assert "padded_1x: SENSITIVE, upward" in result.output
    assert "3 of 3 measured trajectories moved (3 needed)" in result.output
    # Length sensitivity is not assumed to be a defect: the rubric licenses some.
    assert "The rubric licenses some of this." in result.output
    assert "Sidak over 4 intervals" in result.output


def test_probe_reports_an_unmoved_judge_as_not_measured(
    cli_corpus: Judge, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three trajectories, nothing moved, and the report refuses to call that clean.

    "Read it as 'not measured'. It is not 'unbiased'." is the whole point of printing
    the mover threshold: at this n an effect smaller than three trajectories cannot be
    seen, so silence is a property of the instrument.
    """
    _install_client(monkeypatch, _steady)

    result = _probe(cli_corpus.name, "--probe", "order", "--repeats", "2")

    assert result.exit_code == 0, result.output
    assert result.output.isascii()
    assert "INCONCLUSIVE" in result.output
    assert "It is not 'unbiased'." in result.output
    assert "SENSITIVE" not in result.output


def test_probe_separates_an_arm_that_answered_nothing_from_one_that_did_not_move(
    cli_corpus: Judge, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every padded call is rejected, so no arm has a mover count to print.

    The perturbation was delivered on all three trajectories, so this is not the
    ``not_applied`` sentence about a renderer that produced the control's bytes. It is
    also not the mover sentence: with nothing measured there is no denominator for one,
    and "0 of 3 trajectories moved" would describe a judge that never answered as a
    judge that shrugged.
    """
    _install_rejecting_client(monkeypatch)

    result = _probe(cli_corpus.name, "--probe", "length", "--repeats", "2")

    assert result.exit_code == 0, result.output
    assert result.output.isascii()
    assert "padded_1x: NOT MEASURED" in result.output
    assert f"reached the model on {len(SCENARIOS)} trajectories" in result.output
    assert "measured trajectories moved" not in result.output
    assert "NOT APPLIED" not in result.output
    # The failed calls are what a reader has to act on, so they are still disclosed.
    assert "produced no verdict" in result.output


def test_probe_json_prints_the_report_and_nothing_else(
    cli_corpus: Judge, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--json`` is what a script reads, so nothing may precede the document."""
    _install_client(monkeypatch, _grades_padding_up)

    result = _probe(cli_corpus.name, "--probe", "length", "--repeats", "2", "--json")

    assert result.exit_code == 0, result.output
    assert result.output.strip().startswith("{")
    report = json.loads(result.output)
    assert report["probe"] == "length"
    assert report["tasks"] == len(SCENARIOS)
    assert report["min_movers"] == 3
    assert {arm["outcome"] for arm in report["arms"]} == {"biased"}


def test_probe_keeps_the_calls_it_paid_for_when_the_sweep_dies(
    cli_db: sessionmaker[Session], cli_corpus: Judge, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-sweep must not refund nothing, and must not leave the run "running".

    This is the assertion that has to be made through the command rather than through
    an injected session. Every surface rolls its session back on the way out of an
    exception, which is right for the commands that only compute and wrong for the one
    that spends: driven from a test that never rolls back, a runner which merely
    flushed would look like it kept its rows while the real command threw away every
    call it had already bought.
    """
    _install_client(monkeypatch, _dies_on_call(5))

    result = _probe(cli_corpus.name, "--probe", "length", "--repeats", "2")

    assert result.exit_code != 0
    session = cli_db()
    try:
        run = session.scalars(select(BiasProbeRun)).one()
        # "failed" and not "running": a run left mid-status is indistinguishable from
        # one still in flight, and it is the last write of the sweep, so it is the one
        # write no per-row commit carries.
        assert run.status == "failed"
        assert run.completed_at is not None
        # Four calls landed before the fifth died, and all four are on record.
        assert session.scalar(select(func.count()).select_from(BiasProbeResult)) == 4
    finally:
        session.close()


def test_probe_limits_the_sweep_without_pretending_it_probed_more(
    cli_corpus: Judge, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--limit`` is how a first run is made cheap, and the report says what it cost.

    Both counts printed are the ones the run actually rests on, never the corpus it
    was drawn from: two of the three stored trajectories, in two of the three tasks.
    A capped run that quoted the full corpus would advertise independence it did not
    buy, which at these sample sizes is the difference between an interval worth
    reading and one that is a lattice.
    """
    _install_client(monkeypatch, _grades_padding_up)

    result = _probe(cli_corpus.name, "--probe", "length", "--repeats", "2", "--limit", "2")

    assert result.exit_code == 0, result.output
    assert "2 trajectories x 4 arms x 2 repeats" in result.output
    assert "sample     2 trajectories over 2 tasks" in result.output
    assert f"over {len(SCENARIOS)} tasks" not in result.output
