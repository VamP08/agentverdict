"""The probe run and the report read back off its rows. No network, no model.

The judge is an ``httpx.MockTransport`` behind the real ``GroqChatClient``, so the
retry, latency and token accounting under test are the ones production uses; only
the socket is missing. Every verdict below is scripted from the prompt the client was
handed, which is what lets a test say "this judge grades padded transcripts higher"
and then check that the report says so.

The load-bearing test in this file is
``test_a_probe_writes_no_judge_verdicts_and_leaves_the_trajectory_steps_byte_identical``.
A probe judges deliberately corrupted prompts. Two rows-worth of carelessness -- a
verdict landing in ``judge_verdicts``, or a padded step flushed back through the ORM
-- would put those judgements into calibration, into the merge gate, and into the
golden dataset permanently, where no export would flag them. It is asserted by
reading both tables before and after rather than by trusting the code path.
"""

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from agentverdict.bias.perturb import FILLER, LENGTH_VARIANTS, ORDER_VARIANTS
from agentverdict.bias.runner import (
    IDENTITY,
    MIN_REPEATS,
    TAIL_RESAMPLES,
    ZERO_RESIDUE,
    _ProbeRows,
    _score_arm,
    _tail_sized_iterations,
    build_bias_probe_report,
    run_probe,
)
from agentverdict.bias.stats import adjusted_confidence, min_movers
from agentverdict.judging.client import GroqChatClient
from agentverdict.judging.prompts import PROMPT_VERSION, build_messages
from agentverdict.judging.runner import run_eval
from agentverdict.models import (
    BiasProbeResult,
    BiasProbeRun,
    EvalRun,
    HumanLabel,
    Judge,
    JudgeVerdict,
    Step,
    Task,
    Trajectory,
)
from agentverdict.schemas import ArmOutcome, BiasProbeReport

#: Verdict answer signature: (user message, 0-based call index) -> verdict, where
#: ``None`` makes the call fail instead of returning a decision.
Answer = Callable[[str, int], str | None]


def _chat_body_of(content: str) -> dict[str, Any]:
    """A chat completion carrying ``content`` verbatim, so a test can send bad JSON."""
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def _answer(verdict: str) -> str:
    return json.dumps(
        {
            "verdict": verdict,
            "rationale": f"Scripted {verdict}: the tool sequence was checked.",
            "rubric_scores": {},
        }
    )


def _chat_body(verdict: str) -> dict[str, Any]:
    return _chat_body_of(_answer(verdict))


def _probe_client(answer: Answer) -> tuple[GroqChatClient, list[str]]:
    """A judge whose verdict is a function of the prompt, plus every prompt it saw.

    The captured prompts are how the perturbation is checked from the outside: a test
    can assert that filler reached the model and, separately, that it reached nothing
    else.
    """
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        # messages[1] is the probe's user turn in every call; a corrective retry only
        # appends to the list, so this stays the prompt under test.
        user = payload["messages"][1]["content"]
        prompts.append(user)
        verdict = answer(user, len(prompts) - 1)
        if verdict is None:
            return httpx.Response(400, json={"error": {"message": "judge unavailable"}})
        return httpx.Response(200, json=_chat_body(verdict))

    client = GroqChatClient(
        api_key="test-key",
        base_url="https://groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )
    return client, prompts


def _recording_client() -> tuple[GroqChatClient, list[dict[str, Any]]]:
    """A judge that fumbles its first answer, plus every request body it was sent.

    Whole bodies rather than prompts: how a judge is called is the model, the
    temperature and the response format as much as the text. The malformed opener is
    what puts the corrective retry -- the fiddliest part of the call -- on the record.
    """
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        content = "not json at all" if len(requests) == 1 else _answer("pass")
        return httpx.Response(200, json=_chat_body_of(content))

    client = GroqChatClient(
        api_key="test-key",
        base_url="https://groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    )
    return client, requests


def _is_padded(user_message: str) -> bool:
    return any(sentence in user_message for sentence in FILLER)


def _constant(verdict: str) -> Answer:
    """A judge that answers the same way whatever it is shown."""
    return lambda user, index: verdict


def _by_length(*, plain: str, padded: str) -> Answer:
    """A judge that grades padded transcripts differently from bare ones."""
    return lambda user, index: padded if _is_padded(user) else plain


def _steps_fingerprint(session: Session) -> list[str]:
    """Every ``trajectory_steps`` row, serialised so equality really is byte equality.

    ``Step.__tablename__`` is ``trajectory_steps``; ``content`` is JSON, so it is
    dumped with sorted keys rather than compared as a dict that might differ only in
    insertion order.
    """
    rows = session.execute(
        select(Step.id, Step.trajectory_id, Step.index, Step.type, Step.content).order_by(
            Step.trajectory_id, Step.index
        )
    ).all()
    return [json.dumps(list(row), sort_keys=True) for row in rows]


def _trajectories_fingerprint(session: Session) -> list[str]:
    rows = session.execute(
        select(Trajectory.id, Trajectory.task_id, Trajectory.status, Trajectory.meta).order_by(
            Trajectory.id
        )
    ).all()
    return [json.dumps(list(row), sort_keys=True) for row in rows]


def _arms(report: BiasProbeReport) -> dict[str, ArmOutcome]:
    return {arm.variant: arm for arm in report.arms}


@pytest.fixture
def judge(make_judge: Callable[..., Judge]) -> Judge:
    return make_judge(name="probe-judge", model="llama-test-model")


@pytest.fixture
def suite(
    make_task: Callable[..., Task], make_trajectory: Callable[..., Trajectory]
) -> Callable[..., list[Trajectory]]:
    """Store ``tasks`` scenarios with ``per_task`` trajectories each, one agent turn apiece.

    Each scenario gets its own prompt text. Two tasks worded identically would render
    identical judge messages, and a test that scripted the judge by prompt would then
    be scripting several trajectories at once without saying so.
    """

    def _make(tasks: int = 6, per_task: int = 1) -> list[Trajectory]:
        stored: list[Trajectory] = []
        for position in range(tasks):
            task = make_task(
                key=f"probe-task-{position:02d}",
                prompt=f"Scenario {position:02d}: refund the customer for a damaged item.",
                expected_outcome="Refunded once the order checks out.",
            )
            stored.extend(make_trajectory(task=task) for _ in range(per_task))
        return stored

    return _make


# --- refusals ------------------------------------------------------------------


def test_one_repeat_is_refused_before_anything_is_written(
    session: Session, judge: Judge, suite: Callable[..., list[Trajectory]]
) -> None:
    """At R=1 the identity arm has no within-arm pair, so the control measures nothing.

    Every shift a probe reports is read against that floor, so a run without one is
    not a cheaper experiment — it is a set of numbers with nothing under them. The
    refusal happens before the run row exists, so a rejected probe leaves no trace.
    """
    suite(tasks=2)
    client, prompts = _probe_client(_constant("pass"))

    with client, pytest.raises(ValueError, match="repeats must be at least 2"):
        run_probe(session, judge, "length", repeats=1, client=client)

    assert MIN_REPEATS == 2
    assert prompts == []
    assert session.scalar(select(func.count()).select_from(BiasProbeRun)) == 0


def test_an_unknown_probe_name_is_refused_before_anything_is_written(
    session: Session, judge: Judge, suite: Callable[..., list[Trajectory]]
) -> None:
    """The error names the probes that exist, because the caller mistyped one of them."""
    suite(tasks=2)
    client, prompts = _probe_client(_constant("pass"))

    with client, pytest.raises(ValueError, match="unknown probe 'verbosity'") as raised:
        run_probe(session, judge, "verbosity", repeats=2, client=client)

    assert "order" in str(raised.value)
    assert "length" in str(raised.value)
    assert prompts == []
    assert session.scalar(select(func.count()).select_from(BiasProbeRun)) == 0


# --- the data-poisoning guard --------------------------------------------------


def test_a_probe_writes_no_judge_verdicts_and_leaves_the_trajectory_steps_byte_identical(
    session: Session, judge: Judge, suite: Callable[..., list[Trajectory]]
) -> None:
    """The guard the whole milestone is contingent on, asserted from the tables.

    A verdict on a padded transcript is not a verdict on the run. If one reached
    ``judge_verdicts`` it would become this judge's most recent opinion of that
    trajectory, and calibration reads exactly that — the probe would rewrite the
    kappa it exists to explain. If padding reached ``trajectory_steps`` it would be
    in the golden dataset permanently, and every future export would carry it.

    Both tables are read before and after, and the padding is confirmed to have
    reached the model in between, so this cannot pass by the probe doing nothing.
    """
    suite(tasks=3)
    steps_before = _steps_fingerprint(session)
    trajectories_before = _trajectories_fingerprint(session)
    assert session.scalar(select(func.count()).select_from(JudgeVerdict)) == 0

    client, prompts = _probe_client(_by_length(plain="borderline", padded="pass"))
    with client:
        run = run_probe(session, judge, "length", repeats=2, client=client)
    session.commit()
    session.expire_all()  # re-read the rows rather than the identity map's copy

    # The perturbation really was delivered: filler went out on the wire.
    assert any(_is_padded(prompt) for prompt in prompts)
    assert run.status == "completed"
    assert session.scalar(select(func.count()).select_from(BiasProbeResult)) > 0

    # ...and nowhere else. Not one verdict row, and not one eval run either: going
    # through run_eval would have created both.
    assert session.scalar(select(func.count()).select_from(JudgeVerdict)) == 0
    assert session.scalar(select(func.count()).select_from(EvalRun)) == 0

    assert _steps_fingerprint(session) == steps_before
    assert _trajectories_fingerprint(session) == trajectories_before
    stored_text = " ".join(_steps_fingerprint(session))
    for sentence in FILLER:
        assert sentence not in stored_text


def test_the_identity_arm_sends_the_production_prompt(
    session: Session, judge: Judge, make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """The byte-identity guarantee, checked where it actually matters: on the wire.

    ``bias/prompts.py`` is tested against ``build_messages`` directly, but that proves
    nothing about the runner unless the runner renders the control arm through the
    canonical plan. This asserts the bytes the client was handed.
    """
    task = make_task(key="identity-arm-task", expected_outcome="Refunded.", tools_spec=[{"n": 1}])
    trajectory = make_trajectory(task=task)
    client, prompts = _probe_client(_constant("pass"))

    with client:
        run_probe(session, judge, "order", repeats=2, client=client)

    production = build_messages(task, trajectory)[1]["content"]
    assert prompts[0] == production
    assert prompts.count(production) == 2  # both identity repeats, and no other arm


def test_the_probe_calls_the_judge_the_way_production_calls_it(
    session: Session,
    make_judge: Callable[..., Judge],
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """The identity arm's request equals ``run_eval``'s, whole body for whole body.

    The test above pins the prompt text. This pins everything else about the call --
    the model, the temperature read off ``judge.config``, the forced response format,
    and the corrective retry's message list -- because a probe that reported intervals
    for a judge invoked differently from the one grading in production would be
    measuring an artifact nobody ships. Both paths reach
    ``judging.runner.judge_messages``; this is what fails if either grows its own copy.

    The judge is given a non-default temperature deliberately: a hardcoded 0.0 on
    either side passes every other test in this file.
    """
    judge = make_judge(name="warm-judge", model="llama-test-model", config={"temperature": 0.7})
    task = make_task(key="call-shape-task", expected_outcome="Refunded.")
    make_trajectory(task=task)

    production_client, production_requests = _recording_client()
    with production_client:
        run_eval(session, judge, client=production_client)

    probe_client, probe_requests = _recording_client()
    with probe_client:
        run_probe(session, judge, "order", repeats=2, client=probe_client)

    # Two apiece for the first trajectory: the malformed opener and its correction.
    assert len(production_requests) == 2
    assert production_requests[0]["temperature"] == 0.7  # not the 0.0 default
    assert [m["role"] for m in production_requests[1]["messages"]][-2:] == ["assistant", "user"]
    assert probe_requests[:2] == production_requests


# --- what a run records --------------------------------------------------------


def test_the_run_records_the_unperturbed_rubric_and_the_arms_it_planned(
    session: Session, judge: Judge, suite: Callable[..., list[Trajectory]]
) -> None:
    """The artifact under test is the judge that grades in production.

    So the run is stamped with the unperturbed ``PROMPT_VERSION``; each arm's
    departure from it is named per row by ``variant`` and fingerprinted by
    ``prompt_sha``. ``limit`` and the arm list are recorded because neither is
    recoverable from the rows: five trajectories out of twenty-nine and five out of
    five leave identical row sets.
    """
    suite(tasks=2)
    client, _ = _probe_client(_constant("pass"))

    with client:
        run = run_probe(session, judge, "order", repeats=2, limit=1, client=client)

    assert run.judge_prompt_version == PROMPT_VERSION
    assert run.probe == "order"
    assert run.repeats == 2
    assert run.trajectory_count == 1
    assert run.error_count == 0
    assert run.status == "completed"
    assert run.completed_at is not None
    assert run.meta == {"limit": 1, "arms": list(ORDER_VARIANTS)}
    # Four arms, one trajectory, two repeats: eight calls at 100 in / 20 out.
    assert run.input_tokens == 800
    assert run.output_tokens == 160


def test_every_delivered_arm_is_judged_once_per_repeat(
    session: Session, judge: Judge, suite: Callable[..., list[Trajectory]]
) -> None:
    """The schedule is (trajectory, variant, repeat) and nothing about it is random.

    A run that dies partway is therefore a prefix of the run it would have been, which
    is what makes a killed sweep readable instead of a random subsample.
    """
    trajectories = suite(tasks=2)
    client, prompts = _probe_client(_constant("pass"))

    with client:
        run_probe(session, judge, "length", repeats=3, client=client)

    rows = list(session.scalars(select(BiasProbeResult)))
    assert len(rows) == len(trajectories) * len(LENGTH_VARIANTS) * 3
    assert len(prompts) == len(rows)
    assert sorted({row.repeat for row in rows}) == [0, 1, 2]
    assert all(row.verdict == "pass" and row.rationale for row in rows)
    assert all(row.error is None for row in rows)
    # Every repeat of an arm renders the same bytes, and no two arms of a trajectory
    # render the same bytes -- which is the delivery check, seen from the row side.
    for trajectory in trajectories:
        shas = {row.variant: row.prompt_sha for row in rows if row.trajectory_id == trajectory.id}
        assert len(set(shas.values())) == len(LENGTH_VARIANTS)


# --- the delivery check --------------------------------------------------------


def test_a_length_arm_with_nothing_to_pad_is_recorded_once_and_reported_not_applied(
    session: Session,
    judge: Judge,
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """A trajectory with no agent turn cannot be padded, so those arms measured nothing.

    Reporting that as ``inconclusive`` would read as "no length sensitivity detected"
    when the perturbation never happened — the single easiest way for this milestone
    to be worthless. It is ``not_applied``, and the undelivered arm costs one row
    rather than ``repeats`` calls, because the fact on record is that the
    perturbation did not exist here, not that some calls failed.
    """
    task = make_task(key="tools-only-task")
    make_trajectory(
        task=task,
        steps=[
            ("user_message", {"text": "Where is order A123?"}),
            ("tool_call", {"name": "lookup_order", "arguments": {"order_id": "A123"}}),
            (
                "tool_result",
                {"name": "lookup_order", "result": {"status": "lost"}, "is_error": False},
            ),
        ],
    )
    client, prompts = _probe_client(_constant("pass"))

    with client:
        run = run_probe(session, judge, "length", repeats=3, client=client)
    report = build_bias_probe_report(session, judge, probe="length", probe_run_id=run.id)

    assert not any(_is_padded(prompt) for prompt in prompts)
    assert len(prompts) == 3  # the identity arm only

    padded_rows = list(
        session.scalars(select(BiasProbeResult).where(BiasProbeResult.variant == "padded_2x"))
    )
    assert len(padded_rows) == 1
    assert padded_rows[0].verdict is None
    assert padded_rows[0].error is None  # an arm that never applied is not a failure
    assert run.error_count == 0
    assert report.errors == 0

    for variant in ("padded_1x", "padded_2x", "padded_3x"):
        arm = _arms(report)[variant]
        assert arm.outcome == "not_applied"
        assert arm.applied_to == 0
        assert arm.signed_shift is None
        assert arm.movers_observed == 0
        assert arm.power_limited is False


def test_an_order_arm_that_renders_the_control_bytes_is_reported_not_applied(
    session: Session,
    judge: Judge,
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """A task with no expected outcome and no tools has only two sections to order.

    ``reversed_context`` then renders exactly the control's bytes while
    ``context_last`` still moves the transcript, so one arm of the same run is
    ``not_applied`` and the other is measured. Two arms, one prompt: this is the case
    where a fingerprint check earns its place over a variant label.
    """
    task = make_task(key="bare-order-task")  # no expected_outcome, no tools_spec
    make_trajectory(task=task)
    client, _ = _probe_client(_constant("pass"))

    with client:
        run = run_probe(session, judge, "order", repeats=2, client=client)
    report = build_bias_probe_report(session, judge, probe="order", probe_run_id=run.id)

    shas = {
        (row.variant): row.prompt_sha for row in session.scalars(select(BiasProbeResult))
    }
    assert shas["reversed_context"] == shas["identity"]
    assert shas["context_last"] != shas["identity"]

    assert _arms(report)["reversed_context"].outcome == "not_applied"
    assert _arms(report)["context_last"].outcome != "not_applied"
    assert _arms(report)["context_last"].applied_to == 1


# --- the three measured outcomes ----------------------------------------------


def test_a_judge_that_grades_padded_transcripts_higher_is_reported_biased(
    session: Session, judge: Judge, suite: Callable[..., list[Trajectory]]
) -> None:
    """Six scenarios, every one graded up by half a verdict under padding.

    Every trajectory moves by exactly +0.5, so every resample of the six tasks does
    too and the interval collapses onto the shift. Six movers clears the floor this n
    demands (four, read at the Sidak-adjusted level), so the arm is a finding rather
    than a number the bootstrap could not have distinguished from zero.
    """
    suite(tasks=6)
    client, _ = _probe_client(_by_length(plain="borderline", padded="pass"))

    with client:
        run = run_probe(session, judge, "length", repeats=2, client=client)
    report = build_bias_probe_report(session, judge, probe="length", probe_run_id=run.id)

    assert report.trajectories == 6
    assert report.tasks == 6
    assert report.min_movers == min_movers(6, adjusted_confidence(0.95, len(LENGTH_VARIANTS)))
    assert report.min_movers == 4
    assert report.control_disagreement == 0.0  # the judge repeated itself exactly

    for variant in ("padded_1x", "padded_2x", "padded_3x"):
        arm = _arms(report)[variant]
        assert arm.applied_to == 6
        assert arm.signed_shift == 0.5  # borderline 0.5 -> pass 1.0
        assert (arm.shift_ci_low, arm.shift_ci_high) == (0.5, 0.5)
        assert arm.disagreement == 1.0  # every control draw differs from every variant draw
        assert arm.movers_observed == 6
        assert arm.outcome == "biased"
        assert arm.power_limited is False


def test_a_judge_that_grades_padded_transcripts_lower_is_reported_inverse(
    session: Session, judge: Judge, suite: Callable[..., list[Trajectory]]
) -> None:
    """The same evidence with the sign flipped, and it must not be called ``biased``.

    The direction is the claim. A probe that reported movement in either direction
    under one label would be answering a question nobody asked, and the length probe
    in particular has a rubric-licensed downward direction and an unlicensed upward
    one.
    """
    suite(tasks=6)
    client, _ = _probe_client(_by_length(plain="pass", padded="borderline"))

    with client:
        run = run_probe(session, judge, "length", repeats=2, client=client)
    report = build_bias_probe_report(session, judge, probe="length", probe_run_id=run.id)

    arm = _arms(report)["padded_3x"]
    assert arm.signed_shift == -0.5
    assert (arm.shift_ci_low, arm.shift_ci_high) == (-0.5, -0.5)
    assert arm.movers_observed == 6
    assert arm.outcome == "inverse"
    assert arm.power_limited is False


def test_a_judge_that_never_moves_is_inconclusive_and_says_the_instrument_was_blind(
    session: Session, judge: Judge, suite: Callable[..., list[Trajectory]]
) -> None:
    """Nothing moved, so nothing is claimed — and the report says why it could not have.

    Zero movers against a floor of four is not "no bias detected": at this n an effect
    smaller than four trajectories is invisible however real it is. ``power_limited``
    is what separates a measurement from a shrug, and a silent ``inconclusive`` is
    what this milestone exists to refuse.
    """
    suite(tasks=6)
    client, _ = _probe_client(_constant("pass"))

    with client:
        run = run_probe(session, judge, "length", repeats=2, client=client)
    report = build_bias_probe_report(session, judge, probe="length", probe_run_id=run.id)

    for variant in ("padded_1x", "padded_2x", "padded_3x"):
        arm = _arms(report)[variant]
        assert arm.applied_to == 6  # the perturbation was delivered, and did nothing
        assert arm.signed_shift == 0.0
        assert (arm.shift_ci_low, arm.shift_ci_high) == (0.0, 0.0)
        assert arm.movers_observed == 0
        assert arm.outcome == "inconclusive"
        assert arm.power_limited is True


def test_the_headline_is_the_flat_mean_the_interval_resamples(
    session: Session,
    judge: Judge,
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """One task moved on both its trajectories, another held on its one: the shift is 1/3.

    Three trajectories in two tasks, shifts +0.5, +0.5 and 0.0. The printed headline
    is their flat mean, (0.5 + 0.5 + 0.0) / 3 = 0.333, which is the quantity the
    clustered bootstrap actually resamples. A macro-average over tasks would print
    (0.5 + 0.0) / 2 = 0.25 beside an interval built for a different estimator.

    Two movers against a floor of three is also the power rule doing its job: the
    effect is half a verdict on two thirds of the corpus and it is still reported
    ``inconclusive``, because at this n a percentile bootstrap cannot exclude zero
    however large the movement is.
    """
    moving = make_task(key="moving-task", prompt="Scenario MOVING: refund a damaged item.")
    steady = make_task(key="steady-task", prompt="Scenario STEADY: check an order status.")
    make_trajectory(task=moving)
    make_trajectory(task=moving)
    make_trajectory(task=steady)

    def selective(user: str, index: int) -> str:
        if "Scenario MOVING" not in user:
            return "pass"
        return "pass" if _is_padded(user) else "borderline"

    client, _ = _probe_client(selective)
    with client:
        run = run_probe(session, judge, "length", repeats=2, client=client)
    report = build_bias_probe_report(session, judge, probe="length", probe_run_id=run.id)

    arm = _arms(report)["padded_1x"]
    assert report.trajectories == 3
    assert report.tasks == 2
    assert arm.applied_to == 3
    assert arm.signed_shift == pytest.approx(1 / 3)
    assert arm.signed_shift != pytest.approx(0.25)  # the macro-average over tasks
    assert arm.movers_observed == 2
    assert report.min_movers == 3
    assert arm.outcome == "inconclusive"
    assert arm.power_limited is True


@pytest.mark.parametrize(
    ("direction", "forbidden"),
    [("up", "biased"), ("down", "inverse")],
)
def test_a_bootstrap_bound_of_one_in_ten_quadrillion_is_not_a_finding(
    direction: str, forbidden: str
) -> None:
    """A resample whose exact mean is zero must not be read as an interval clearing zero.

    Verdict scores are 0.0/0.5/1.0, so a shift over three repeats is a multiple of a
    sixth — and sixths have no exact binary representation. Summing a resample that
    cancels to exactly zero therefore lands on 1.4e-17 rather than on 0.0, and a
    strict ``low > 0.0`` reads that last bit as a judge graded up by the
    perturbation. The scripted draws below are ordinary verdict triples, the seed is
    ordinary, and seven of eight trajectories moved, so the mover guard passes: 4.2%
    of seeds on this configuration used to print SENSITIVE beside an interval that
    renders as ``[+0.000, +0.479]``.

    Parametrised in both directions because ``high < 0.0`` manufactures ``inverse``
    from the same residue with the sign flipped.
    """
    pairs = [
        (("borderline", "pass", "pass"), ("fail", "pass", "pass")),  # -1/6
        (("fail", "pass", "pass"), ("fail", "borderline", "pass")),  # -1/6
        (("fail", "fail", "fail"), ("fail", "fail", "fail")),  # 0
        (("fail", "fail", "borderline"), ("fail", "borderline", "pass")),  # +1/3
        *[((("fail",) * 3), ("fail", "borderline", "pass"))] * 4,  # +1/2 each
    ]
    if direction == "down":
        pairs = [(variant, control) for control, variant in pairs]

    task_keys: dict[str, str] = {}
    draws: dict[tuple[str, str], tuple[str | None, ...]] = {}
    shas: dict[tuple[str, str], str] = {}
    for position, (control, variant) in enumerate(pairs):
        trajectory_id = f"t{position}"
        # One trajectory per task, so the cluster bootstrap resamples all eight.
        task_keys[trajectory_id] = f"task-{position}"
        draws[(trajectory_id, IDENTITY)] = control
        draws[(trajectory_id, "padded_1x")] = variant
        shas[(trajectory_id, IDENTITY)] = "identity-sha"
        shas[(trajectory_id, "padded_1x")] = "variant-sha"  # delivered
    rows = _ProbeRows(task_keys=task_keys, draws=draws, shas=shas, errors=0)

    level = adjusted_confidence(0.95, len(LENGTH_VARIANTS))
    floor = min_movers(len(pairs), confidence=level)
    arm = _score_arm(
        rows, "padded_1x", "", floor=floor, confidence=level, iterations=1000, seed=17
    )

    assert arm.movers_observed == 7 >= floor  # the power check cannot catch this one
    assert arm.outcome != forbidden
    assert arm.outcome == "inconclusive"
    # Reported as the zero it is, so a reader of the JSON who applies the same test by
    # hand reaches the same answer the `outcome` field did.
    bound = arm.shift_ci_low if direction == "up" else arm.shift_ci_high
    assert bound == 0.0


def test_a_real_bound_survives_the_residue_threshold(
    session: Session, judge: Judge, suite: Callable[..., list[Trajectory]]
) -> None:
    """The smallest interval a probe can actually produce is orders above the threshold.

    Shifts live on a lattice of 1/(2*repeats) and a resample mean divides it by the
    pooled trajectory count, so no attainable effect is small enough for
    ``ZERO_RESIDUE`` to swallow. Guards the fix against being widened into one that
    discards findings.
    """
    suite(tasks=6)
    client, _ = _probe_client(_by_length(plain="borderline", padded="pass"))
    with client:
        run = run_probe(session, judge, "length", repeats=2, client=client)
    report = build_bias_probe_report(session, judge, probe="length", probe_run_id=run.id)

    arm = _arms(report)["padded_1x"]
    assert arm.outcome == "biased"
    assert arm.shift_ci_low == 0.5 > ZERO_RESIDUE * 1e6


def test_the_resample_count_follows_the_adjusted_tail() -> None:
    """A bound read at the Sidak tail needs the resamples that tail implies, not 1000.

    The correction narrows each tail from 2.5% to 0.637% over four arms. At a fixed
    1000 resamples the lower bound is then ``means[6]`` -- an order statistic estimated
    from six draws -- and which side of zero it lands on is decided by the seed rather
    than by the judge. The mover guard does not cover it: an arm is admitted at
    ``movers == min_movers``, which is by construction the first mover count whose atom
    at zero falls below the tail, so a publishable arm sits against that boundary by
    design. Over random shift vectors on six to ten tasks the published verdict changed
    across seeds 0-11 for one in nine of them; sizing the count from the tail cuts that
    to one in sixty.

    The density is the merge gate's, not a new invention -- which is what the last
    assertion pins: at the uncorrected level the formula returns exactly the 10,000 that
    ``compare_runs`` already uses for an interval it thresholds rather than reads.
    """
    level = adjusted_confidence(0.95, len(ORDER_VARIANTS))
    iterations = _tail_sized_iterations(1000, level)
    tail = (1.0 - level) / 2.0

    assert int(tail * 999) == 6  # what a fixed 1000 resolved the adjusted tail to
    assert iterations * tail >= TAIL_RESAMPLES
    assert int(tail * (iterations - 1)) >= TAIL_RESAMPLES - 1
    assert _tail_sized_iterations(1000, 0.95) == 10_000
    # A caller asking for more keeps it; a level admitting no interval is left alone.
    assert _tail_sized_iterations(100_000, level) == 100_000
    assert _tail_sized_iterations(1000, 1.0) == 1000


# --- the control arm and failed calls ------------------------------------------


def test_the_control_arm_reports_the_judges_own_test_retest_noise(
    session: Session, judge: Judge, suite: Callable[..., list[Trajectory]]
) -> None:
    """Three byte-identical calls per trajectory, two of which agree: a floor of 2/3.

    This is a headline in its own right — "does this judge give the same answer
    twice" — and the number every shift has to be read against. It is computed over
    every trajectory that has two usable identity draws, never over the subset some
    arm happened to apply to.
    """
    suite(tasks=3)
    seen: dict[str, int] = {}

    def flaky(user: str, index: int) -> str:
        # Per prompt, not per call: every arm gets pass, pass, fail in that order, so
        # the control's floor is a property of the judge rather than of the schedule.
        seen[user] = seen.get(user, 0) + 1
        return "fail" if seen[user] == 3 else "pass"

    client, _ = _probe_client(flaky)
    with client:
        run = run_probe(session, judge, "order", repeats=3, client=client)
    report = build_bias_probe_report(session, judge, probe="order", probe_run_id=run.id)

    # [pass, pass, fail]: of the C(3,2) = 3 distinct unordered pairs, two differ.
    assert report.control_disagreement == pytest.approx(2 / 3)
    assert report.control_ci_low == pytest.approx(2 / 3)
    assert report.control_ci_high == pytest.approx(2 / 3)
    assert report.repeats == 3


def test_the_control_covers_every_trajectory_not_just_the_ones_an_arm_reached(
    session: Session,
    judge: Judge,
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """The judge's noise floor is a property of the judge, not of one arm's coverage.

    Two trajectories: one with an agent turn, which the padded arms perturb and on
    which the judge repeats itself exactly; one with no agent turn, which they cannot
    perturb at all and on which the judge wavered (2/3 over its three identity draws).

    The floor is the mean over both, (0.0 + 2/3) / 2 = 0.333. Scoping it to the
    trajectories some arm happened to apply to would report 0.0 — a judge that never
    disagrees with itself — and every shift in the run would then be read against a
    floor measured on the easy half of the corpus.
    """
    paddable = make_task(key="paddable-task", prompt="Scenario PADDABLE: refund a damaged item.")
    make_trajectory(task=paddable)
    bare = make_task(key="bare-task", prompt="Scenario BARE: look an order up.")
    make_trajectory(
        task=bare,
        steps=[
            ("user_message", {"text": "Where is order A123?"}),
            ("tool_call", {"name": "lookup_order", "arguments": {"order_id": "A123"}}),
        ],
    )
    seen: dict[str, int] = {}

    def wavers_on_the_bare_one(user: str, index: int) -> str:
        if "Scenario BARE" not in user:
            return "pass"
        seen[user] = seen.get(user, 0) + 1
        return "fail" if seen[user] == 3 else "pass"

    client, _ = _probe_client(wavers_on_the_bare_one)
    with client:
        run = run_probe(session, judge, "length", repeats=3, client=client)
    report = build_bias_probe_report(session, judge, probe="length", probe_run_id=run.id)

    assert _arms(report)["padded_1x"].applied_to == 1  # the bare trajectory has no agent turn
    assert report.control_disagreement == pytest.approx(1 / 3)
    # Two clusters, so a resample draws 0.0, 1/3 or 2/3 and the interval spans them.
    assert report.control_ci_low == pytest.approx(0.0)
    assert report.control_ci_high == pytest.approx(2 / 3)


def test_a_failed_call_is_recorded_and_the_sweep_carries_on(
    session: Session, judge: Judge, suite: Callable[..., list[Trajectory]]
) -> None:
    """One bad call never aborts a run, and it is never silently dropped either.

    A probe that hid its failures would report every figure as though it rested on the
    full repeat count. The failed draw is stored with its error, counted on the run,
    and surfaced on the report.
    """
    suite(tasks=2)

    def one_bad_call(user: str, index: int) -> str | None:
        return None if index == 1 else "pass"

    client, prompts = _probe_client(one_bad_call)
    with client:
        run = run_probe(session, judge, "length", repeats=2, client=client)
    report = build_bias_probe_report(session, judge, probe="length", probe_run_id=run.id)

    assert run.status == "completed"
    assert run.error_count == 1
    assert len(prompts) == 2 * len(LENGTH_VARIANTS) * 2  # the failed call was still made
    errored = session.scalars(
        select(BiasProbeResult).where(BiasProbeResult.error.is_not(None))
    ).one()
    assert errored.verdict is None
    assert "400" in errored.error
    assert report.errors == 1


def test_an_arm_that_answered_nothing_is_reported_unmeasured_and_not_unmoved(
    session: Session, judge: Judge, suite: Callable[..., list[Trajectory]]
) -> None:
    """Every padded call fails, and the arm must not read as a judge that held steady.

    This is the length probe's expected failure rather than an exotic one: the padded
    arms carry the longest prompts, so a provider that rejects one rejects exactly one
    side of every pair. Delivery still succeeded — the bytes differed on all six
    trajectories — so the arm is not ``not_applied``, which would send a reader off to
    debug a renderer that worked. What it measured is a separate count, and every
    figure is denominated in that one: ``0 of 6 moved`` beside ``applied 6/6`` is a
    statement about a judge that never answered, made in the vocabulary of a judge that
    answered six times and did not budge.
    """
    suite(tasks=6)

    def the_padded_prompts_are_rejected(user: str, index: int) -> str | None:
        return None if _is_padded(user) else "pass"

    client, _ = _probe_client(the_padded_prompts_are_rejected)
    with client:
        run = run_probe(session, judge, "length", repeats=2, client=client)
    report = build_bias_probe_report(session, judge, probe="length", probe_run_id=run.id)

    assert report.errors == 6 * 3 * 2  # six trajectories, three padded arms, two draws
    assert report.control_disagreement == 0.0  # the control was never touched
    for variant in ("padded_1x", "padded_2x", "padded_3x"):
        arm = _arms(report)[variant]
        assert arm.applied_to == 6  # the perturbation reached the model everywhere
        assert arm.measured_on == 0  # and came back from nowhere
        assert arm.signed_shift is None
        assert arm.disagreement is None
        assert arm.movers_observed == 0
        # Not `not_applied`: that name belongs to a renderer that produced the control's
        # bytes, and this one did not. Not `inconclusive` either -- in this report that
        # word means a measurement that found nothing, and the outcome is what a reader
        # acts on. `no_data` is the one name that sends them to the error count.
        assert arm.outcome == "no_data"
        assert arm.power_limited is True


def test_an_unexpected_failure_marks_the_run_failed_rather_than_leaving_it_running(
    session: Session, judge: Judge, suite: Callable[..., list[Trajectory]]
) -> None:
    """A run left at ``running`` is indistinguishable from one still in flight.

    Anything that gets past the per-call handler is not a judge failure, so the run is
    closed as failed and the rows written so far are kept: a sweep killed at call
    three measured three calls, and that is worth more than a clean slate.
    """
    suite(tasks=2)
    client, _ = _probe_client(_constant("pass"))

    def explode(position: int, planned: int, trajectory: Trajectory, row: BiasProbeResult) -> None:
        if position == 3:
            raise RuntimeError("the progress callback blew up")

    with client, pytest.raises(RuntimeError, match="blew up"):
        run_probe(session, judge, "length", repeats=2, client=client, on_progress=explode)

    run = session.scalars(select(BiasProbeRun)).one()
    assert run.status == "failed"
    assert run.completed_at is not None
    assert session.scalar(select(func.count()).select_from(BiasProbeResult)) == 3


@contextmanager
def _caller_session(factory: sessionmaker[Session]) -> Iterator[Session]:
    """The lifecycle every surface wraps a session in (``cli._open_session``).

    Reproduced rather than imported because the migration step it starts with needs a
    configured database, and the part under test is only the boundary: commit on the
    way out, roll back on an exception, close either way. Note that ``KeyboardInterrupt``
    is not an ``Exception``, so an interrupt reaches no ``except`` clause here at all --
    the transaction dies in ``close()``, which is precisely why the runner cannot leave
    its rows merely flushed.
    """
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@pytest.mark.parametrize("abort", [RuntimeError("killed at call five"), KeyboardInterrupt()])
def test_an_interrupted_sweep_keeps_the_calls_it_already_paid_for(
    session: Session,
    session_factory: sessionmaker[Session],
    judge: Judge,
    suite: Callable[..., list[Trajectory]],
    abort: BaseException,
) -> None:
    """The partial run survives the caller's rollback, which is the only place it counts.

    ``test_an_unexpected_failure_marks_the_run_failed_rather_than_leaving_it_running``
    drives a bare session and so proves only that the rows reached the transaction. A
    probe is the most expensive command in the tool -- trajectories x arms x repeats
    paid model calls -- and every surface that opens a session discards that transaction
    on the way out: by rollback for an exception, and by ``close()`` for a Ctrl-C, which
    is not an ``Exception`` and is the interrupt a human actually sends. Both are driven
    here, and the rows are read back through a session that did not write them.
    """
    suite(tasks=2)
    client, _ = _probe_client(_constant("pass"))

    def explode(position: int, planned: int, trajectory: Trajectory, row: BiasProbeResult) -> None:
        if position == 5:
            raise abort

    with client, pytest.raises(type(abort)):
        with _caller_session(session_factory) as probing:
            probed = probing.scalar(select(Judge).where(Judge.name == judge.name))
            run_probe(probing, probed, "length", repeats=2, client=client, on_progress=explode)

    # A different session: what a later `agentverdict probe` report, or the API, can see.
    with _caller_session(session_factory) as reader:
        stored = reader.scalars(select(BiasProbeRun)).one()
        assert stored.status == "failed"
        assert stored.completed_at is not None
        # Five rows written, and the fifth is the one whose callback raised -- it was
        # committed before the callback ran, so the call it paid for is not lost either.
        assert reader.scalar(select(func.count()).select_from(BiasProbeResult)) == 5
        # The token bill is stored with the rows rather than only in the lost session,
        # so the run says what it cost.
        assert stored.input_tokens > 0


def test_a_completed_sweep_is_durable_without_the_caller_committing(
    session: Session,
    session_factory: sessionmaker[Session],
    judge: Judge,
    suite: Callable[..., list[Trajectory]],
) -> None:
    """A finished run is never left wearing ``running``, whatever the caller does next.

    The caller here abandons its session without committing -- standing in for anything
    that fails between ``run_probe`` returning and the surface's commit, such as the
    report build or a second Ctrl-C. The sweep is over by then and its status has to say so.
    """
    suite(tasks=2)
    client, _ = _probe_client(_constant("pass"))

    probing = session_factory()
    with client:
        probed = probing.scalar(select(Judge).where(Judge.name == judge.name))
        run_probe(probing, probed, "length", repeats=2, client=client)
    probing.close()  # no commit: the caller's transaction is discarded

    with _caller_session(session_factory) as reader:
        stored = reader.scalars(select(BiasProbeRun)).one()
        assert stored.status == "completed"
        assert stored.completed_at is not None
        assert reader.scalar(select(func.count()).select_from(BiasProbeResult)) == 2 * len(
            LENGTH_VARIANTS
        ) * 2


# --- reading the report back ---------------------------------------------------


def test_the_report_names_the_cluster_count_behind_its_intervals(
    session: Session, judge: Judge, suite: Callable[..., list[Trajectory]]
) -> None:
    """Two tasks, two trajectories each: four rows resting on two independent draws.

    Trajectories replayed from one task share the prompt, the expected outcome and the
    tools block — exactly what the order probe permutes — so the bootstrap resamples
    tasks. Printing four without printing two would read as twice the independence
    the run actually has.
    """
    suite(tasks=2, per_task=2)
    client, _ = _probe_client(_constant("pass"))

    with client:
        run = run_probe(session, judge, "order", repeats=2, client=client)
    report = build_bias_probe_report(session, judge, probe="order", probe_run_id=run.id)

    assert report.trajectories == 4
    assert report.tasks == 2
    assert report.judge_prompt_version == PROMPT_VERSION
    assert report.confidence == 0.95
    assert report.adjusted_confidence == pytest.approx(0.95 ** (1 / len(ORDER_VARIANTS)))
    assert [arm.variant for arm in report.arms] == [
        variant for variant in ORDER_VARIANTS if variant != "identity"
    ]


def test_the_report_describes_the_run_it_was_given_rather_than_the_newest(
    session: Session, judge: Judge, suite: Callable[..., list[Trajectory]]
) -> None:
    """On a shared database "the newest run for this judge" is somebody else's run.

    The CLI reports the run it just executed, by id, for the same reason the merge
    gate takes a run id from a file instead of asking for the latest one.
    """
    suite(tasks=2)
    moving, _ = _probe_client(_by_length(plain="borderline", padded="pass"))
    with moving:
        first = run_probe(session, judge, "length", repeats=2, client=moving)
    steady, _ = _probe_client(_constant("pass"))
    with steady:
        second = run_probe(session, judge, "length", repeats=2, client=steady)

    older = build_bias_probe_report(session, judge, probe="length", probe_run_id=first.id)
    newer = build_bias_probe_report(session, judge, probe="length", probe_run_id=second.id)
    latest = build_bias_probe_report(session, judge, probe="length")

    assert _arms(older)["padded_1x"].signed_shift == 0.5
    assert _arms(newer)["padded_1x"].signed_shift == 0.0
    assert _arms(latest)["padded_1x"].signed_shift == 0.0  # the newest completed run


def test_a_run_belonging_to_another_judge_or_probe_is_not_reported(
    session: Session,
    judge: Judge,
    make_judge: Callable[..., Judge],
    suite: Callable[..., list[Trajectory]],
) -> None:
    """Answering with it would mislabel every number in the report."""
    suite(tasks=2)
    client, _ = _probe_client(_constant("pass"))
    with client:
        run = run_probe(session, judge, "order", repeats=2, client=client)

    stranger = make_judge(name="other-judge", model="llama-test-model")
    wrong_judge = build_bias_probe_report(session, stranger, probe="order", probe_run_id=run.id)
    wrong_probe = build_bias_probe_report(session, judge, probe="length", probe_run_id=run.id)

    for report in (wrong_judge, wrong_probe):
        assert report.repeats == 0
        assert report.trajectories == 0
        assert report.arms == []


def test_an_unprobed_judge_gets_an_empty_report_rather_than_zeros(
    session: Session, judge: Judge
) -> None:
    """No probe has run, which is a state to describe rather than a fault.

    Every statistic is null instead of zero, so nothing here can be rendered as a
    judge that passed a test nobody administered.
    """
    report = build_bias_probe_report(session, judge, probe="order")

    assert report.judge_id == judge.id
    assert report.judge_name == judge.name
    assert report.repeats == 0
    assert report.trajectories == 0
    assert report.tasks == 0
    assert report.arms == []
    assert report.control_disagreement is None
    assert report.control_ci_low is None
    assert report.min_movers is None
    assert report.judge_prompt_version is None
    assert report.errors == 0


def test_the_limit_selects_the_same_trajectories_every_time(
    session: Session, judge: Judge, suite: Callable[..., list[Trajectory]]
) -> None:
    """Ordered by (created_at, id), so two capped runs are comparable with each other.

    A limit that picked a different slice each time would make a cheap probe and an
    expensive one describe different corpora under the same heading.
    """
    suite(tasks=4)
    first_client, _ = _probe_client(_constant("pass"))
    with first_client:
        first = run_probe(session, judge, "order", repeats=2, limit=2, client=first_client)
    second_client, _ = _probe_client(_constant("pass"))
    with second_client:
        second = run_probe(session, judge, "order", repeats=2, limit=2, client=second_client)

    def touched(run: BiasProbeRun) -> set[str]:
        return set(
            session.scalars(
                select(BiasProbeResult.trajectory_id).where(
                    BiasProbeResult.probe_run_id == run.id
                )
            )
        )

    assert len(touched(first)) == 2
    assert touched(first) == touched(second)
    assert first.trajectory_count == second.trajectory_count == 2


# --- direction against the human labels ----------------------------------------


def test_a_flip_toward_the_humans_is_counted_apart_from_one_away(
    session: Session,
    judge: Judge,
    suite: Callable[..., list[Trajectory]],
    make_label: Callable[..., HumanLabel],
) -> None:
    """DESIGN part 2 clause 9: a shift says how far, never which way against the humans.

    The padded arms move every trajectory from ``pass`` to ``fail`` here, so the shift
    is identical everywhere. But three of the six trajectories were labeled ``fail`` by
    the annotators and three ``pass``, so the same movement is the judge being corrected
    on one half and knocked off a verdict the humans agreed with on the other. A report
    that could not tell those apart would file both under "length sensitivity", and the
    half that got *better* under padding is the more interesting finding of the two.
    """
    trajectories = suite(tasks=6)
    for position, trajectory in enumerate(trajectories):
        # Three humans-say-fail (the padded answer is right), three humans-say-pass
        # (the padded answer is wrong). Two annotators agreeing, so each is a majority.
        verdict = "fail" if position < 3 else "pass"
        make_label(trajectory, "vamp", verdict)
        make_label(trajectory, "momo", verdict)

    client, _ = _probe_client(_by_length(plain="pass", padded="fail"))
    with client:
        run = run_probe(session, judge, "length", repeats=2, client=client)
    report = build_bias_probe_report(session, judge, probe="length", probe_run_id=run.id)

    for variant in ("padded_1x", "padded_2x", "padded_3x"):
        arm = _arms(report)[variant]
        assert arm.movers_observed == 6  # every trajectory moved, by the same amount
        assert arm.movers_labeled == 6  # and every one had a majority to judge it by
        assert arm.moved_toward_humans == 3
        assert arm.moved_away_from_humans == 3


def test_an_unlabeled_corpus_still_probes_and_claims_no_direction(
    session: Session, judge: Judge, suite: Callable[..., list[Trajectory]]
) -> None:
    """A probe compares the judge against itself, so labels are a bonus, not a input.

    With nothing labeled the direction counts stay at zero rather than defaulting to
    "moved away", which would invent a finding out of an absence of ground truth.
    """
    suite(tasks=6)
    client, _ = _probe_client(_by_length(plain="pass", padded="fail"))
    with client:
        run = run_probe(session, judge, "length", repeats=2, client=client)
    report = build_bias_probe_report(session, judge, probe="length", probe_run_id=run.id)

    arm = _arms(report)["padded_1x"]
    assert arm.movers_observed == 6
    assert (arm.movers_labeled, arm.moved_toward_humans, arm.moved_away_from_humans) == (0, 0, 0)


def test_a_tie_between_annotators_is_not_a_direction_to_move_toward(
    session: Session,
    judge: Judge,
    suite: Callable[..., list[Trajectory]],
    make_label: Callable[..., HumanLabel],
) -> None:
    """With two annotators every disagreement is a tie, and a tie names no target.

    Scoring against a tie would mean picking one annotator arbitrarily and reporting
    the judge's distance from a coin flip.
    """
    trajectories = suite(tasks=6)
    for trajectory in trajectories:
        make_label(trajectory, "vamp", "pass")
        make_label(trajectory, "momo", "fail")

    client, _ = _probe_client(_by_length(plain="pass", padded="fail"))
    with client:
        run = run_probe(session, judge, "length", repeats=2, client=client)
    report = build_bias_probe_report(session, judge, probe="length", probe_run_id=run.id)

    arm = _arms(report)["padded_1x"]
    assert arm.movers_observed == 6
    assert arm.movers_labeled == 0  # six ties, no majorities
