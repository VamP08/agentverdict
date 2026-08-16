"""The rubric as one source: the judge's prompt, the stored row, and the scores contract.

Three surfaces have to name identical criteria, and not one of them fails loudly when they
drift. A prompt asking for a key no rubric lists still gets answered; the answer still
reaches ``judge_verdicts.rubric_scores``; calibration still opens a column for it and
still prints a kappa, over a sample quietly reduced to the rows that happened to carry the
key. Nothing raises and the number means nothing. The tests here are the tripwire that
turns that into a red build.

Offline: the one runner test drives a mock transport, and everything else is text.
"""

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from agentverdict.cli import app as cli_app
from agentverdict.config import get_settings
from agentverdict.db import reset_engine
from agentverdict.judging.client import GroqChatClient
from agentverdict.judging.prompts import CORRECTION_PROMPT, SYSTEM_PROMPT
from agentverdict.judging.runner import run_eval
from agentverdict.models import Judge, JudgeVerdict, Rubric, Trajectory
from agentverdict.rubric import (
    CRITERIA,
    RUBRIC_NAME,
    RUBRIC_VERSION,
    SCORE_LABELS,
    RubricDriftError,
    ensure_rubric,
    score_label,
    validate_scores,
)
from agentverdict.schemas import JudgeDecision

#: One line of the prompt's criteria block: ``- goal_achieved (1.0 | 0.5 | 0.0): ...``.
_BULLET = re.compile(r"^- (\w+) \(([^)]+)\):", re.MULTILINE)

#: The ``rubric_scores`` object inside the answer contract. Parsed separately from the
#: block above because this is the line the judge copies its shape from: the two can
#: disagree, and when they do it is the contract the answers follow.
_SCORES_SHAPE = re.compile(r'"rubric_scores": \{(.*)\}\}')

#: The keys of the criteria, in rubric order. Written out rather than derived so that a
#: reader can check the other five surfaces against something.
EXPECTED_KEYS = [
    "goal_achieved",
    "tools_correct",
    "consent_obtained",
    "ended_waiting",
    "communication_clear",
]


def _asked_for(prompt: str) -> list[str]:
    """The criterion keys the answer contract in ``prompt`` tells the judge to fill."""
    shape = _SCORES_SHAPE.search(prompt)
    assert shape is not None, "the answer contract no longer contains a rubric_scores object"
    return re.findall(r'"(\w+)":', shape.group(1))


def _decision(**scores: float) -> dict[str, object]:
    """A well-formed judge answer carrying ``scores``, for exercising the contract."""
    return {"verdict": "pass", "rationale": "Looked the order up first.", "rubric_scores": scores}


# --- the three surfaces agree ------------------------------------------------


def test_the_prompt_the_rubric_row_and_criteria_name_one_set_of_keys(session: Session) -> None:
    """The anti-drift tripwire: without it the judge can be asked for unlisted keys.

    Every surface renders ``CRITERIA`` today, so this passes by construction — which is
    exactly the point. It fails the moment someone hand-types a key into the prompt or
    edits a stored rubric row, and that failure has no other symptom: the judge answers
    the invented key, the report opens a column for it, and the column is compared
    against nothing any annotator was ever shown.
    """
    assert [criterion.key for criterion in CRITERIA] == EXPECTED_KEYS

    assert [key for key, _ in _BULLET.findall(SYSTEM_PROMPT)] == EXPECTED_KEYS
    assert _asked_for(SYSTEM_PROMPT) == EXPECTED_KEYS
    # The correction turn restates the shape, so a judge that answered badly is not
    # re-asked under a different rubric than the one it just failed.
    assert _asked_for(CORRECTION_PROMPT) == EXPECTED_KEYS

    row = ensure_rubric(session)
    assert (row.name, row.version) == (RUBRIC_NAME, RUBRIC_VERSION)
    assert [entry["key"] for entry in row.criteria] == EXPECTED_KEYS
    # The wording travels too: the row is what a report reads to say which questions an
    # annotator was shown, and a key with someone else's question behind it says nothing.
    assert [entry["description"] for entry in row.criteria] == [
        criterion.description for criterion in CRITERIA
    ]


def test_the_prompt_offers_exactly_the_values_each_criterion_accepts() -> None:
    """A prompt advertising a score the rubric refuses costs a correction turn per call.

    ``JudgeDecision`` rejects anything off the scale, so a block offering 0.5 on a yes/no
    criterion would have the judge answer as instructed, be refused, and be asked again —
    silently doubling the cost of every eval run that touched it.
    """
    offered = dict(_BULLET.findall(SYSTEM_PROMPT))
    for criterion in CRITERIA:
        assert offered[criterion.key] == " | ".join(
            score_label(value) for value in criterion.values
        )
    # Spelled out for the two shapes the rubric actually uses.
    assert offered["goal_achieved"] == "1.0 | 0.5 | 0.0"
    assert offered["tools_correct"] == "1.0 | 0.0"


def test_only_consent_obtained_may_be_omitted_and_the_prompt_says_so() -> None:
    """Absence means the question did not arise, so the judge has to be told which one.

    Left out of the prompt, a judge asked for five keys answers five: it would write
    ``consent_obtained: 0.0`` for a run that only looked something up, which is the
    statement *the agent moved someone's money without asking*. Every read-only
    trajectory in the corpus would book a consent violation that never happened.
    """
    assert [c.key for c in CRITERIA if not c.applicable_always] == ["consent_obtained"]

    assert "Omit consent_obtained when the question did not arise" in SYSTEM_PROMPT
    # Marked in the answer contract as well as in the block, because a key listed
    # unconditionally on the line the judge copies gets answered unconditionally.
    marked = re.findall(r'"(\w+)": [^,}]*\(include only', SYSTEM_PROMPT)
    assert marked == ["consent_obtained"]

    # And the reason is stated, not just the instruction: a judge told only "omit it"
    # has no way to know that 0.0 is the wrong way to say the same thing.
    assert "A 0.0 says the agent failed that criterion" in SYSTEM_PROMPT


def test_score_label_gives_one_spelling_per_answer() -> None:
    """The agreement statistics compare strings, so "1" and "1.0" would be two answers.

    They are the same answer to a person and two categories to a confusion matrix, which
    would split a criterion's sample in half and leave both halves degenerate.
    """
    assert score_label(1.0) == "1.0"
    assert score_label(1) == "1.0"
    assert score_label(0.5) == "0.5"
    assert score_label(0.0) == "0.0"
    assert score_label(0) == "0.0"

    # Every value any criterion accepts has to be in the comparison vocabulary: a score
    # outside it is dropped by the statistics rather than counted, so a criterion that
    # gained a 0.25 would report an n smaller than the number of answers behind it.
    every_value = {score_label(value) for c in CRITERIA for value in c.values}
    assert every_value <= set(SCORE_LABELS)


# --- the stored rubric row ---------------------------------------------------


def test_ensure_rubric_is_idempotent(
    session: Session, session_factory: sessionmaker[Session]
) -> None:
    """Every deployment and every labeling POST calls it; twice has to mean once.

    A second row for the same rules would split labels between two ids, and a report that
    reads ``rubric_id`` to say which rulebook was on screen would then describe one
    annotation round as two rounds that cannot be compared.
    """
    first = ensure_rubric(session)
    second = ensure_rubric(session)
    assert second.id == first.id

    # A separate session is the real case: the next request, or the next process.
    with session_factory() as other:
        third = ensure_rubric(other)
    assert third.id == first.id

    assert session.scalar(select(func.count()).select_from(Rubric)) == 1


def test_ensure_rubric_refuses_a_stored_row_that_no_longer_matches(session: Session) -> None:
    """Editing a version that labels already point at re-describes finished work.

    The row is provenance. Rewriting it in place would claim annotators answered criteria
    that were never on screen, so the helper refuses and names the fix instead of
    silently adopting the new wording.
    """
    row = ensure_rubric(session)
    row.criteria = [dict(entry, description="Reworded after the fact.") for entry in row.criteria]
    session.commit()

    with pytest.raises(RubricDriftError) as raised:
        ensure_rubric(session)
    assert "Bump RUBRIC_VERSION" in str(raised.value)


# --- init-db seeds the row ----------------------------------------------------


@pytest.fixture
def cli_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[sessionmaker[Session]]:
    """Point the application's own settings at a scratch SQLite file, then restore.

    The CLI resolves its database through ``get_settings``/``get_engine``, the same
    way ``tests/test_gating_surfaces.py`` drives ``compare``. Nothing is migrated
    here: creating the schema is ``init-db``'s job, and doing it for the command
    would hide a command that stopped doing it.
    """
    url = f"sqlite:///{(tmp_path / 'init.db').as_posix()}"
    monkeypatch.setenv("AGENTVERDICT_DATABASE_URL", url)
    get_settings.cache_clear()
    reset_engine()
    engine = create_engine(url)
    try:
        yield sessionmaker(bind=engine)
    finally:
        engine.dispose()
        reset_engine()
        get_settings.cache_clear()


def test_init_db_seeds_the_rubric_row_and_twice_means_once(
    cli_db: sessionmaker[Session],
) -> None:
    """The row is provenance, so it has to exist before anyone can be shown the rules.

    Left to the first label POST, a deployment nobody has annotated yet has no rulebook
    to name, and an API client cannot stamp a ``rubric_id`` it has no way to look up.
    And ``init-db`` runs on every deployment, so the second run has to find the first
    run's row rather than duplicate it or fail.
    """
    runner = CliRunner()

    first = runner.invoke(cli_app, ["init-db"])
    assert first.exit_code == 0, first.output
    assert "Database schema is up to date." in first.output
    assert f"Rubric: {RUBRIC_NAME} v{RUBRIC_VERSION}, {len(CRITERIA)} criteria." in first.output

    with cli_db() as db:
        row = db.scalars(select(Rubric)).one()
        assert (row.name, row.version) == (RUBRIC_NAME, RUBRIC_VERSION)
        assert [entry["key"] for entry in row.criteria] == EXPECTED_KEYS

    second = runner.invoke(cli_app, ["init-db"])
    assert second.exit_code == 0, second.output
    with cli_db() as db:
        assert db.scalar(select(func.count()).select_from(Rubric)) == 1


def test_init_db_reports_a_drifted_rubric_row_instead_of_a_traceback(
    cli_db: sessionmaker[Session],
) -> None:
    """Drift fails the deploy step with the fix named, not the next labeling session.

    Uncaught, ``RubricDriftError`` surfaces as a blank 500 in front of whichever
    annotator presses save next, on every retry. ``init-db`` is where a criteria edit
    that forgot its version bump should stop instead, and a stack trace is not a
    message an operator can act on.
    """
    runner = CliRunner()
    assert runner.invoke(cli_app, ["init-db"]).exit_code == 0
    with cli_db() as db:
        row = db.scalars(select(Rubric)).one()
        row.criteria = [dict(entry, description="Older wording.") for entry in row.criteria]
        db.commit()

    result = runner.invoke(cli_app, ["init-db"])

    assert result.exit_code == 1
    assert "Rubric check failed" in result.output
    assert "Bump RUBRIC_VERSION" in result.output
    # The handled path: the message above is the whole output, the exception never
    # escapes the command, and no success line prints beside the failure.
    assert "Traceback" not in result.output
    assert not isinstance(result.exception, RubricDriftError)
    assert "Database schema is up to date." not in result.output


# --- what counts as an answer ------------------------------------------------


def test_validate_scores_keeps_a_partial_answer_and_refuses_an_invented_one() -> None:
    """Missing keys are an answer about the run; unknown keys are not an answer at all.

    A rater that invents a criterion has not answered this rubric, and a key kept quietly
    would arrive in calibration looking like data — compared against a column nobody else
    was asked to fill, on an n small enough that it would still print a kappa.
    """
    assert validate_scores({}) == {}
    assert validate_scores({"goal_achieved": 0.5}) == {"goal_achieved": 0.5}

    with pytest.raises(ValueError, match="unknown rubric criteria: vibes"):
        validate_scores({"vibes": 1.0})
    # 0.5 is a permitted score — for goal_achieved. tools_correct is a yes/no question,
    # and "the arguments were partly correct" is not one of its answers.
    with pytest.raises(ValueError, match="tools_correct=0.5"):
        validate_scores({"tools_correct": 0.5})
    with pytest.raises(ValueError, match="goal_achieved=0.7"):
        validate_scores({"goal_achieved": 0.7})


def test_judge_decision_rejects_a_criterion_the_rubric_does_not_define() -> None:
    """A rejected answer costs one correction turn and says what was wrong.

    Kept, the invented key would reach the report as a column with no human counterpart.
    """
    with pytest.raises(ValidationError, match="unknown rubric criteria: helpfulness"):
        JudgeDecision.model_validate(_decision(helpfulness=1.0))


def test_judge_decision_rejects_a_value_the_criterion_does_not_offer() -> None:
    """Off-scale scores are refused rather than rounded to the nearest permitted one.

    A judge that answered 0.5 to a yes/no question has not answered it, and snapping the
    value would record an opinion nobody held. Left alone, 0.5 becomes a third category
    in that criterion's confusion matrix and both real answers lose rows to it.
    """
    with pytest.raises(ValidationError, match="ended_waiting=0.5"):
        JudgeDecision.model_validate(_decision(ended_waiting=0.5))
    with pytest.raises(ValidationError, match="goal_achieved=2.0"):
        JudgeDecision.model_validate(_decision(goal_achieved=2.0))


def test_judge_decision_accepts_an_answer_that_omits_the_optional_criterion() -> None:
    """A read-only run has no consent to score, and an empty object is a round-one judge."""
    decision = JudgeDecision.model_validate(_decision(goal_achieved=1.0, ended_waiting=1.0))
    assert decision.rubric_scores == {"goal_achieved": 1.0, "ended_waiting": 1.0}
    assert "consent_obtained" not in decision.rubric_scores

    assert JudgeDecision.model_validate(_decision()).rubric_scores == {}


def test_the_judges_scores_reach_the_verdict_row(
    session: Session, make_trajectory: Callable[..., Trajectory]
) -> None:
    """Calibration reads ``judge_verdicts.rubric_scores`` and nothing else.

    A runner that parsed the scores and then dropped them would leave every per-criterion
    row at n=0 while the judge answered faithfully on every call — which reads as a rubric
    nobody fills in rather than as a lost write, and points at the annotators.
    """
    judge = Judge(name="rubric-judge", model="test-model")
    session.add(judge)
    session.commit()
    make_trajectory()

    answer = json.dumps(
        {
            "verdict": "pass",
            "rationale": "Looked the order up, reported the ETA, then offered tracking.",
            "rubric_scores": {"goal_achieved": 1.0, "ended_waiting": 1.0},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": answer}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    with GroqChatClient(
        api_key="test-key",
        base_url="https://groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
    ) as client:
        run_eval(session, judge, client=client)

    row = session.scalars(select(JudgeVerdict)).one()
    assert row.rubric_scores == {"goal_achieved": 1.0, "ended_waiting": 1.0}
    assert "consent_obtained" not in row.rubric_scores
