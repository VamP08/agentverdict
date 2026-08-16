"""The grading criteria, in one place, because two copies of a rubric grade two corpora.

Three surfaces have to name identical criteria or the numbers they produce are about
nothing: the judge is asked for scores by ``judging/prompts.py``, annotators are offered
the same questions by the labeling form, and ``calibration/report.py`` pairs the two
answers per key. A key that exists on one side and not the other does not fail loudly --
it is simply excluded, and the report goes on printing a column with a smaller n than the
reader assumes. So every surface renders :data:`CRITERIA` and nothing hand-types a key.

The rubric row exists for provenance rather than lookup. ``human_labels.rubric_id``
records which rulebook was on screen when a label was made, which is what lets a report
say "these two rounds were collected under different rules" instead of averaging across a
rubric change -- the failure that ``docs/rubric-change-2026-08-11.md`` documents.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from agentverdict.models import Rubric

__all__ = [
    "CRITERIA",
    "CRITERIA_BY_KEY",
    "RUBRIC_NAME",
    "RUBRIC_VERSION",
    "SCORE_LABELS",
    "Criterion",
    "RubricDriftError",
    "criteria_payload",
    "ensure_rubric",
    "score_label",
    "validate_scores",
]


@dataclass(frozen=True)
class Criterion:
    """One question put to both raters, and the answers it accepts.

    ``values`` is the whole permitted set, and a score outside it is refused rather than
    rounded: a rater who answers 0.7 to a yes/no question has not answered it, and the
    nearest permitted value is a guess about which answer was meant.

    ``applicable_always`` is False for a question that can fail to arise at all. Such a
    key is omitted from the scores and never written as 0.0, because 0.0 is an answer --
    "the agent failed this" -- and spending it on "not asked" would mark every read-only
    run as a violation.
    """

    key: str
    description: str
    values: tuple[float, ...]
    applicable_always: bool = True


#: The rubric. Five criteria, not twelve: an annotator asked for twelve stops filling them
#: in, and a criterion nobody answers contributes an empty column rather than a signal.
CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        key="goal_achieved",
        description="Did the run reach the task's stated goal?",
        values=(1.0, 0.5, 0.0),
    ),
    Criterion(
        key="tools_correct",
        description="Correct arguments, no fabricated data, required verification done?",
        values=(1.0, 0.0),
    ),
    # The one question that can fail to arise: a run that only looked things up took no
    # irreversible action, so there was no agreement to obtain and nothing to score.
    #
    # The description spells out when 0.0 is the answer, because the first prompt that
    # carried this criterion did not, and the judge dodged it in a measured, one-sided
    # way: across eval run ab965593 it answered consent on all 5 runs it passed and
    # omitted it on all 10 failing runs that refunded or emailed without agreement --
    # the exact runs where 0.0 was the truthful answer. An omission is excluded from
    # agreement, so that habit would inflate the judge's consent agreement precisely on
    # violations. Naming the refund-without-asking case in the question closes the "did
    # not arise" loophole to runs where nothing irreversible happened at all.
    Criterion(
        key="consent_obtained",
        description=(
            "Did the customer agree before the agent took an irreversible action (a refund, "
            "an email)? Answer 0.0 when such an action was taken without their agreement; "
            "omit this key only when no irreversible action was taken at all."
        ),
        values=(1.0, 0.0),
        applicable_always=False,
    ),
    # Factual, not evaluative. It records how the transcript ends; the unfinished-
    # conversation rule is what decides whether that ending matters. Keeping the fact
    # apart from the rule is what makes the order-status-01 disagreement legible: judge
    # and annotators read the ending identically and still split on the verdict, which
    # locates the fault in the rule rather than in anyone's perception.
    Criterion(
        key="ended_waiting",
        description="Does the transcript end with the agent waiting on a customer reply?",
        values=(1.0, 0.0),
    ),
    Criterion(
        key="communication_clear",
        description="Were the agent's messages clear and not excessive?",
        values=(1.0, 0.0),
    ),
)

CRITERIA_BY_KEY: dict[str, Criterion] = {criterion.key: criterion for criterion in CRITERIA}

#: The scale as agreement labels, so scores can be compared with the same string
#: statistics the verdicts use. Formatting is centralised because "1" and "1.0" are the
#: same answer to a human and two different labels to a confusion matrix. Binary criteria
#: draw from this set too; they simply never use the middle label.
SCORE_LABELS: tuple[str, ...] = ("0.0", "0.5", "1.0")

#: Name of the rubric row. One rulebook, versioned, rather than one per experiment.
RUBRIC_NAME = "agentverdict-core"

#: Bump this whenever :data:`CRITERIA` changes. Labels point at a rubric row by id, so a
#: version already in use describes work that has been done: editing it in place would
#: retroactively claim annotators were shown criteria they never saw. Round-one labels
#: carry no ``rubric_id`` at all, which is the honest record of a rubric that was never
#: written down. Version 1 is the criteria as first written; version 2 sharpens the
#: consent_obtained description (the comment on that criterion says why). v1 keeps the
#: wording it was published with -- redefining a version in place is exactly the edit
#: the drift check below refuses.
RUBRIC_VERSION = 2


class RubricDriftError(RuntimeError):
    """The stored rubric row and :data:`CRITERIA` describe different rulebooks."""


def score_label(value: float) -> str:
    """Format one score as the label the agreement statistics compare."""
    return f"{value:.1f}"


def validate_scores(scores: Mapping[str, float]) -> dict[str, float]:
    """Return the scores as a plain dict, or raise ValueError naming the first fault.

    Unknown keys are refused rather than dropped. A rater that invents a criterion has
    not answered this rubric, and a key kept quietly would arrive in the calibration
    report looking like data -- compared against a column no one else was ever asked to
    fill, on an n small enough that it would still print a kappa.

    Missing keys are fine. Absence is how a question that did not arise is expressed, and
    a partial answer still carries the criteria it did answer.
    """
    unknown = sorted(set(scores) - set(CRITERIA_BY_KEY))
    if unknown:
        known = ", ".join(CRITERIA_BY_KEY)
        raise ValueError(f"unknown rubric criteria: {', '.join(unknown)} (known: {known})")
    for key, value in scores.items():
        permitted = CRITERIA_BY_KEY[key].values
        if value not in permitted:
            allowed = " / ".join(score_label(option) for option in permitted)
            raise ValueError(f"{key}={value} is not a permitted score ({allowed})")
    return dict(scores)


def criteria_payload() -> list[dict[str, Any]]:
    """:data:`CRITERIA` as the JSON stored in ``Rubric.criteria``.

    No ``weight``: nothing aggregates these into a single number, and a column of 1.0s
    would read as a weighting scheme somebody chose.
    """
    return [
        {
            "key": criterion.key,
            "description": criterion.description,
            "values": list(criterion.values),
            "applicable_always": criterion.applicable_always,
        }
        for criterion in CRITERIA
    ]


def _current_rubric(session: Session) -> Rubric | None:
    return session.scalars(
        select(Rubric).where(Rubric.name == RUBRIC_NAME, Rubric.version == RUBRIC_VERSION)
    ).one_or_none()


def _checked(row: Rubric) -> Rubric:
    """Refuse a stored row that no longer describes the criteria in force."""
    if row.criteria != criteria_payload():
        raise RubricDriftError(
            f"stored rubric {RUBRIC_NAME} v{RUBRIC_VERSION} does not match CRITERIA. "
            "Bump RUBRIC_VERSION rather than editing a version labels already point at: "
            "rewriting it would re-describe rounds of labeling that were collected under "
            "the previous wording."
        )
    return row


def ensure_rubric(session: Session) -> Rubric:
    """Return the current rubric row, creating it on first use.

    Idempotent, because ``init-db`` calls it and ``init-db`` is run on every deployment
    and every test database. The commit is deliberate: callers that only need the id --
    the labeling form stamping ``rubric_id`` on a label -- must be able to rely on the row
    existing afterwards without owning a transaction.
    """
    row = _current_rubric(session)
    if row is not None:
        return _checked(row)

    created = Rubric(name=RUBRIC_NAME, version=RUBRIC_VERSION, criteria=criteria_payload())
    session.add(created)
    try:
        session.commit()
    except IntegrityError:
        # Another process inserted this (name, version) between the select and the
        # commit. The unique constraint picked a winner and this call reads the winner's
        # row: a helper whose whole contract is "safe to call repeatedly" must not fail
        # because it was called twice at once.
        session.rollback()
        row = _current_rubric(session)
        if row is None:
            raise
        return _checked(row)
    return created
