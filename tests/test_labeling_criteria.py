"""The labeling form's criteria: what an annotator is offered and what gets stored.

The form is the human half of the rubric, and the two halves are compared per key, so a
question posted under a different name than the judge answers is a column with one side
filled in. It is also the surface a non-engineer uses, which is why every criterion is
optional and why "did not arise" is a button rather than a convention: an annotator who
must answer five questions to save a verdict will answer them badly.

The distinction these tests exist to protect is the one at the centre of the rubric — "did
not arise" stores nothing, "no" stores 0.0 — and round-one compatibility, since a form
that started requiring criteria would make the first annotation round unrepeatable.
"""

import re
from collections.abc import Callable

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from agentverdict.models import HumanLabel, Rubric, Trajectory
from agentverdict.rubric import CRITERIA, CRITERIA_BY_KEY

#: A fully answered form, as the browser posts it. The field names are written out rather
#: than built from ``CRITERIA``: they are the contract between template and route, and a
#: renamed prefix has to break something.
FULL_FORM = {
    "annotator": "momo",
    "verdict": "pass",
    "criterion_goal_achieved": "1.0",
    "criterion_tools_correct": "1.0",
    "criterion_consent_obtained": "",  # the "did not arise" option
    "criterion_ended_waiting": "1.0",
    "criterion_communication_clear": "0.0",
}


def _criteria_fieldset(html: str) -> str:
    """Just the criteria block of the rendered form, so the verdict radios cannot match."""
    start = html.index('id="criteria"')
    return html[start : html.index("</fieldset>", start)]


# --- what the form offers ----------------------------------------------------


def test_the_form_offers_every_criterion_below_the_verdict(
    client: TestClient, make_trajectory: Callable[..., Trajectory]
) -> None:
    """All five, in rubric order, after the one answer that is actually required.

    Order is part of the contract, not a preference. The verdict is what the report is
    built on and the only required field; criteria placed above it would put five
    optional questions between an annotator and the thing they came to do.
    """
    trajectory = make_trajectory()
    html = client.get(f"/label/{trajectory.id}").text

    positions = [html.index(f'name="criterion_{c.key}"') for c in CRITERIA]
    assert positions == sorted(positions)  # rendered in rubric order
    assert html.index('name="verdict"') < positions[0]

    # One question in full, to show the wording on screen is the rubric's own and not a
    # second copy typed into the template.
    assert CRITERIA_BY_KEY["ended_waiting"].description in html


def test_a_fresh_form_pre_selects_nothing_and_only_consent_can_be_skipped(
    client: TestClient, make_trajectory: Callable[..., Trajectory]
) -> None:
    """A pre-selected radio is an answer nobody gave, and radios cannot be un-clicked.

    Every criterion is optional, so the honest default is silence. The one empty-valued
    option is ``consent_obtained``'s "did not arise": it posts, the parser drops it, and
    no key reaches ``rubric_scores`` — which is a different statement about the run than
    the 0.0 the "no" button next to it stores.
    """
    trajectory = make_trajectory()
    fieldset = _criteria_fieldset(client.get(f"/label/{trajectory.id}").text)

    assert "checked" not in fieldset
    assert fieldset.count('value=""') == 1
    assert 'name="criterion_consent_obtained" value=""' in fieldset
    assert "did not arise" in fieldset


# --- what the form stores ----------------------------------------------------


def test_the_form_round_trips_criteria_into_rubric_scores(
    client: TestClient,
    make_trajectory: Callable[..., Trajectory],
    session_factory: sessionmaker[Session],
) -> None:
    """Every answered criterion reaches the label; the skipped one leaves no key at all.

    Absence is the record that the question did not arise. Written as 0.0 it would say
    the agent took an irreversible action without agreement, and this run only looked
    something up.
    """
    trajectory = make_trajectory()
    response = client.post(f"/label/{trajectory.id}", data=FULL_FORM, follow_redirects=False)
    assert response.status_code in (302, 303)

    with session_factory() as check:
        label = check.scalars(select(HumanLabel)).one()
        rubric = check.scalars(select(Rubric)).one()

    assert label.rubric_scores == {
        "goal_achieved": 1.0,
        "tools_correct": 1.0,
        "ended_waiting": 1.0,
        "communication_clear": 0.0,
    }
    assert "consent_obtained" not in label.rubric_scores
    # Which rulebook was on screen, so a later report can say two rounds were collected
    # under different rules instead of averaging across a rubric change.
    assert label.rubric_id == rubric.id


def test_no_and_did_not_arise_are_stored_differently(
    client: TestClient,
    make_trajectory: Callable[..., Trajectory],
    session_factory: sessionmaker[Session],
) -> None:
    """The single distinction the whole rubric turns on, made by two adjacent buttons.

    "No" says the agent moved someone's money without asking. "Did not arise" says
    nothing irreversible happened. Collapsing them would mark every read-only run in the
    corpus as a consent violation, and would report the judge as agreeing about it.
    """
    trajectory = make_trajectory()
    for annotator, consent in [("momo", ""), ("vamp", "0.0")]:
        client.post(
            f"/label/{trajectory.id}",
            data={
                "annotator": annotator,
                "verdict": "pass",
                "criterion_consent_obtained": consent,
            },
            follow_redirects=False,
        )

    with session_factory() as check:
        stored = {
            label.annotator: label.rubric_scores
            for label in check.scalars(select(HumanLabel)).all()
        }

    assert stored["momo"] == {}
    assert stored["vamp"] == {"consent_obtained": 0.0}


def test_a_label_with_only_a_verdict_still_saves(
    client: TestClient,
    make_trajectory: Callable[..., Trajectory],
    session_factory: sessionmaker[Session],
) -> None:
    """Round one was collected without criteria and must not become retroactively invalid.

    Nothing here is required but the verdict, so an annotator who is sure of the call and
    unsure of the breakdown submits the call. The label is stored with an empty score
    object, not with a row of zeros — a zero is an answer, and this annotator gave none.
    """
    trajectory = make_trajectory()
    response = client.post(
        f"/label/{trajectory.id}",
        data={"annotator": "vamp", "verdict": "borderline"},
        follow_redirects=False,
    )
    assert response.status_code in (302, 303)

    with session_factory() as check:
        label = check.scalars(select(HumanLabel)).one()

    assert label.verdict == "borderline"
    assert label.rubric_scores == {}
    assert label.rubric_id is not None  # the rulebook is recorded even with no answers


def test_resubmitting_replaces_the_criteria_instead_of_merging(
    client: TestClient,
    make_trajectory: Callable[..., Trajectory],
    session_factory: sessionmaker[Session],
) -> None:
    """A second pass by the same annotator is a correction, not an addition.

    Merged, an answer the annotator has since removed would survive in the label as an
    opinion they no longer hold, and nothing on the form would show it was still there.
    """
    trajectory = make_trajectory()
    client.post(f"/label/{trajectory.id}", data=FULL_FORM, follow_redirects=False)
    client.post(
        f"/label/{trajectory.id}",
        data={
            "annotator": "momo",
            "verdict": "borderline",
            "criterion_goal_achieved": "0.5",
        },
        follow_redirects=False,
    )

    with session_factory() as check:
        label = check.scalars(select(HumanLabel)).one()
        assert check.scalar(select(func.count()).select_from(HumanLabel)) == 1

    assert label.verdict == "borderline"
    assert label.rubric_scores == {"goal_achieved": 0.5}


# --- what the form refuses ---------------------------------------------------


def test_a_score_the_criterion_does_not_offer_is_refused_and_nothing_is_saved(
    client: TestClient,
    make_trajectory: Callable[..., Trajectory],
    session_factory: sessionmaker[Session],
) -> None:
    """Only a hand-built POST reaches here, and it must not reach the table.

    0.5 is a permitted score for goal_achieved and not for tools_correct, which is a
    yes/no question. Stored, it would become a third category in that criterion's
    confusion matrix and take rows from both real answers. The re-rendered form keeps the
    answers already given, so a rejected submission does not cost the annotator the pass.
    """
    trajectory = make_trajectory()
    response = client.post(
        f"/label/{trajectory.id}",
        data={
            "annotator": "momo",
            "verdict": "pass",
            "criterion_goal_achieved": "1.0",
            "criterion_tools_correct": "0.5",
        },
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert "tools_correct" in response.text  # the message names what was refused
    assert re.search(r'name="criterion_goal_achieved" value="1\.0"\s+checked', response.text)

    with session_factory() as check:
        assert check.scalar(select(func.count()).select_from(HumanLabel)) == 0


def test_a_field_naming_a_criterion_the_rubric_does_not_have_is_ignored(
    client: TestClient,
    make_trajectory: Callable[..., Trajectory],
    session_factory: sessionmaker[Session],
) -> None:
    """An extra form field cannot smuggle a column into the calibration report.

    The route reads the criterion fields by key from ``CRITERIA`` rather than sweeping up
    everything prefixed ``criterion_``, so a stale field left in a template — or an
    invented one in a hand-built POST — contributes nothing instead of arriving in the
    report as a criterion no judge was ever asked about.
    """
    trajectory = make_trajectory()
    response = client.post(
        f"/label/{trajectory.id}",
        data={
            "annotator": "momo",
            "verdict": "pass",
            "criterion_goal_achieved": "1.0",
            "criterion_vibes": "1.0",
        },
        follow_redirects=False,
    )
    assert response.status_code in (302, 303)

    with session_factory() as check:
        label = check.scalars(select(HumanLabel)).one()

    assert label.rubric_scores == {"goal_achieved": 1.0}
