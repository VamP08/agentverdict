"""API tests for /api/trajectories/{id}/labels."""

import re
from collections.abc import Callable

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from agentverdict.models import HumanLabel, Trajectory

LABEL_PAYLOAD = {
    "annotator": "alice",
    "verdict": "pass",
    "rationale": "Verified the order before refunding.",
    # Two criteria answered and the rest left out, which is the ordinary shape of a
    # label: a criterion nobody answered carries no key, and consent_obtained in
    # particular is absent whenever the run took no irreversible action.
    "rubric_scores": {"goal_achieved": 1.0, "tools_correct": 1.0},
    "time_spent_s": 12.5,
}


def test_create_label(
    client: TestClient, make_trajectory: Callable[..., Trajectory]
) -> None:
    trajectory = make_trajectory()
    response = client.post(f"/api/trajectories/{trajectory.id}/labels", json=LABEL_PAYLOAD)
    assert response.status_code == 201
    body = response.json()
    assert body["trajectory_id"] == trajectory.id
    assert body["annotator"] == "alice"
    assert body["verdict"] == "pass"
    assert body["rationale"] == LABEL_PAYLOAD["rationale"]
    assert body["rubric_scores"] == LABEL_PAYLOAD["rubric_scores"]
    assert body["time_spent_s"] == 12.5
    assert body["id"]


def test_resubmit_same_annotator_replaces_in_place(
    client: TestClient,
    make_trajectory: Callable[..., Trajectory],
    session_factory: sessionmaker[Session],
) -> None:
    trajectory = make_trajectory()
    first = client.post(f"/api/trajectories/{trajectory.id}/labels", json=LABEL_PAYLOAD).json()

    updated = dict(LABEL_PAYLOAD, verdict="fail", rationale="Missed the confirmation step.")
    response = client.post(f"/api/trajectories/{trajectory.id}/labels", json=updated)
    assert response.status_code == 201
    second = response.json()
    assert second["id"] == first["id"]
    assert second["verdict"] == "fail"

    labels = client.get(f"/api/trajectories/{trajectory.id}/labels").json()
    assert len(labels) == 1
    assert labels[0]["verdict"] == "fail"
    assert labels[0]["rationale"] == "Missed the confirmation step."

    with session_factory() as check:
        row_count = check.scalar(select(func.count()).select_from(HumanLabel))
    assert row_count == 1


def test_second_annotator_adds_a_row(
    client: TestClient, make_trajectory: Callable[..., Trajectory]
) -> None:
    trajectory = make_trajectory()
    client.post(f"/api/trajectories/{trajectory.id}/labels", json=LABEL_PAYLOAD)
    response = client.post(
        f"/api/trajectories/{trajectory.id}/labels",
        json={"annotator": "bob", "verdict": "borderline"},
    )
    assert response.status_code == 201

    labels = client.get(f"/api/trajectories/{trajectory.id}/labels").json()
    assert len(labels) == 2
    assert {label["annotator"] for label in labels} == {"alice", "bob"}
    verdicts = {label["annotator"]: label["verdict"] for label in labels}
    assert verdicts == {"alice": "pass", "bob": "borderline"}


def test_unknown_trajectory_returns_404(client: TestClient) -> None:
    assert client.post("/api/trajectories/missing/labels", json=LABEL_PAYLOAD).status_code == 404
    assert client.get("/api/trajectories/missing/labels").status_code == 404


def test_scores_off_the_rubric_are_refused(
    client: TestClient,
    make_trajectory: Callable[..., Trajectory],
    session_factory: sessionmaker[Session],
) -> None:
    """A key or a value the rubric does not define never reaches the table.

    Stored, either one would be read back by the calibration report as an answer:
    an invented key becomes a column the judge was never asked to fill, and 0.7 on a
    yes/no criterion becomes a category of its own in the confusion matrix.
    """
    trajectory = make_trajectory()
    invented = client.post(
        f"/api/trajectories/{trajectory.id}/labels",
        json={"annotator": "alice", "verdict": "pass", "rubric_scores": {"vibes": 1.0}},
    )
    assert invented.status_code == 400
    assert "vibes" in invented.json()["detail"]

    off_scale = client.post(
        f"/api/trajectories/{trajectory.id}/labels",
        json={
            "annotator": "alice",
            "verdict": "pass",
            "rubric_scores": {"tools_correct": 0.7},
        },
    )
    assert off_scale.status_code == 400

    with session_factory() as check:
        assert check.scalar(select(func.count()).select_from(HumanLabel)) == 0


def test_invalid_verdict_returns_422(
    client: TestClient, make_trajectory: Callable[..., Trajectory]
) -> None:
    trajectory = make_trajectory()
    response = client.post(
        f"/api/trajectories/{trajectory.id}/labels",
        json={"annotator": "alice", "verdict": "excellent"},
    )
    assert response.status_code == 422


def test_reopening_your_own_label_prefills_it_rather_than_offering_a_blank_form(
    client: TestClient,
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """A blank form on a labeled run is a delete button wearing the shape of an edit.

    Submitting replaces the whole row. An annotator who reopened a run to fix a typo in
    the rationale would drop every criterion answer they had given, and nothing would say
    so -- the redirect looks identical either way.
    """
    trajectory = make_trajectory()
    form = {
        "annotator": "vamp",
        "verdict": "pass",
        "rationale": "Looked it up, reported the ETA.",
        "criterion_goal_achieved": "1.0",
        "criterion_ended_waiting": "1.0",
        "criterion_communication_clear": "0.0",
    }
    saved = client.post(f"/label/{trajectory.id}", data=form, follow_redirects=False)
    assert saved.status_code == 303

    # Same annotator, coming back. The cookie is what scopes the prefill.
    page = client.get(f"/label/{trajectory.id}")
    assert page.status_code == 200
    body = page.text
    assert "Looked it up, reported the ETA." in body
    # The three answered criteria come back selected, and the two untouched ones do not.
    assert body.count("checked") >= 4  # verdict + three criteria
    for field, value in (
        ("criterion_goal_achieved", "1.0"),
        ("criterion_ended_waiting", "1.0"),
        ("criterion_communication_clear", "0.0"),
    ):
        marker = f'name="{field}" value="{value}"'
        assert marker in body
        offset = body.index(marker)
        assert "checked" in body[offset : offset + 120], f"{field}={value} came back unselected"


def test_another_annotators_answers_are_never_prefilled(
    client: TestClient,
    make_trajectory: Callable[..., Trajectory],
) -> None:
    """Prefill must not become anchoring: the panel is measured on independent readings."""
    trajectory = make_trajectory()
    client.post(
        f"/label/{trajectory.id}",
        data={"annotator": "vamp", "verdict": "fail", "rationale": "Refunded with no consent."},
        follow_redirects=False,
    )
    # A second annotator arrives; the cookie now names them.
    page = client.get(f"/label/{trajectory.id}", cookies={"agentverdict_annotator": "momo"})

    assert page.status_code == 200
    # The page lists every annotator's stored label further down, by design. What must be
    # empty is the *form* -- the textarea momo is about to type into.
    textarea = re.search(r'<textarea[^>]*name="rationale"[^>]*>(.*?)</textarea>', page.text, re.S)
    assert textarea is not None
    assert textarea.group(1).strip() == ""


def test_an_api_label_that_scores_criteria_is_stamped_with_the_current_rubric(
    client: TestClient,
    make_trajectory: Callable[..., Trajectory],
    session: Session,
) -> None:
    """Criteria answered under no named rulebook would be provenance-blind forever.

    The web form stamps every save; an API caller scoring criteria gets the same
    treatment. A bare verdict stays unstamped -- that is what a round-one label
    legitimately looks like, and inventing provenance for it would claim the annotator
    saw criteria that did not exist when they judged.
    """
    scored = make_trajectory()
    bare = make_trajectory()

    with_criteria = client.post(
        f"/api/trajectories/{scored.id}/labels",
        json={"annotator": "api-ann", "verdict": "pass", "rubric_scores": {"tools_correct": 1.0}},
    )
    verdict_only = client.post(
        f"/api/trajectories/{bare.id}/labels",
        json={"annotator": "api-ann", "verdict": "pass"},
    )

    assert with_criteria.status_code == 201
    assert verdict_only.status_code == 201
    assert with_criteria.json()["rubric_id"] is not None
    assert verdict_only.json()["rubric_id"] is None
