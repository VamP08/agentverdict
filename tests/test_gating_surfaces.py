"""The gate's two surfaces: the ``compare`` command CI runs, and GET /api/comparisons.

The command's exit code *is* the product — it is the bit a GitHub Action turns into
a blocked merge — so these tests assert exit codes first and prose second. The
important assertions are the zeros: an inconclusive or improved run has to exit 0
*even with* ``--fail-on-regression``, or the gate blocks on noise and gets switched
off within a week.

The CLI is exercised end to end against a throwaway SQLite file, because a command
that works only against an injected session is not the command CI runs. The fixture
repoints ``AGENTVERDICT_DATABASE_URL`` and restores the cached settings and engine
afterwards, the same way tests/test_migrations.py does. The API tests use the
in-memory database and dependency override from conftest.
"""

import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner, Result

from agentverdict.cli import app as cli_app
from agentverdict.config import get_settings
from agentverdict.db import get_session_factory, reset_engine
from agentverdict.migrate import upgrade_to_head
from agentverdict.models import EvalRun, Judge, JudgeVerdict, Task, Trajectory

#: Six tasks, one trajectory each. Sorted order is the order the gate reads them in.
SUITE: tuple[str, ...] = tuple(f"gate-{position}" for position in range(1, 7))

ALL_PASS: dict[str, str] = dict.fromkeys(SUITE, "pass")
ALL_FAIL: dict[str, str] = dict.fromkeys(SUITE, "fail")
ALL_BORDERLINE: dict[str, str] = dict.fromkeys(SUITE, "borderline")

#: Three tasks fall, two rise, one holds. The suite mean drops by 0.167 and the
#: bootstrap interval still contains zero — the change is indistinguishable from
#: which tasks happen to be in the suite.
NOISY_BASE: dict[str, str] = dict(
    zip(SUITE, ("pass", "pass", "pass", "fail", "fail", "pass"), strict=True)
)
NOISY_CANDIDATE: dict[str, str] = dict(
    zip(SUITE, ("fail", "fail", "fail", "pass", "pass", "pass"), strict=True)
)

runner = CliRunner()


# --- CLI: agentverdict compare ------------------------------------------------


@pytest.fixture
def cli_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[sessionmaker[Session]]:
    """Point the application's own settings at a scratch SQLite file, then restore.

    The CLI resolves its database through ``get_settings``/``get_engine``, so the
    only honest way to drive it is to move those globals and put them back.
    """
    url = f"sqlite:///{(tmp_path / 'gate.db').as_posix()}"
    monkeypatch.setenv("AGENTVERDICT_DATABASE_URL", url)
    get_settings.cache_clear()
    reset_engine()
    try:
        upgrade_to_head(url)
        yield get_session_factory()
    finally:
        # Leave the process-wide caches exactly as they were found.
        reset_engine()
        get_settings.cache_clear()


def _store_runs(
    factory: sessionmaker[Session],
    base: Mapping[str, str],
    candidate: Mapping[str, str],
    *,
    same_judge: bool = True,
) -> tuple[str, str]:
    """Persist a baseline and a candidate eval run, one trajectory per named task.

    Each run gets its own trajectories for the same tasks — what replay produces,
    and what pairing by task key has to cope with.
    """
    session = factory()
    try:
        base_judge = Judge(name="gate-judge", model="llama-test-model")
        session.add(base_judge)
        candidate_judge = base_judge
        if not same_judge:
            candidate_judge = Judge(name="rival-judge", model="llama-test-model")
            session.add(candidate_judge)

        keys = sorted({*base, *candidate})
        tasks = {key: Task(key=key, prompt=f"Scenario for {key}.") for key in keys}
        session.add_all(tasks.values())
        session.flush()

        stored: list[EvalRun] = []
        for judge, verdicts in ((base_judge, base), (candidate_judge, candidate)):
            run = EvalRun(judge_id=judge.id, status="completed")
            session.add(run)
            session.flush()
            for key, verdict in verdicts.items():
                trajectory = Trajectory(task_id=tasks[key].id, source="replay")
                session.add(trajectory)
                session.flush()
                session.add(
                    JudgeVerdict(
                        eval_run_id=run.id,
                        judge_id=judge.id,
                        trajectory_id=trajectory.id,
                        verdict=verdict,
                    )
                )
            stored.append(run)
        session.commit()
        return stored[0].id, stored[1].id
    finally:
        session.close()


def _compare(*args: str) -> Result:
    """Run ``agentverdict compare`` exactly as CI would."""
    return runner.invoke(cli_app, ["compare", *args])


def _payload(result: Result) -> dict[str, Any]:
    """Parse the ``--json`` document the command printed."""
    text = result.stdout
    return json.loads(text[text.index("{") : text.rindex("}") + 1])


# --- CLI: agentverdict eval --run-id-file -------------------------------------
#
# The gate has to act on the run it just produced. Asking the database for "the newest
# run under this judge" is wrong on precisely the setup this project recommends -- a
# shared database, where two pull requests running at once each pick up whichever run
# finished last, and each then gates the other branch.


def _seed_judgeable(factory: sessionmaker[Session]) -> None:
    """One task with one unjudged trajectory, so ``eval --skip-replay`` has work."""
    session = factory()
    try:
        judge = Judge(name="gate-judge", model="llama-test-model")
        task = Task(key="gate-1", prompt="Scenario for gate-1.")
        session.add_all((judge, task))
        session.flush()
        session.add(Trajectory(task_id=task.id, source="replay"))
        session.commit()
    finally:
        session.close()


def test_eval_writes_the_run_it_created_to_the_run_id_file(
    cli_db: sessionmaker[Session], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentverdict.judging import runner as judging_runner
    from agentverdict.judging.client import ChatResult

    _seed_judgeable(cli_db)

    class _StubClient:
        def chat_json(
            self, model: str, messages: list[dict[str, str]], **kwargs: Any
        ) -> ChatResult:
            body = '{"verdict": "pass", "rationale": "Did the thing.", "rubric_scores": {}}'
            return ChatResult(content=body, input_tokens=10, output_tokens=5, latency_ms=1.0)

        def close(self) -> None:
            pass

    monkeypatch.setattr(judging_runner, "GroqChatClient", _StubClient)
    target = tmp_path / "candidate-run-id"

    result = runner.invoke(
        cli_app,
        ["eval", "--judge", "gate-judge", "--skip-replay", "--run-id-file", str(target)],
    )

    assert result.exit_code == 0
    written = target.read_text(encoding="utf-8").strip()
    session = cli_db()
    try:
        stored = session.get(EvalRun, written)
    finally:
        session.close()
    # The full id, not the eight-character form the console prints for reading.
    assert stored is not None
    assert stored.id == written


def test_eval_clears_a_stale_run_id_when_it_cannot_produce_one(
    cli_db: sessionmaker[Session], tmp_path: Path
) -> None:
    # A caller reads this file to learn what *this* invocation produced. Leaving an
    # earlier id behind when the eval dies would hand the gate a run that has nothing
    # to do with the code under test, and it would compare it without complaint.
    target = tmp_path / "candidate-run-id"
    target.write_text("yesterdays-run-id\n", encoding="utf-8")

    result = runner.invoke(
        cli_app, ["eval", "--judge", "no-such-judge", "--run-id-file", str(target)]
    )

    assert result.exit_code != 0
    assert not target.exists()


def test_compare_exits_one_on_a_regression_with_the_flag(
    cli_db: sessionmaker[Session],
) -> None:
    base_id, candidate_id = _store_runs(cli_db, ALL_PASS, ALL_BORDERLINE)

    result = _compare(base_id, candidate_id, "--fail-on-regression")

    # Exit 1 is the whole feature: it is what blocks the merge.
    assert result.exit_code == 1
    assert "Outcome: regression" in result.output
    assert "Gate: FAILED" in result.output
    assert "-0.500" in result.output


def test_compare_exits_zero_on_a_regression_without_the_flag(
    cli_db: sessionmaker[Session],
) -> None:
    base_id, candidate_id = _store_runs(cli_db, ALL_PASS, ALL_BORDERLINE)

    result = _compare(base_id, candidate_id)

    # The regression is still found and reported; only the flag makes it fatal, so
    # a team can watch the gate for a week before letting it block anything.
    assert result.exit_code == 0
    assert "Outcome: regression" in result.output
    assert "Gate: reporting only" in result.output


def test_compare_exits_zero_on_an_inconclusive_result_even_with_the_flag(
    cli_db: sessionmaker[Session],
) -> None:
    """The suite average fell and the command still succeeds. That is the design."""
    base_id, candidate_id = _store_runs(cli_db, NOISY_BASE, NOISY_CANDIDATE)

    result = _compare(base_id, candidate_id, "--fail-on-regression")

    assert result.exit_code == 0
    assert "Outcome: inconclusive" in result.output
    assert "Gate: passed" in result.output
    assert "-0.167" in result.output  # the mean really did drop
    assert "Outcome: regression" not in result.output


def test_compare_exits_zero_on_an_improvement_even_with_the_flag(
    cli_db: sessionmaker[Session],
) -> None:
    base_id, candidate_id = _store_runs(cli_db, ALL_FAIL, ALL_PASS)

    result = _compare(base_id, candidate_id, "--fail-on-regression")

    assert result.exit_code == 0
    assert "Outcome: improvement" in result.output
    assert "Gate: passed" in result.output


def test_compare_exits_two_on_an_unknown_base_run_id(cli_db: sessionmaker[Session]) -> None:
    _, candidate_id = _store_runs(cli_db, ALL_PASS, ALL_PASS)

    result = _compare("no-such-run", candidate_id)

    # A gate that answered "inconclusive" to a typo would silently stop gating.
    assert result.exit_code == 2  # cannot compare, not a measured regression
    assert "no-such-run" in result.output
    assert "the base run" in result.output
    assert "Outcome:" not in result.output


def test_compare_exits_two_on_an_unknown_candidate_run_id(cli_db: sessionmaker[Session]) -> None:
    base_id, _ = _store_runs(cli_db, ALL_PASS, ALL_PASS)

    result = _compare(base_id, "no-such-run", "--fail-on-regression")

    assert result.exit_code == 2  # cannot compare, not a measured regression
    assert "the candidate run" in result.output


def test_compare_exits_two_when_the_two_runs_used_different_judges(
    cli_db: sessionmaker[Session],
) -> None:
    base_id, candidate_id = _store_runs(cli_db, ALL_PASS, ALL_FAIL, same_judge=False)

    result = _compare(base_id, candidate_id, "--fail-on-regression")

    assert result.exit_code == 2  # cannot compare, not a measured regression
    assert "Cannot compare these runs" in result.output
    assert "gate-judge" in result.output
    assert "rival-judge" in result.output
    # A drop this large must not be reported as a regression: it measures the
    # judge swap at least as much as the agent.
    assert "Outcome:" not in result.output


def test_compare_reports_a_suite_too_small_to_resample(cli_db: sessionmaker[Session]) -> None:
    base_id, candidate_id = _store_runs(cli_db, {"only-task": "pass"}, {"only-task": "fail"})

    result = _compare(base_id, candidate_id, "--fail-on-regression")

    assert result.exit_code == 0
    assert "too small to resample" in result.output
    # Neither a pass nor a failure: the gate never ran, and says so rather than
    # letting a green exit code read as "no regression found".
    assert "this is not a passing gate" in result.output
    assert "Outcome: regression" not in result.output


def test_compare_names_the_tasks_only_one_run_scored(cli_db: sessionmaker[Session]) -> None:
    base_id, candidate_id = _store_runs(
        cli_db,
        {**ALL_PASS, "retired-case": "pass"},
        {**ALL_PASS, "added-case": "pass"},
    )

    result = _compare(base_id, candidate_id)

    assert result.exit_code == 0
    assert "Excluded from the statistics: 2 tasks present in one run only" in result.output
    assert "retired-case" in result.output
    assert "added-case" in result.output


def test_compare_json_carries_the_machine_readable_verdict(
    cli_db: sessionmaker[Session],
) -> None:
    base_id, candidate_id = _store_runs(cli_db, ALL_PASS, ALL_BORDERLINE)

    result = _compare(base_id, candidate_id, "--json", "--fail-on-regression")

    assert result.exit_code == 1
    assert result.stdout.strip().startswith("{")  # nothing a parser has to skip
    payload = _payload(result)
    assert payload["base_run_id"] == base_id
    assert payload["candidate_run_id"] == candidate_id
    assert payload["judge_name"] == "gate-judge"
    assert payload["outcome"] == "regression"
    assert payload["blocks_merge"] is True
    assert payload["tasks_compared"] == len(SUITE)
    assert payload["base_score"] == 1.0
    assert payload["candidate_score"] == 0.5
    assert payload["delta"] == -0.5
    assert payload["delta_ci_low"] == -0.5
    assert payload["delta_ci_high"] == -0.5
    assert [row["task_key"] for row in payload["task_deltas"]] == sorted(SUITE)
    assert payload["summary"].startswith("Regression:")


def test_compare_json_of_a_noisy_change_does_not_block(cli_db: sessionmaker[Session]) -> None:
    base_id, candidate_id = _store_runs(cli_db, NOISY_BASE, NOISY_CANDIDATE)

    result = _compare(base_id, candidate_id, "--json", "--fail-on-regression")

    assert result.exit_code == 0
    payload = _payload(result)
    assert payload["delta"] < 0.0
    assert payload["delta_ci_low"] < 0.0 < payload["delta_ci_high"]
    assert payload["outcome"] == "inconclusive"
    assert payload["blocks_merge"] is False


# --- API: GET /api/comparisons ------------------------------------------------


@pytest.fixture
def seed_run(
    make_eval_run: Callable[..., EvalRun],
    make_task: Callable[..., Task],
    make_trajectory: Callable[..., Trajectory],
    make_verdict: Callable[..., JudgeVerdict],
) -> Callable[[Judge, Mapping[str, str]], EvalRun]:
    """Store one eval run scoring each named task once, in the in-memory database.

    Tasks are shared between calls; trajectories never are.
    """
    tasks: dict[str, Task] = {}

    def _seed(judge: Judge, verdicts: Mapping[str, str]) -> EvalRun:
        run = make_eval_run(judge)
        for key, verdict in verdicts.items():
            if key not in tasks:
                tasks[key] = make_task(key=key)
            make_verdict(judge, make_trajectory(task=tasks[key]), verdict, eval_run=run)
        return run

    return _seed


def test_comparisons_endpoint_returns_the_comparison(
    client: TestClient,
    make_judge: Callable[..., Judge],
    seed_run: Callable[[Judge, Mapping[str, str]], EvalRun],
) -> None:
    judge = make_judge(name="gate-judge", model="llama-test-model")
    base = seed_run(judge, ALL_PASS)
    candidate = seed_run(judge, ALL_BORDERLINE)

    response = client.get(
        "/api/comparisons",
        params={"base_run_id": base.id, "candidate_run_id": candidate.id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["base_run_id"] == base.id
    assert body["candidate_run_id"] == candidate.id
    assert body["judge_id"] == judge.id
    assert body["judge_name"] == "gate-judge"
    assert body["tasks_compared"] == len(SUITE)
    assert body["base_score"] == 1.0
    assert body["candidate_score"] == 0.5
    assert body["delta"] == -0.5
    assert body["delta_ci_low"] == -0.5
    assert body["delta_ci_high"] == -0.5
    assert body["outcome"] == "regression"
    assert body["blocks_merge"] is True
    assert body["base_only_tasks"] == []
    assert body["candidate_only_tasks"] == []
    # The evidence comes back with the verdict, so a reviewer can check it.
    assert [row["task_key"] for row in body["task_deltas"]] == sorted(SUITE)
    assert body["task_deltas"][0]["delta"] == -0.5
    assert body["task_deltas"][0]["base_trajectories"] == 1


def test_comparisons_endpoint_does_not_block_on_noise(
    client: TestClient,
    make_judge: Callable[..., Judge],
    seed_run: Callable[[Judge, Mapping[str, str]], EvalRun],
) -> None:
    judge = make_judge(name="gate-judge")
    base = seed_run(judge, NOISY_BASE)
    candidate = seed_run(judge, NOISY_CANDIDATE)

    response = client.get(
        "/api/comparisons",
        params={"base_run_id": base.id, "candidate_run_id": candidate.id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["delta"] < 0.0  # the mean fell...
    assert body["delta_ci_low"] < 0.0 < body["delta_ci_high"]  # ...within the noise
    assert body["outcome"] == "inconclusive"
    assert body["blocks_merge"] is False


def test_comparisons_endpoint_excludes_tasks_only_one_run_scored(
    client: TestClient,
    make_judge: Callable[..., Judge],
    seed_run: Callable[[Judge, Mapping[str, str]], EvalRun],
) -> None:
    judge = make_judge(name="gate-judge")
    base = seed_run(judge, {"shared-1": "pass", "shared-2": "pass", "retired": "fail"})
    candidate = seed_run(judge, {"shared-1": "pass", "shared-2": "pass", "added": "fail"})

    body = client.get(
        "/api/comparisons",
        params={"base_run_id": base.id, "candidate_run_id": candidate.id},
    ).json()

    assert body["tasks_compared"] == 2
    assert body["base_only_tasks"] == ["retired"]
    assert body["candidate_only_tasks"] == ["added"]
    assert body["base_score"] == 1.0  # the excluded pair moves neither score
    assert body["candidate_score"] == 1.0
    assert [row["task_key"] for row in body["task_deltas"]] == ["shared-1", "shared-2"]


def test_comparisons_endpoint_unknown_base_run_returns_404(
    client: TestClient,
    make_judge: Callable[..., Judge],
    seed_run: Callable[[Judge, Mapping[str, str]], EvalRun],
) -> None:
    candidate = seed_run(make_judge(), ALL_PASS)

    response = client.get(
        "/api/comparisons",
        params={"base_run_id": "no-such-run", "candidate_run_id": candidate.id},
    )

    assert response.status_code == 404
    # The id is echoed back: "one of your two run ids is wrong" is not actionable.
    assert "no-such-run" in response.json()["detail"]


def test_comparisons_endpoint_unknown_candidate_run_returns_404(
    client: TestClient,
    make_judge: Callable[..., Judge],
    seed_run: Callable[[Judge, Mapping[str, str]], EvalRun],
) -> None:
    base = seed_run(make_judge(), ALL_PASS)

    response = client.get(
        "/api/comparisons",
        params={"base_run_id": base.id, "candidate_run_id": "gone-missing"},
    )

    assert response.status_code == 404
    assert "gone-missing" in response.json()["detail"]


def test_comparisons_endpoint_mismatched_judges_returns_400(
    client: TestClient,
    make_judge: Callable[..., Judge],
    seed_run: Callable[[Judge, Mapping[str, str]], EvalRun],
) -> None:
    base = seed_run(make_judge(name="gate-judge"), ALL_PASS)
    candidate = seed_run(make_judge(name="rival-judge"), ALL_FAIL)

    response = client.get(
        "/api/comparisons",
        params={"base_run_id": base.id, "candidate_run_id": candidate.id},
    )

    # A bad request, not a server fault: the two runs are not on a common scale.
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "gate-judge" in detail
    assert "rival-judge" in detail


def test_comparisons_endpoint_requires_both_run_ids(client: TestClient) -> None:
    assert client.get("/api/comparisons").status_code == 422
