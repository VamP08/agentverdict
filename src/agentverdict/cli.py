"""Typer CLI: init-db, import, export, stats, serve, judge, replay, eval, calibrate, probe, compare.

Schema creation goes through Alembic (``agentverdict.migrate``) so upgrades of an
existing database work; ``db.init_db()`` remains the fast create_all path for tests.

Heavy or optional machinery — alembic, uvicorn, the agents package, the judging
package — is imported inside the command bodies so plain data commands start fast
and work without a model API key. ``calibrate`` and ``compare`` are two of those
key-free commands: they only read rows that are already stored, which is what
makes them safe to run inside a CI gate. ``probe`` is the exception in this family:
it re-asks the judge about every stored trajectory under every arm, so it spends
tokens and never belongs inside the gate.
"""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agentverdict.db import get_session_factory
from agentverdict.importer import compute_stats, export_jsonl, import_jsonl
from agentverdict.models import (
    BiasProbeResult,
    BiasProbeRun,
    EvalRun,
    Judge,
    JudgeVerdict,
    Rubric,
    Task,
    Trajectory,
)
from agentverdict.schemas import (
    AgreementRead,
    ArmOutcome,
    BiasProbeReport,
    CalibrationReport,
    ReplayReport,
    RunComparison,
)

if TYPE_CHECKING:  # import-time cost avoided; only needed for annotations
    from agentverdict.agents.base import AgentAdapter

#: Adapter names accepted by ``--adapter``. Adding an agent means adding a builder here.
ADAPTERS = ("groq-agent",)

#: Disagreements listed in the text calibration report; ``--json`` carries every row.
_DISAGREEMENT_LIMIT = 10
#: Characters of judge rationale kept on a disagreement line.
_RATIONALE_WIDTH = 88
#: Widest task key printed in full before it is trimmed, so the columns stay aligned.
_TASK_KEY_WIDTH = 36
#: Exit codes for ``compare``. CI has to tell these apart: 1 means a regression was
#: measured and the merge should stop; 2 means the comparison could not be made at
#: all (an id that does not resolve, two runs from different judges). Collapsing
#: them into one nonzero code makes a missing baseline indistinguishable from a
#: broken agent, and the pull request gets told its quality dropped when nobody
#: measured anything.
EXIT_REGRESSION = 1
EXIT_CANNOT_COMPARE = 2

#: Task rows printed by ``compare``; ``--json`` always carries every row.
_TASK_DELTA_LIMIT = 15
#: Width the rubric-mix warning wraps to: it is one long sentence by design (every
#: surface prints it verbatim), and one unbroken line would be the easiest way to
#: make sure nobody reads it.
_WARNING_WIDTH = 88
#: Width the excluded-task name lists wrap to, so a large suite stays readable.
_EXCLUDED_WIDTH = 88

#: Probe name -> what the report calls it. The literature's terms are for *pairwise*
#: judges ("here are A and B, which is better?") and this judge is a pointwise
#: absolute grader, so each probe is re-derived and named for what it actually
#: perturbs rather than borrowing "position bias" and "verbosity bias".
_PROBE_TITLES = {"order": "order sensitivity", "length": "length sensitivity"}
#: Probe name -> what its arms do to the prompt, printed so a reader never has to
#: guess what "reversed_context" meant.
_PROBE_SUMMARY = {
    "order": "prompt sections re-ordered; the transcript itself is never re-ordered",
    "length": "filler appended to agent turns; nothing is deleted or rewritten",
}

app = typer.Typer(
    name="agentverdict",
    help="AgentVerdict: golden-dataset workbench for tool-calling agent evaluation.",
    no_args_is_help=True,
)
judge_app = typer.Typer(help="Manage LLM judges and evaluation runs.", no_args_is_help=True)
app.add_typer(judge_app, name="judge")


@contextmanager
def _open_session() -> Iterator[Session]:
    """Yield a session against an up-to-date database; commit on success."""
    from agentverdict.migrate import upgrade_to_head

    upgrade_to_head()
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _require_judge(session: Session, name: str) -> Judge:
    """Look up a judge by name, exiting with a helpful message when it is missing."""
    judge = session.scalar(select(Judge).where(Judge.name == name))
    if judge is None:
        typer.echo(
            f"No judge named '{name}'. Register it with: agentverdict judge add {name}",
            err=True,
        )
        raise typer.Exit(code=1)
    return judge


def _require_eval_run(session: Session, judge: Judge, eval_run_id: str) -> EvalRun:
    """Look up one of this judge's eval runs, exiting when it is unknown or foreign."""
    run = session.get(EvalRun, eval_run_id)
    if run is None:
        typer.echo(
            f"No eval run with id '{eval_run_id}'. List runs with: agentverdict judge list",
            err=True,
        )
        raise typer.Exit(code=1)
    if run.judge_id != judge.id:
        typer.echo(
            f"Eval run {eval_run_id[:8]} belongs to judge '{run.judge.name}', not '{judge.name}'.",
            err=True,
        )
        raise typer.Exit(code=1)
    return run


def _require_run(session: Session, eval_run_id: str, role: str) -> EvalRun:
    """Look up an eval run by id alone, exiting when the id is unknown.

    Unlike ``_require_eval_run`` this does not tie the run to a named judge:
    ``compare`` is handed two ids and takes the judge from the runs themselves.
    Whether the two runs share a judge is the comparison's decision, not the
    lookup's — a gate that quietly compared verdicts from two different judges
    would be measuring the judges, not the agent.
    """
    run = session.get(EvalRun, eval_run_id)
    if run is None:
        typer.echo(
            f"No eval run with id '{eval_run_id}' (the {role} run)."
            " Run summaries print a shortened id; compare needs the full one.",
            err=True,
        )
        # Not a regression: nothing was measured. See EXIT_CANNOT_COMPARE.
        raise typer.Exit(code=EXIT_CANNOT_COMPARE)
    return run


def _build_adapter(adapter: str, model: str | None) -> AgentAdapter:
    """Construct the named agent-under-test, wired to the bundled mock tool fixture."""
    if adapter not in ADAPTERS:
        typer.echo(
            f"Unknown adapter '{adapter}'. Valid adapters: {', '.join(ADAPTERS)}",
            err=True,
        )
        raise typer.Exit(code=1)

    from agentverdict.agents.groq_agent import GroqReferenceAgent
    from agentverdict.agents.mock_tools import MockToolRegistry, default_fixture_path

    # Load the fixture by explicit path so a missing or malformed file is reported
    # here, rather than silently yielding an agent whose every tool call errors.
    fixture = default_fixture_path()
    try:
        registry = MockToolRegistry.from_file(fixture)
    except (OSError, ValueError) as exc:
        typer.echo(f"Could not load mock tools from {fixture}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    return GroqReferenceAgent(model=model, registry=registry)


def _replay_progress(
    position: int,
    total: int,
    task: Task,
    trajectory: Trajectory | None,
    error: str | None,
) -> None:
    """Print one replayed attempt as it lands (exactly one of trajectory/error is set)."""
    if trajectory is None:
        typer.echo(f"  [{position}/{total}] {task.key}: ERROR: {(error or 'unknown')[:80]}")
        return
    typer.echo(
        f"  [{position}/{total}] {task.key}: {trajectory.status} ({len(trajectory.steps)} steps)"
    )


def _judge_progress(position: int, total: int, trajectory: Trajectory, row: JudgeVerdict) -> None:
    """Print one judged trajectory as it lands."""
    outcome = row.verdict or f"ERROR: {(row.error or 'unknown')[:60]}"
    typer.echo(f"  [{position}/{total}] {trajectory.task.key} {trajectory.id[:8]}: {outcome}")


def _probe_progress(
    position: int, total: int, trajectory: Trajectory, row: BiasProbeResult
) -> None:
    """Print one probe call as it lands, at one line per call rather than per trajectory.

    A probe makes arms x repeats calls per trajectory, and the arm is the whole
    point, so collapsing them to one line per trajectory would hide which arm was
    running when a sweep stalled or started erroring.
    """
    if row.verdict is None and row.error is None:
        # Neither judged nor failed: the delivery check found this arm rendering the
        # control's bytes, so no call was made for it.
        outcome = "not applied"
    else:
        outcome = row.verdict or f"ERROR: {(row.error or 'unknown')[:60]}"
    typer.echo(
        f"  [{position}/{total}] {trajectory.task.key} {trajectory.id[:8]}"
        f"  {row.variant} r{row.repeat}: {outcome}"
    )


def _print_replay_summary(report: ReplayReport) -> None:
    noun = "trajectory" if report.trajectories_created == 1 else "trajectories"
    tasks = "task" if report.tasks_attempted == 1 else "tasks"
    typer.echo(
        f"Replay complete: {report.trajectories_created} {noun} created from "
        f"{report.tasks_attempted} {tasks} via {report.adapter}"
    )
    if report.error_count:
        typer.echo(f"  errors      {report.error_count}")
        for error in report.errors:
            typer.echo(f"    ! {error}", err=True)
    typer.echo(f"  tokens      {report.input_tokens} in / {report.output_tokens} out")


def _print_eval_summary(run: EvalRun) -> None:
    typer.echo(f"Eval run {run.id[:8]} completed: {run.trajectory_count} trajectories")
    for verdict, count in sorted(run.verdict_counts.items()):
        typer.echo(f"  {verdict:<11} {count}")
    if run.error_count:
        typer.echo(f"  errors      {run.error_count}")
    typer.echo(f"  tokens      {run.input_tokens} in / {run.output_tokens} out")
    agreement = (run.meta or {}).get("human_agreement")
    if agreement:
        typer.echo(
            f"  human agreement: {agreement['agree']}/{agreement['total']}"
            f" ({agreement['rate']:.0%})"
        )


def _count(value: int, singular: str, plural: str) -> str:
    """Pluralize a count for prose: 1 trajectory, 3 trajectories."""
    return f"{value} {singular if value == 1 else plural}"


def _fmt_stat(value: float | None, digits: int = 3) -> str:
    """Format a statistic, or "n/a" when it is undefined — never a fake 0.0."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _fmt_delta(value: float | None, digits: int = 3) -> str:
    """Format a change with an explicit sign: "-0.075" and "+0.075" must not look alike."""
    return "n/a" if value is None else f"{value:+.{digits}f}"


def _fmt_ci(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "CI n/a"
    return f"95% CI [{low:.3f}, {high:.3f}]"


def _fmt_interval(low: float | None, high: float | None, *, signed: bool = True) -> str:
    """Bracketed interval, with no confidence level baked into the string.

    ``_fmt_ci`` names 95% because that is the level ``calibrate`` and ``compare``
    read their intervals at. A probe reads its intervals at the Sidak-adjusted
    level instead, which depends on how many arms the run had, so the level is
    printed once by the report rather than repeated wrongly on every row.

    ``signed`` is on for the shifts, where the direction is the claim and "-0.075"
    must not look like "+0.075", and off for the disagreement rates, which cannot
    be negative and would only look odd carrying a plus sign.
    """
    if low is None or high is None:
        return "n/a"
    if signed:
        return f"[{low:+.3f}, {high:+.3f}]"
    return f"[{low:.3f}, {high:.3f}]"


def _trim(text: str | None, width: int) -> str:
    """Collapse text onto one line and cut it to width with an ellipsis."""
    if not text:
        return ""
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    return flat[: width - 3].rstrip() + "..."


def _labels(agreement: AgreementRead) -> list[str]:
    """The verdict vocabulary to render, falling back to the canonical order."""
    from agentverdict.calibration.stats import VERDICT_ORDER

    return list(agreement.labels) or list(VERDICT_ORDER)


def _calibration_scope(report: CalibrationReport) -> str:
    parts: list[str] = []
    if report.eval_run_id:
        parts.append(f"eval run {report.eval_run_id[:8]}")
    if report.task_key:
        parts.append(f"task '{report.task_key}'")
    # An annotator filter changes which labels count, so it belongs in the scope
    # line: a second annotation round scored on its own must never be mistaken
    # for the whole dataset.
    if report.annotators:
        parts.append(f"annotators {', '.join(report.annotators)}")
    return ", ".join(parts) if parts else "whole dataset (every run, every task)"


def _print_calibration_header(report: CalibrationReport) -> None:
    typer.echo(f"Calibration: judge '{report.judge_name}'  [{report.judge_model}]")
    typer.echo(f"  scope      {_calibration_scope(report)}")
    # The rulebooks are named, never averaged silently: a kappa computed across a
    # rubric edit describes neither rulebook, and nothing else on the page would
    # show the split.
    rubrics = ", ".join(
        f"{name} ({_count(count, 'label', 'labels')})"
        for name, count in report.label_rubrics.items()
    )
    prompts = ", ".join(
        f"{version} ({_count(count, 'verdict', 'verdicts')})"
        for version, count in report.judge_prompt_versions.items()
    )
    typer.echo(f"  rubric     {rubrics or 'no labels in scope'}")
    typer.echo(f"  prompt     {prompts or 'no verdicts in scope'}")
    stamp = report.generated_at
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(UTC)
    typer.echo(f"  generated  {stamp:%Y-%m-%d %H:%M:%S} UTC")
    if report.rubric_warning:
        # Verbatim, and before any statistic: a reader who has already taken the
        # headline kappa at face value cannot be un-misled by a footnote.
        typer.echo("")
        for line in textwrap.wrap(
            f"WARNING: {report.rubric_warning}",
            width=_WARNING_WIDTH,
            initial_indent="  ",
            subsequent_indent="  ",
        ):
            typer.echo(line)


def _print_calibration_coverage(report: CalibrationReport) -> None:
    typer.echo("Coverage")
    typer.echo(f"  judged trajectories   {report.judged_trajectories:>5}")
    typer.echo(f"  labeled trajectories  {report.labeled_trajectories:>5}")
    typer.echo(f"  compared              {report.compared:>5}")
    if report.judge_errors:
        typer.echo(f"  judge errors          {report.judge_errors:>5}  (failed calls, excluded)")
    if report.ties_excluded:
        typer.echo(f"  ties excluded         {report.ties_excluded:>5}  (no human majority)")


def _print_calibration_headline(agreement: AgreementRead) -> None:
    matched = sum(agreement.matrix.get(label, {}).get(label, 0) for label in _labels(agreement))
    typer.echo("Agreement with the human majority")
    typer.echo(
        f"  Cohen's kappa        {_fmt_stat(agreement.cohens_kappa):>6}"
        f"  {_fmt_ci(agreement.kappa_ci_low, agreement.kappa_ci_high)}"
        f"  ({agreement.interpretation})"
    )
    if (
        agreement.kappa_ci_low is None
        and agreement.bootstrap_iterations
        and agreement.bootstrap_usable < agreement.bootstrap_iterations
    ):
        # Say why the interval is missing: a high discard rate is itself the
        # finding — the sample is too lopsided to resample meaningfully.
        typer.echo(
            f"    no interval: only {agreement.bootstrap_usable} of "
            f"{agreement.bootstrap_iterations} resamples had a defined kappa"
        )
    typer.echo(
        f"  weighted kappa       {_fmt_stat(agreement.weighted_kappa):>6}"
        "  ordinal: fail < borderline < pass"
    )
    typer.echo(
        f"  observed agreement   {_fmt_pct(agreement.observed_agreement):>6}"
        f"  {matched} of {agreement.n} matched"
    )


def _print_judge_vs_annotator(report: CalibrationReport) -> None:
    """Print the judge against each annotator on exactly the items both of them rated.

    The headline above is scored against the human *majority*, which drops the
    trajectories the annotators tied on — with two annotators that is every
    disagreement, so the judge is graded on an easier sample than the humans are.
    These rows put every rater on the same items, which is what makes the judge's
    number and the human numbers comparable at all.
    """
    rows = report.judge_vs_annotator
    if not rows:
        return
    typer.echo("Judge against each annotator (same items, like-for-like)")
    stub = max(len(row.annotator) for row in rows)
    count = max(len(str(row.n)) for row in rows)
    for row in rows:
        typer.echo(
            f"  {row.annotator:<{stub}}   n={str(row.n).rjust(count)}"
            f"  kappa {_fmt_stat(row.cohens_kappa):>6}"
            f"  agreement {_fmt_pct(row.observed_agreement):>6}"
        )
    typer.echo("  These use the same trajectories as the human numbers below.")
    if report.ties_excluded:
        # Only claim the samples differ when trajectories were actually dropped.
        dropped = _count(report.ties_excluded, "tied trajectory", "tied trajectories")
        typer.echo(f"  The headline kappa above dropped {dropped}, so it")
        typer.echo("  is measured on a different sample and is not comparable to these rows.")


def _print_confusion_matrix(agreement: AgreementRead) -> None:
    """Render human-vs-judge counts as a small aligned table with row/column totals."""
    labels = _labels(agreement)
    rows: list[tuple[str, list[int]]] = []
    for label in labels:
        counts = [agreement.matrix.get(label, {}).get(column, 0) for column in labels]
        rows.append((label, [*counts, sum(counts)]))
    # Column totals are computed before the totals row itself joins `rows`.
    totals = [sum(row[position] for _, row in rows) for position in range(len(labels) + 1)]
    rows.append(("total", totals))

    headers = [*labels, "total"]
    stub = max(len(name) for name, _ in rows)
    widths = [
        max(len(header), *(len(str(row[position])) for _, row in rows))
        for position, header in enumerate(headers)
    ]
    typer.echo("Confusion matrix (rows: human majority, columns: judge)")
    head = "  ".join(header.rjust(width) for header, width in zip(headers, widths, strict=True))
    typer.echo(f"  {'':<{stub}}  {head}")
    for name, row in rows:
        cells = "  ".join(str(count).rjust(width) for count, width in zip(row, widths, strict=True))
        typer.echo(f"  {name:<{stub}}  {cells}")


def _print_calibration_per_class(agreement: AgreementRead) -> None:
    if not agreement.per_class:
        return
    stub = max(len("verdict"), *(len(metrics.label) for metrics in agreement.per_class))
    typer.echo("Per-verdict scores (human majority as truth)")
    typer.echo(f"  {'verdict':<{stub}}  {'precision':>9}  {'recall':>7}  {'f1':>7}  {'support':>7}")
    for metrics in agreement.per_class:
        typer.echo(
            f"  {metrics.label:<{stub}}  {_fmt_stat(metrics.precision):>9}"
            f"  {_fmt_stat(metrics.recall):>7}  {_fmt_stat(metrics.f1):>7}"
            f"  {metrics.support:>7}"
        )


def _print_human_ceiling(report: CalibrationReport, *, reading_aid: bool = True) -> None:
    """Print how well the humans agree — the context the judge's kappa needs."""
    typer.echo("Human agreement")
    if not report.human_ceiling and not report.human_baseline:
        if report.annotators:
            # The filter removed the other raters, not the dataset. Telling the
            # reader to double-label here would send them after a no-op.
            named = ", ".join(report.annotators)
            typer.echo(f"  Only labels from {named} count in this scope, so no second rater is")
            typer.echo("  left to compare against. Drop --annotator, or name another annotator,")
            typer.echo("  to read the judge against the human bar.")
        else:
            typer.echo("  No two annotators have labeled the same trajectory, so there is nothing")
            typer.echo("  to compare the judge against. Double-label a slice of the set to get it.")
        return

    if report.human_baseline:
        typer.echo("  Each annotator vs the majority of the others (the like-for-like number):")
        stub = max(len(row.annotator) for row in report.human_baseline)
        count = max(len(str(row.n)) for row in report.human_baseline)
        for row in report.human_baseline:
            typer.echo(
                f"    {row.annotator:<{stub}}  n={str(row.n).rjust(count)}"
                f"  kappa {_fmt_stat(row.cohens_kappa):>6}"
                f"  agreement {_fmt_pct(row.observed_agreement):>6}"
            )

    if report.human_ceiling:
        typer.echo("  Annotator against annotator, on what they both labeled:")
        stub = max(
            len(f"{pair.annotator_a} vs {pair.annotator_b}") for pair in report.human_ceiling
        )
        count = max(len(str(pair.n)) for pair in report.human_ceiling)
        for pair in report.human_ceiling:
            name = f"{pair.annotator_a} vs {pair.annotator_b}"
            typer.echo(
                f"    {name:<{stub}}  n={str(pair.n).rjust(count)}"
                f"  kappa {_fmt_stat(pair.cohens_kappa):>6}"
                f"  agreement {_fmt_pct(pair.observed_agreement):>6}"
            )

    if reading_aid:
        typer.echo("  Compare the judge against the first block, not the second: the judge is")
        typer.echo("  scored against the human majority, and a majority is quieter than any one")
        typer.echo("  annotator, so pairwise numbers set an unfairly low bar.")


def _print_calibration_criteria(report: CalibrationReport) -> None:
    """One row per rubric criterion, over the trajectories both sides scored it on.

    The marginals sit beside kappa because a criterion nearly every run passes
    degenerates: two raters who answer alike 29 times in 30 can still score near
    zero, and only the counts separate that from a criterion they truly split on.
    ``unanswered`` sits beside each n for the reason the coverage block exists at
    all — a kappa computed on three trajectories must not be read as one computed
    on all of them.
    """
    if not report.criteria:
        return
    typer.echo("Per-criterion agreement (judge vs the human panel, where both answered)")
    if all(row.n == 0 for row in report.criteria):
        # An empty table is a fact about the corpus, not about the judge: five n=0
        # kappas would read as five criteria the two sides never once agreed on.
        typer.echo("  Nothing measured yet: no compared trajectory carries criterion answers")
        typer.echo("  from both sides, so these columns await annotator answers. Label a slice")
        typer.echo("  with the criteria filled in, then rerun calibrate.")
        return
    headers = ("criterion", "n", "unanswered", "kappa", "agreement", "human 1.0", "judge 1.0")
    rows = [
        (
            row.key,
            str(row.n),
            str(report.criteria_unanswered.get(row.key, 0)),
            _fmt_stat(row.cohens_kappa),
            _fmt_pct(row.observed_agreement),
            str(row.human_positive),
            str(row.judge_positive),
        )
        for row in report.criteria
    ]
    widths = [
        max(len(header), *(len(row[position]) for row in rows))
        for position, header in enumerate(headers)
    ]

    def line(cells: tuple[str, ...]) -> str:
        # Only the key is a word; every column after it is a number, and numbers
        # that do not line up are numbers nobody reads down.
        return "  " + "  ".join(
            cell.ljust(width) if position == 0 else cell.rjust(width)
            for position, (cell, width) in enumerate(zip(cells, widths, strict=True))
        )

    typer.echo(line(headers))
    for row in rows:
        typer.echo(line(row))
    typer.echo("  An unanswered criterion was excluded, never scored 0.0: absence means the")
    typer.echo("  question did not arise. Read each kappa against the two counts beside it --")
    typer.echo("  a criterion almost every run passes cannot produce a high one.")
    if any(row.n and row.cohens_kappa is None for row in report.criteria):
        # Otherwise the empty kappa reads as a broken statistic rather than the
        # finding it is.
        typer.echo("  A kappa of n/a beside a non-zero n means at least one side gave a single")
        typer.echo("  answer throughout; chance agreement is then total, and the counts beside")
        typer.echo("  it say the rest.")


def _print_rule_disagreements(report: CalibrationReport) -> None:
    """Which artifact the disagreements point at: the rubric, or the judge.

    ``order-status-01`` is the case this block exists for. The judge's rationale
    agreed with both annotators about every fact of the run and the verdicts
    still split, so re-prompting the judge would have moved nothing: the rule
    that turns answers into a verdict was what needed rewriting.
    """
    total = report.disagreements_total
    rule = report.rule_disagreements
    undetermined = report.rule_undetermined
    perception = total - rule - undetermined
    if all(row.n == 0 for row in report.criteria):
        # The same corpus-level guard as the criteria table above, so the two blocks
        # cannot contradict each other on one screen. Gating on the counts instead
        # would also fire when every split is undetermined over a thin overlap --
        # and that undetermined tally is a measured number, which prints below.
        typer.echo("  Rubric disagreements: n/a. No compared trajectory carries criterion")
        typer.echo("  answers from both sides, so none of these can be attributed to the rubric")
        typer.echo("  or to the judge. Label a slice with the criteria filled in to place them.")
        return
    typer.echo(f"  Rubric disagreements: {rule} of {total}.")
    if rule:
        typer.echo("  On those, judge and annotators answered every criterion either side scored")
        typer.echo("  identically and still returned different verdicts, so the difference lives")
        typer.echo("  in the rule that turns answers into a verdict. A large count here says fix")
        typer.echo("  the rubric, not the judge: re-prompting a judge that already reads the")
        typer.echo("  runs as the annotators do moves nothing.")
    if perception:
        typer.echo(f"  Perception differences: {perception} of {total}. The two sides answered")
        typer.echo("  some criterion differently, so they are reading the runs differently --")
        typer.echo("  that is judge work (prompt, examples, model), not a rubric rewrite.")
    if undetermined:
        typer.echo(f"  Undetermined: {undetermined} of {total}. One side answered a criterion")
        typer.echo("  the other left out, or neither answered any; those cannot be placed either")
        typer.echo("  way, and the fix for them is more labeling rather than a rewrite.")


def _print_calibration_disagreements(report: CalibrationReport) -> None:
    # The true count, not the length of a list that may have been truncated.
    total = report.disagreements_total
    if total == 0:
        typer.echo("Disagreements")
        typer.echo("  None: the judge matched the human majority on every compared trajectory.")
        return
    shown = report.disagreements[:_DISAGREEMENT_LIMIT]
    if total > len(shown):
        typer.echo(f"Worst disagreements (showing {len(shown)} of {total}; --json prints all)")
    else:
        typer.echo(f"Worst disagreements ({total})")
    # Above the rows, because it says how many of them are worth reading as judge
    # errors before anyone reads the first rationale as one.
    _print_rule_disagreements(report)
    keys = [_trim(row.task_key, _TASK_KEY_WIDTH) for row in shown]
    stub = max(len(key) for key in keys)
    verdicts = [f"human {row.human_verdict} vs judge {row.judge_verdict}" for row in shown]
    span = max(len(text) for text in verdicts)
    for row, key, text in zip(shown, keys, verdicts, strict=True):
        typer.echo(
            f"  {key:<{stub}}  {row.trajectory_id[:8]}  {text:<{span}}  (distance {row.distance})"
        )
        rationale = _trim(row.judge_rationale, _RATIONALE_WIDTH)
        typer.echo(f"    {rationale}" if rationale else "    (no rationale recorded)")


def _print_calibration_empty_state(report: CalibrationReport) -> None:
    """Explain why there are no statistics, instead of rendering a table of zeros."""
    typer.echo("No agreement statistics yet")
    typer.echo(
        "  Nothing in this scope carries both a human label and a verdict from judge"
        f" '{report.judge_name}',"
    )
    typer.echo("  so kappa, the confusion matrix, and the per-verdict scores are all undefined.")
    typer.echo("  Zeros would read like findings, so they are left out.")

    labeled, judged = report.labeled_trajectories, report.judged_trajectories
    if labeled == 0 and judged == 0:
        typer.echo("  Missing: human labels AND judge verdicts. Neither exists in this scope.")
    elif labeled == 0:
        typer.echo(
            f"  Missing: human labels. {_count(judged, 'trajectory', 'trajectories')} judged,"
            " none labeled."
        )
    elif judged == 0:
        typer.echo(
            f"  Missing: judge verdicts. {_count(labeled, 'trajectory', 'trajectories')} labeled,"
            f" none judged by '{report.judge_name}'."
        )
    else:
        typer.echo(
            f"  Both exist ({labeled} labeled, {judged} judged) but on different trajectories:"
        )
        typer.echo("  label some of the judged runs, or judge some of the labeled ones.")
    if report.ties_excluded:
        typer.echo(
            "  Excluded: "
            f"{_count(report.ties_excluded, 'labeled trajectory', 'labeled trajectories')}"
            " with no majority verdict (annotator ties)."
        )
    if report.judge_errors:
        typer.echo(
            f"  Excluded: {_count(report.judge_errors, 'judge verdict', 'judge verdicts')}"
            " recording a failed call."
        )


def _print_calibration_next_steps(report: CalibrationReport) -> None:
    labeled, judged = report.labeled_trajectories, report.judged_trajectories
    task_filter = f" --task-key {report.task_key}" if report.task_key else ""
    # Show only the half that is actually missing; when both sides exist but do not
    # overlap, either command can close the gap, so both are listed.
    steps: list[str] = []
    if labeled == 0 or judged > 0:
        steps.append("agentverdict serve  (then label trajectories at http://127.0.0.1:8000/label)")
    if judged == 0 or labeled > 0:
        steps.append(f"agentverdict judge run {report.judge_name}{task_filter}")
    typer.echo("Do this next")
    for position, step in enumerate(steps, start=1):
        typer.echo(f"  {position}. {step}")
    if report.eval_run_id:
        typer.echo(
            "  This report is scoped to one eval run; drop --eval-run to compare against every run."
        )
    if report.annotators:
        # A mistyped annotator name looks exactly like an unlabeled dataset, so
        # name the filter that is hiding the labels rather than leaving it implicit.
        typer.echo(
            f"  Only labels from {', '.join(report.annotators)} count here;"
            " drop --annotator to use every annotator's labels."
        )


def _print_calibration_report(report: CalibrationReport) -> None:
    _print_calibration_header(report)
    typer.echo("")
    _print_calibration_coverage(report)
    typer.echo("")
    if report.compared == 0:
        _print_calibration_empty_state(report)
        if report.human_ceiling:
            typer.echo("")
            _print_human_ceiling(report, reading_aid=False)
        typer.echo("")
        _print_calibration_next_steps(report)
        return
    _print_calibration_headline(report.agreement)
    typer.echo("")
    if report.judge_vs_annotator:
        _print_judge_vs_annotator(report)
        typer.echo("")
    _print_confusion_matrix(report.agreement)
    typer.echo("")
    _print_calibration_per_class(report.agreement)
    typer.echo("")
    _print_human_ceiling(report)
    typer.echo("")
    _print_calibration_criteria(report)
    typer.echo("")
    _print_calibration_disagreements(report)


def _print_probe_run_summary(run: BiasProbeRun) -> None:
    """What the sweep cost, before the report says what it found."""
    arms = len((run.meta or {}).get("arms") or ())
    typer.echo(
        f"Probe run {run.id[:8]} {run.status}: {run.trajectory_count} trajectories"
        f" x {arms} arms x {run.repeats} repeats"
    )
    if run.error_count:
        typer.echo(f"  errors      {run.error_count}")
    typer.echo(f"  tokens      {run.input_tokens} in / {run.output_tokens} out")


def _print_probe_header(report: BiasProbeReport) -> None:
    title = _PROBE_TITLES.get(report.probe, report.probe)
    typer.echo(f"Bias probe: judge '{report.judge_name}'  [{title}]")
    if report.repeats:
        typer.echo(f"  perturbs   {_PROBE_SUMMARY.get(report.probe, report.probe)}")
        typer.echo(
            f"  sample     {_count(report.trajectories, 'trajectory', 'trajectories')} over"
            f" {_count(report.tasks, 'task', 'tasks')}, {report.repeats} repeats per arm"
        )
        # The cluster count is never left implicit. Trajectories replayed from one
        # task share the prompt, the expected outcome and the tools block - exactly
        # what the order probe permutes - so the bootstrap resamples tasks, and a
        # reader who takes `trajectories` for the sample size reads every interval
        # as far tighter than it is.
        typer.echo("             intervals resample tasks, so the task count is the n behind them")
        # "Same judge" is not the same claim as "same rubric" once the prompt is
        # editable, and these numbers only describe the rubric that was in force.
        typer.echo(f"  rubric     {report.judge_prompt_version or 'unrecorded'}")
    stamp = report.generated_at
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(UTC)
    typer.echo(f"  generated  {stamp:%Y-%m-%d %H:%M:%S} UTC")


def _print_probe_control(report: BiasProbeReport) -> None:
    """The judge's own test-retest noise, printed before any arm is discussed.

    This is a finding in its own right and not a footnote to the arms: it is the
    answer to "does this judge even give the same answer twice", and it is the
    floor every shift below has to clear before it means anything.
    """
    typer.echo(f"Judge test-retest ({report.repeats} byte-identical calls per trajectory)")
    typer.echo(
        f"  disagreement  {_fmt_stat(report.control_disagreement):>6}"
        f"   {_fmt_interval(report.control_ci_low, report.control_ci_high, signed=False)}"
        f" at {report.adjusted_confidence:.1%}"
    )
    if report.control_disagreement is None:
        typer.echo("  Undefined: no trajectory has two usable identity draws, so this judge's")
        typer.echo("  own noise was never measured and the arms below have no floor to be read")
        typer.echo("  against. Re-run the probe once the identity arm returns verdicts.")
        return
    typer.echo("  How often two identical prompts came back with different verdicts, over the")
    typer.echo("  distinct unordered pairs within the control arm. Read every shift below")
    typer.echo("  against it: movement smaller than the judge's own noise says nothing about")
    typer.echo("  the perturbation.")


def _probe_row(cells: tuple[str, ...], widths: list[int]) -> str:
    """Lay one arm row out: the name and the outcome are words, everything between is a number.

    Numbers that do not line up are numbers nobody scans, so only the two text
    columns are left-aligned.
    """
    rendered = [
        cell.ljust(width) if position in (0, len(cells) - 1) else cell.rjust(width)
        for position, (cell, width) in enumerate(zip(cells, widths, strict=True))
    ]
    return "  " + "  ".join(rendered).rstrip()


def _print_probe_arms(report: BiasProbeReport) -> None:
    """One row per perturbed arm: the shift, its interval, and what powered it."""
    needed = "n/a" if report.min_movers is None else str(report.min_movers)
    headers = (
        "arm",
        "applied",
        "measured",
        "shift",
        "interval",
        "disagree",
        "movers",
        "needed",
        "outcome",
    )
    rows = [
        (
            arm.variant,
            f"{arm.applied_to}/{report.trajectories}",
            f"{arm.measured_on}/{report.trajectories}",
            _fmt_delta(arm.signed_shift),
            _fmt_interval(arm.shift_ci_low, arm.shift_ci_high),
            _fmt_stat(arm.disagreement),
            # An arm that measured nothing is dashed rather than zeroed. "0 moved of 4
            # needed" is a sentence about a judge that held steady, and this arm either
            # never asked it anything or never got an answer back.
            "-" if arm.measured_on == 0 else str(arm.movers_observed),
            "-" if arm.measured_on == 0 else needed,
            arm.outcome,
        )
        for arm in report.arms
    ]
    typer.echo("Arms against the identity control (fail 0.0, borderline 0.5, pass 1.0)")
    if not rows:
        typer.echo("  This probe has no perturbed arms, so there is nothing to compare.")
        return
    widths = [
        max(len(header), *(len(row[position]) for row in rows))
        for position, header in enumerate(headers)
    ]
    typer.echo(_probe_row(headers, widths))
    for row in rows:
        typer.echo(_probe_row(row, widths))
    typer.echo("  applied  trajectories whose rendered prompt actually differed from the control's")
    typer.echo("  measured of those, the ones the judge answered on both sides, so a shift exists;")
    typer.echo("           every column to its right is denominated in this, not in 'applied'")
    typer.echo("  shift    the headline: mean signed movement along the verdict scale")
    typer.echo("  disagree secondary: how often a control draw and an arm draw differ at all")
    typer.echo("  movers   trajectories that moved; 'needed' is how many must move before a")
    typer.echo("           bootstrap at this n can exclude zero, whatever the size of the effect")


def _probe_arm_note(arm: ArmOutcome, report: BiasProbeReport) -> list[str]:
    """The sentences that turn one arm's row into a claim, or into a refusal to make one."""
    needed = report.min_movers
    # "measured" rather than a bare count: the threshold beside it is set by the run's
    # n, so an arm that lost trajectories to failed calls can be unable to reach it
    # however its survivors behaved, and "2 of 2 moved (3 needed)" reads as a
    # contradiction unless the denominator says which two.
    moved = f"{arm.movers_observed} of {arm.measured_on} measured trajectories moved"
    if arm.outcome == "not_applied":
        return [
            f"{arm.variant}: NOT APPLIED. The rendered prompt matched the control for every",
            "  trajectory, so this arm measured nothing at all. That is a broken arm, not a",
            "  judge that shrugged the perturbation off - fix the harness before reading it.",
        ]
    if arm.measured_on == 0:
        # The other way to measure nothing, and it must not borrow the vocabulary of
        # the first: the perturbation was delivered here, so pointing the reader at the
        # renderer would send them to debug the half that worked.
        return [
            f"{arm.variant}: NOT MEASURED. The perturbation reached the model on"
            f" {_count(arm.applied_to, 'trajectory', 'trajectories')},",
            "  but no trajectory came back with a verdict on both sides, so there is no shift",
            "  to report and nothing held still either. Read the failed calls below, not this",
            "  arm - a judge that never answered is not a judge that was unmoved.",
        ]
    if arm.outcome in ("biased", "inverse"):
        direction = "upward" if arm.outcome == "biased" else "downward"
        side = "above" if arm.outcome == "biased" else "below"
        return [
            f"{arm.variant}: SENSITIVE, {direction}. The shift is"
            f" {_fmt_delta(arm.signed_shift)}, the whole interval lies",
            f"  {side} zero, and {moved} ({needed} needed).",
            f"  {arm.description}",
        ]
    if not arm.power_limited:
        return [
            f"{arm.variant}: INCONCLUSIVE. {moved} ({needed} needed),",
            "  but the interval straddles zero: the movement is no larger than resampling",
            "  the tasks produces on its own.",
        ]
    if needed is None:
        return [
            f"{arm.variant}: INCONCLUSIVE, and no interval was possible. Fewer than two",
            "  trajectories were measured, so nothing can be resampled at any effect size.",
            "  Read it as 'not measured'. It is not 'unbiased'.",
        ]
    return [
        f"{arm.variant}: INCONCLUSIVE, and the instrument could not have said otherwise.",
        f"  Only {moved}, and this n needs {needed} before",
        "  any interval can exclude zero - so an effect this small is invisible here",
        "  however real it is. Read it as 'not measured'. It is not 'unbiased'.",
    ]


def _print_probe_arm_notes(report: BiasProbeReport) -> None:
    if not report.arms:
        return
    typer.echo("Reading the arms")
    for arm in report.arms:
        for line in _probe_arm_note(arm, report):
            typer.echo(f"  {line}")
    if report.probe == "length" and any(
        arm.outcome in ("biased", "inverse") for arm in report.arms
    ):
        # Length sensitivity is not assumed to be a defect. The judge's own rubric
        # names "excessive or confusing communication" as a borderline criterion, and
        # on the golden set both annotators independently marked a verbose-but-correct
        # run down for it. Reporting obedience to the written rubric as bias would be
        # the exact error this milestone exists to catch.
        typer.echo("")
        typer.echo("  The rubric licenses some of this. 'Excessive or confusing communication'")
        typer.echo("  is a borderline criterion the judge is told to apply, and both annotators")
        typer.echo("  marked a verbose-but-correct run down for it unprompted. A downward shift")
        typer.echo("  under padding is a defect only where it exceeds what the rubric asks for.")


def _print_probe_errors(report: BiasProbeReport) -> None:
    """Disclose failed calls, because dropping them quietly flatters every arm."""
    typer.echo("Failed calls")
    typer.echo(
        f"  {_count(report.errors, 'call', 'calls')} produced no verdict. Those draws are dropped"
    )
    typer.echo("  rather than counted as agreement, so every figure above rests on fewer draws")
    typer.echo("  than the repeat count implies.")
    if report.probe == "length":
        typer.echo("  They are not spread evenly either: the padded arms carry the longest")
        typer.echo("  prompts, so they lose the most draws exactly where the dose is largest.")


def _print_probe_confidence(report: BiasProbeReport) -> None:
    """Print the multiplicity correction rather than applying it silently."""
    intervals = len(report.arms) + 1  # the perturbed arms, plus the test-retest floor
    typer.echo("Confidence")
    typer.echo(
        f"  {report.confidence:.1%} nominal, every interval above read at"
        f" {report.adjusted_confidence:.1%}"
    )
    typer.echo(
        f"  (Sidak over {intervals} intervals: {_count(len(report.arms), 'arm', 'arms')}"
        " plus the test-retest floor)."
    )
    typer.echo(
        f"  Uncorrected, that many {report.confidence:.0%} intervals give a judge with no"
        " sensitivity at"
    )
    typer.echo("  all a comfortable chance of looking sensitive in one of them.")


def _print_probe_empty_state(report: BiasProbeReport) -> None:
    """Explain that nothing has been measured, instead of rendering zeros."""
    typer.echo("No probe results")
    typer.echo(f"  Judge '{report.judge_name}' has no stored '{report.probe}' probe run, so its")
    typer.echo("  test-retest floor, every shift and every interval are undefined. Zeros here")
    typer.echo("  would read as a judge that passed a test nobody ran, so they are left out.")
    typer.echo("")
    typer.echo("Do this next")
    typer.echo(f"  1. agentverdict probe {report.judge_name} --probe {report.probe}")
    typer.echo("     Unlike calibrate and compare, this one calls the model: every trajectory,")
    typer.echo("     under every arm, once per repeat. Use --limit to try it on a slice first.")


def _print_probe_no_trajectories(report: BiasProbeReport) -> None:
    """A run that had nothing to judge is not a run that found nothing."""
    typer.echo("The probe ran, but there was nothing to judge")
    typer.echo("  No trajectory reached any arm, so every arm is empty and no figure below it")
    typer.echo("  would mean anything. A probe needs no human labels - it compares the judge")
    typer.echo("  with itself - but it does need trajectories.")
    typer.echo("")
    typer.echo("Do this next")
    typer.echo("  1. agentverdict import examples/sample_trajectories.jsonl")
    typer.echo("     or agentverdict replay, to record fresh runs from the stored tasks")
    typer.echo(f"  2. agentverdict probe {report.judge_name} --probe {report.probe}")


def _print_probe_report(report: BiasProbeReport) -> None:
    _print_probe_header(report)
    typer.echo("")
    # `repeats` is zero only when no run was found: the runner refuses to record one
    # below two draws per arm.
    if report.repeats == 0:
        _print_probe_empty_state(report)
        return
    if report.trajectories == 0:
        _print_probe_no_trajectories(report)
        return
    _print_probe_control(report)
    typer.echo("")
    _print_probe_arms(report)
    typer.echo("")
    _print_probe_arm_notes(report)
    if report.errors:
        typer.echo("")
        _print_probe_errors(report)
    typer.echo("")
    _print_probe_confidence(report)


def _print_comparison_header(comparison: RunComparison) -> None:
    typer.echo(
        f"Comparison: base {comparison.base_run_id[:8]}"
        f"  ->  candidate {comparison.candidate_run_id[:8]}"
    )
    # Naming the rubric alongside the judge: "same judge" is not the same claim as
    # "same yardstick" once the prompt is editable, and these numbers only mean
    # something relative to one.
    rubric = comparison.judge_prompt_version or "unrecorded"
    typer.echo(f"  judge      '{comparison.judge_name}'  (rubric {rubric})")
    typer.echo(
        f"  compared   {_count(comparison.tasks_compared, 'task', 'tasks')} present in both runs"
    )
    if comparison.input_tokens or comparison.output_tokens:
        typer.echo(f"  tokens     {comparison.input_tokens} in / {comparison.output_tokens} out")
    stamp = comparison.generated_at
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(UTC)
    typer.echo(f"  generated  {stamp:%Y-%m-%d %H:%M:%S} UTC")


def _print_comparison_scores(comparison: RunComparison) -> None:
    """The two suite scores and the change between them, with its interval."""
    typer.echo("Suite score (mean over shared tasks; fail 0.0, borderline 0.5, pass 1.0)")
    typer.echo(f"  base       {_fmt_stat(comparison.base_score):>7}")
    typer.echo(f"  candidate  {_fmt_stat(comparison.candidate_score):>7}")
    if comparison.tasks_compared >= 2:
        typer.echo(
            f"  delta      {_fmt_delta(comparison.delta):>7}"
            f"   {_fmt_ci(comparison.delta_ci_low, comparison.delta_ci_high)}"
        )
        return
    # A single task cannot be resampled, so there is no interval to show. Saying
    # that here beats printing "CI n/a" and leaving the reader to guess why.
    typer.echo(f"  delta      {_fmt_delta(comparison.delta):>7}   (one task: not resampled)")


def _print_comparison_errors(comparison: RunComparison) -> None:
    """Disclose judge calls that failed, because their absence flatters the score."""
    total = comparison.base_judge_errors + comparison.candidate_judge_errors
    if not total:
        return
    typer.echo(
        f"  judge errors  {comparison.base_judge_errors} in base,"
        f" {comparison.candidate_judge_errors} in candidate - excluded from the scores"
    )
    typer.echo(
        "  A failed call is not a failed run, so it is not scored as one. But the calls"
    )
    typer.echo(
        "  that fail are the long, hard transcripts, so the scores above read high."
    )


def _print_comparison_gate(comparison: RunComparison, *, fail_on_regression: bool) -> None:
    """Say exactly what this result does to a merge, and what the flag changed."""
    if comparison.blocks_merge:
        if fail_on_regression:
            typer.echo("  Gate: FAILED - this command exits 1, which is what blocks the merge.")
        else:
            typer.echo("  Gate: reporting only. Pass --fail-on-regression to make CI fail here.")
        return
    if fail_on_regression:
        typer.echo("  Gate: passed - only a regression exits 1, so nothing is blocked.")
    else:
        typer.echo("  Gate: not enabled. Pass --fail-on-regression to block merges on this.")


def _print_comparison_too_small(comparison: RunComparison, *, fail_on_regression: bool) -> None:
    """Explain the missing verdict instead of printing an interval of "n/a".

    Fewer than two shared tasks is not a quiet pass and not a quiet failure: the
    gate never ran. A reader who sees an empty interval next to a green exit code
    will read it as "no regression found", which is the one conclusion the data
    cannot support.
    """
    typer.echo("Outcome: no gate decision - the shared suite is too small to resample.")
    if comparison.tasks_compared == 0:
        typer.echo("  The two runs have no task in common, so there is nothing to compare.")
    else:
        typer.echo("  Only one task appears in both runs. The gate works by resampling tasks,")
        typer.echo("  and a single task resamples to itself every time, so there is no interval")
        typer.echo("  and no way to separate a real change from noise.")
    typer.echo("  Nothing was blocked and nothing was cleared - this is not a passing gate.")
    typer.echo("  Judge at least two of the same tasks in both runs, then compare again.")
    if fail_on_regression:
        typer.echo("  --fail-on-regression exits 0 here: there is no evidence to block on.")


def _print_comparison_outcome(comparison: RunComparison, *, fail_on_regression: bool) -> None:
    """State the result in words — the numbers above do not interpret themselves.

    ``comparison.summary`` is deliberately not echoed: it restates the scores and
    the interval already printed above, and these lines spend their space on what
    those numbers mean for the merge instead. ``--json`` carries the summary for
    the pull-request comment that wants one sentence.
    """
    if comparison.tasks_compared < 2:
        _print_comparison_too_small(comparison, fail_on_regression=fail_on_regression)
        return

    if comparison.outcome == "regression":
        typer.echo("Outcome: regression - the candidate run scores worse than the baseline.")
        typer.echo("  The whole 95% interval lies below zero, so the drop survives resampling")
        typer.echo("  the tasks: it is larger than this suite's own run-to-run noise.")
    elif comparison.outcome == "improvement":
        typer.echo("Outcome: improvement - the candidate run scores better than the baseline.")
        typer.echo("  The whole 95% interval lies above zero. An improvement never fails a gate.")
    else:
        typer.echo("Outcome: inconclusive - this change is indistinguishable from noise.")
        typer.echo("  The 95% interval straddles zero, so resampling the tasks yields a drop as")
        typer.echo("  readily as a gain: the suite cannot tell these two runs apart.")
    _print_comparison_gate(comparison, fail_on_regression=fail_on_regression)


def _print_task_deltas(comparison: RunComparison) -> None:
    """Per-task movement, worst first — where the suite score actually moved."""
    rows = sorted(comparison.task_deltas, key=lambda row: (row.delta, row.task_key))
    if not rows:
        return
    shown = rows[:_TASK_DELTA_LIMIT]
    if len(rows) > len(shown):
        typer.echo(
            f"Per-task change (worst first, showing {len(shown)} of {len(rows)};"
            " --json prints all)"
        )
    else:
        typer.echo(f"Per-task change (worst first, {_count(len(rows), 'task', 'tasks')})")
    keys = [_trim(row.task_key, _TASK_KEY_WIDTH) for row in shown]
    stub = max(len("task"), *(len(key) for key in keys))
    typer.echo(f"  {'task':<{stub}}  {'base':>6}  {'cand':>6}  {'delta':>7}")
    for row, key in zip(shown, keys, strict=True):
        typer.echo(
            f"  {key:<{stub}}  {_fmt_stat(row.base_score):>6}"
            f"  {_fmt_stat(row.candidate_score):>6}  {_fmt_delta(row.delta):>7}"
        )


def _print_excluded_tasks(comparison: RunComparison) -> None:
    """Name every task that only one run covered.

    Pairing is by task, so a task missing from either side is dropped from the
    statistics entirely. Left unsaid, that turns an edited suite into what looks
    like an agent that changed: the tasks are therefore named, not counted.
    """
    total = len(comparison.base_only_tasks) + len(comparison.candidate_only_tasks)
    typer.echo(
        f"Excluded from the statistics: {_count(total, 'task', 'tasks')} present in one run only"
    )
    for label, names in (
        ("base only", comparison.base_only_tasks),
        ("candidate only", comparison.candidate_only_tasks),
    ):
        if not names:
            continue
        typer.echo(f"  {label} ({len(names)}):")
        for line in textwrap.wrap(
            ", ".join(sorted(names)),
            width=_EXCLUDED_WIDTH,
            initial_indent="    ",
            subsequent_indent="    ",
        ):
            typer.echo(line)
    typer.echo("  A suite that changed shape - rather than an agent that changed behavior - is")
    typer.echo("  the likeliest explanation for a result that reads as surprising.")


def _print_comparison(comparison: RunComparison, *, fail_on_regression: bool) -> None:
    _print_comparison_header(comparison)
    typer.echo("")
    if comparison.tasks_compared:
        _print_comparison_scores(comparison)
        _print_comparison_errors(comparison)
        typer.echo("")
    _print_comparison_outcome(comparison, fail_on_regression=fail_on_regression)
    if comparison.task_deltas:
        typer.echo("")
        _print_task_deltas(comparison)
    if comparison.base_only_tasks or comparison.candidate_only_tasks:
        typer.echo("")
        _print_excluded_tasks(comparison)


def _ensured_rubric(session: Session) -> Rubric:
    """The current rubric row, reporting drift as a fixable message rather than a trace.

    Drift is a deployment fault: the stored row a version already points at describes
    labeling that has been done under it, so the answer is always a version bump and
    never an edit. Checking here is what moves the failure to the command that sets a
    database up instead of to whoever presses save next.
    """
    from agentverdict.rubric import RubricDriftError, ensure_rubric

    try:
        return ensure_rubric(session)
    except RubricDriftError as exc:
        typer.echo(f"Rubric check failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("init-db")
def init_db_cmd() -> None:
    """Create or upgrade the schema and the rubric row (safe to run repeatedly)."""
    # Seeded here rather than on first use: a deployment nobody has labeled yet
    # still has a rulebook to name, and a criteria edit that forgot its version
    # bump fails this command instead of the first annotator to press save --
    # uncaught, the drift error is a 500 in the middle of a labeling session.
    with _open_session() as session:
        rubric = _ensured_rubric(session)
        name, version, criteria_count = rubric.name, rubric.version, len(rubric.criteria)
    typer.echo("Database schema is up to date.")
    typer.echo(f"Rubric: {name} v{version}, {criteria_count} criteria.")


@app.command("import")
def import_cmd(
    path: Annotated[Path, typer.Argument(help="Path to a JSONL trajectory bundle.")],
) -> None:
    """Import a JSONL trajectory bundle and print an import summary."""
    if not path.is_file():
        typer.echo(f"No such file: {path}", err=True)
        raise typer.Exit(code=1)
    with _open_session() as session:
        report = import_jsonl(session, path)
    typer.echo(f"Imported {path}:")
    typer.echo(f"  tasks created:        {report.tasks_created}")
    typer.echo(f"  trajectories created: {report.trajectories_created}")
    typer.echo(f"  steps created:        {report.steps_created}")
    if report.errors:
        typer.echo(f"  lines skipped:        {len(report.errors)}")
        for error in report.errors:
            typer.echo(f"    ! {error}", err=True)


@app.command("export")
def export_cmd(
    path: Annotated[Path, typer.Argument(help="Destination JSONL file.")],
    task_key: Annotated[
        str | None,
        typer.Option("--task-key", help="Only export trajectories for this task key."),
    ] = None,
) -> None:
    """Export trajectories (optionally filtered by task key) to a JSONL bundle."""
    with _open_session() as session:
        count = export_jsonl(session, path, task_key=task_key)
    noun = "trajectory" if count == 1 else "trajectories"
    scope = f" for task '{task_key}'" if task_key else ""
    typer.echo(f"Wrote {count} {noun}{scope} to {path}")


@app.command()
def stats() -> None:
    """Print dataset counts, label coverage, and the verdict histogram."""
    with _open_session() as session:
        report = compute_stats(session)
    if report.trajectory_count:
        coverage = f"{report.labeled_trajectory_count / report.trajectory_count:.0%}"
    else:
        coverage = "n/a"
    typer.echo(f"Tasks:                {report.task_count}")
    typer.echo(f"Trajectories:         {report.trajectory_count}")
    typer.echo(f"Labeled trajectories: {report.labeled_trajectory_count} ({coverage} coverage)")
    typer.echo(f"Labels:               {report.label_count}")
    if report.labels_by_verdict:
        typer.echo("Labels by verdict:")
        for verdict, count in sorted(report.labels_by_verdict.items()):
            typer.echo(f"  {verdict:<11} {count}")


@judge_app.command("add")
def judge_add(
    name: Annotated[str, typer.Argument(help="Unique judge name, e.g. groq-70b.")],
    model: Annotated[
        str | None, typer.Option(help="Model id (default: the configured judge model).")
    ] = None,
    description: Annotated[str | None, typer.Option(help="Free-text description.")] = None,
) -> None:
    """Register a judge (a model plus prompting recipe)."""
    from agentverdict.config import get_settings

    with _open_session() as session:
        existing = session.scalar(select(Judge).where(Judge.name == name))
        if existing is not None:
            typer.echo(f"Judge '{name}' already exists (model {existing.model}).", err=True)
            raise typer.Exit(code=1)
        judge = Judge(name=name, model=model or get_settings().judge_model, description=description)
        session.add(judge)
    typer.echo(f"Registered judge '{name}' using model {judge.model}.")


@judge_app.command("list")
def judge_list() -> None:
    """List registered judges with their verdict counts."""
    with _open_session() as session:
        rows = session.execute(
            select(Judge, func.count(JudgeVerdict.id))
            .outerjoin(JudgeVerdict, JudgeVerdict.judge_id == Judge.id)
            .group_by(Judge.id)
            .order_by(Judge.name)
        ).all()
        if not rows:
            typer.echo("No judges registered. Add one with: agentverdict judge add NAME")
            return
        for judge, verdict_count in rows:
            suffix = f" — {judge.description}" if judge.description else ""
            typer.echo(f"{judge.name}  [{judge.model}]  verdicts: {verdict_count}{suffix}")


@judge_app.command("run")
def judge_run(
    name: Annotated[str, typer.Argument(help="Name of a registered judge.")],
    task_key: Annotated[
        str | None, typer.Option("--task-key", help="Only judge trajectories of this task.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option(help="Judge at most this many trajectories.", min=1)
    ] = None,
) -> None:
    """Judge stored trajectories with an LLM and print a scored summary."""
    from agentverdict.judging.client import JudgeClientError
    from agentverdict.judging.runner import run_eval

    try:
        with _open_session() as session:
            judge = _require_judge(session, name)
            run = run_eval(
                session, judge, task_key=task_key, limit=limit, on_progress=_judge_progress
            )
    except JudgeClientError as exc:
        typer.echo(f"Judge run failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _print_eval_summary(run)


@app.command()
def replay(
    adapter: Annotated[
        str, typer.Option("--adapter", help=f"Agent under test ({', '.join(ADAPTERS)}).")
    ] = "groq-agent",
    task_key: Annotated[
        str | None, typer.Option("--task-key", help="Only replay this task key.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option(help="Replay at most this many tasks.", min=1)
    ] = None,
    repeats: Annotated[int, typer.Option(help="Attempts per task.", min=1)] = 1,
    model: Annotated[
        str | None, typer.Option(help="Model id for the agent (default: the configured model).")
    ] = None,
) -> None:
    """Run an agent against stored tasks and record the trajectories it produces."""
    from agentverdict.agents.replay import run_replay
    from agentverdict.judging.client import JudgeClientError

    try:
        agent = _build_adapter(adapter, model)
        typer.echo(f"Replaying tasks with '{agent.name}'...")
        with _open_session() as session:
            report = run_replay(
                session,
                agent,
                task_key=task_key,
                limit=limit,
                repeats=repeats,
                on_progress=_replay_progress,
            )
    except JudgeClientError as exc:
        typer.echo(f"Replay failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _print_replay_summary(report)


@app.command("eval")
def eval_cmd(
    judge_name: Annotated[str, typer.Option("--judge", help="Name of a registered judge.")],
    adapter: Annotated[
        str, typer.Option("--adapter", help=f"Agent under test ({', '.join(ADAPTERS)}).")
    ] = "groq-agent",
    task_key: Annotated[
        str | None, typer.Option("--task-key", help="Only use this task key.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option(help="Cap tasks replayed / rows judged.", min=1)
    ] = None,
    repeats: Annotated[int, typer.Option(help="Attempts per task.", min=1)] = 1,
    model: Annotated[
        str | None, typer.Option(help="Model id for the agent (default: the configured model).")
    ] = None,
    skip_replay: Annotated[
        bool,
        typer.Option("--skip-replay", help="Judge existing trajectories instead of replaying."),
    ] = False,
    run_id_file: Annotated[
        Path | None,
        typer.Option("--run-id-file", help="Write the new eval run's id to this file."),
    ] = None,
) -> None:
    """Replay tasks and judge exactly those runs: the end-to-end evaluation.

    ``--run-id-file`` exists for automation. A caller that needs to act on the run it
    just created -- a CI gate comparing it against a baseline, say -- would otherwise
    have to go looking for "the newest run for this judge", which is a different run
    the moment two jobs share a database. The file is written only when a run is
    actually recorded, so its absence is a reliable signal that there is nothing to
    compare rather than a stale id from an earlier attempt.
    """
    from agentverdict.agents.replay import run_replay
    from agentverdict.judging.client import JudgeClientError
    from agentverdict.judging.runner import run_eval

    # Clear any previous id up front. A caller reads this file to learn what this
    # invocation produced; leaving yesterday's id in place when today's eval dies would
    # hand it a run that has nothing to do with the code under test.
    if run_id_file is not None:
        run_id_file.unlink(missing_ok=True)

    try:
        with _open_session() as session:
            # Resolve the judge before spending any tokens on replay.
            judge = _require_judge(session, judge_name)

            if skip_replay:
                typer.echo(f"Judging stored trajectories with judge '{judge.name}'...")
                run = run_eval(
                    session, judge, task_key=task_key, limit=limit, on_progress=_judge_progress
                )
            else:
                agent = _build_adapter(adapter, model)
                typer.echo(f"Replaying tasks with '{agent.name}'...")
                report = run_replay(
                    session,
                    agent,
                    task_key=task_key,
                    limit=limit,
                    repeats=repeats,
                    on_progress=_replay_progress,
                )
                # Commit the fresh trajectories so a later judging failure cannot
                # discard runs that already cost real model calls.
                session.commit()
                _print_replay_summary(report)
                if not report.trajectory_ids:
                    typer.echo("No trajectories were created, so there is nothing to judge.")
                    return
                fresh = len(report.trajectory_ids)
                typer.echo(
                    f"Judging {fresh} new {'trajectory' if fresh == 1 else 'trajectories'}"
                    f" with judge '{judge.name}'..."
                )
                # trajectory_ids pins the judged set to exactly what was just replayed;
                # task_key rides along only to record the filter on the eval run. `limit`
                # already bounded the replay, so re-applying it here would judge fewer
                # trajectories than were created.
                run = run_eval(
                    session,
                    judge,
                    task_key=task_key,
                    trajectory_ids=report.trajectory_ids,
                    on_progress=_judge_progress,
                )
    except JudgeClientError as exc:
        typer.echo(f"Eval failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if run_id_file is not None:
        run_id_file.write_text(f"{run.id}\n", encoding="utf-8")

    _print_eval_summary(run)


@app.command()
def calibrate(
    name: Annotated[str, typer.Argument(help="Name of a registered judge.")],
    eval_run: Annotated[
        str | None,
        typer.Option("--eval-run", help="Only use verdicts from this eval run id."),
    ] = None,
    task_key: Annotated[
        str | None, typer.Option("--task-key", help="Only compare trajectories of this task.")
    ] = None,
    annotator: Annotated[
        list[str] | None,
        typer.Option(
            "--annotator",
            help="Only count labels from this annotator; repeat to name a set (default: all).",
        ),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the full report as JSON and nothing else.")
    ] = False,
) -> None:
    """Score a judge against the human-labeled golden set (reads stored rows; no model calls)."""
    from agentverdict.calibration.report import build_calibration_report

    with _open_session() as session:
        judge = _require_judge(session, name)
        if eval_run is not None:
            _require_eval_run(session, judge, eval_run)
        try:
            report = build_calibration_report(
                session,
                judge,
                eval_run_id=eval_run,
                task_key=task_key,
                # `or None`: older Typer releases hand a repeatable option an empty
                # list rather than None when it is absent, and "no names given"
                # means every annotator, not nobody.
                annotators=annotator or None,
                # Uncapped: the text view trims for readability but reports the
                # true total, and --json is documented to carry every row.
                max_disagreements=sys.maxsize,
            )
        except (LookupError, ValueError) as exc:
            typer.echo(f"Could not build the calibration report: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(report.model_dump_json(indent=2))
        return
    _print_calibration_report(report)


@app.command()
def probe(
    name: Annotated[str, typer.Argument(help="Name of a registered judge.")],
    probe_name: Annotated[
        str, typer.Option("--probe", help="Which sensitivity to measure: order or length.")
    ] = "order",
    limit: Annotated[
        int | None, typer.Option(help="Probe at most this many trajectories.", min=1)
    ] = None,
    repeats: Annotated[
        int, typer.Option(help="Draws per arm. Must be at least 2 (see the error).")
    ] = 3,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the full report as JSON and nothing else.")
    ] = False,
) -> None:
    """Measure where a judge is systematically wrong by re-asking it under perturbed prompts.

    This command calls the model, which is why it is separate from calibrate and
    never runs inside the merge gate: it makes trajectories x arms x repeats calls.
    Nothing it produces enters judge_verdicts -- a verdict on a deliberately
    reordered or padded transcript is not a verdict on the run -- so calibration and
    the gate are unaffected by anything measured here.

    The exit code stays zero whatever the probe finds. A detected sensitivity is a
    measurement to act on, not a gate: blocking a merge on it would make probing a
    judge more expensive than leaving it unexamined.
    """
    from agentverdict.bias.runner import (
        MIN_REPEATS,
        PROBES,
        build_bias_probe_report,
        run_probe,
    )
    from agentverdict.judging.client import JudgeClientError

    # Both checks run before the database is opened, so a typo costs nothing and a
    # run that cannot be interpreted is never started.
    if probe_name not in PROBES:
        typer.echo(
            f"Unknown probe '{probe_name}'. Valid probes: {', '.join(PROBES)}",
            err=True,
        )
        raise typer.Exit(code=1)
    if repeats < MIN_REPEATS:
        typer.echo(
            f"--repeats must be at least {MIN_REPEATS}, got {repeats}. At one draw per arm the"
            " identity arm has no within-arm pair, so the judge's own noise floor is undefined"
            " and every shift would be measured against nothing.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        with _open_session() as session:
            # Resolved before the announcement, so a mistyped judge name never reads
            # as a sweep that started and then failed.
            judge = _require_judge(session, name)
            if not as_json:
                title = _PROBE_TITLES.get(probe_name, probe_name)
                typer.echo(f"Probing judge '{judge.name}' for {title}...")
            run = run_probe(
                session,
                judge,
                probe_name,
                repeats=repeats,
                limit=limit,
                # Silent under --json, which is documented to print the report and
                # nothing else.
                on_progress=None if as_json else _probe_progress,
            )
            # Report the run just made rather than "the newest run for this judge",
            # which is a different run the moment two jobs share a database.
            report = build_bias_probe_report(session, judge, probe=probe_name, probe_run_id=run.id)
    except JudgeClientError as exc:
        typer.echo(f"Probe failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(report.model_dump_json(indent=2))
        return
    typer.echo("")
    _print_probe_run_summary(run)
    typer.echo("")
    _print_probe_report(report)


@app.command()
def compare(
    base_run_id: Annotated[str, typer.Argument(help="Eval run id of the stored baseline.")],
    candidate_run_id: Annotated[str, typer.Argument(help="Eval run id of the new run.")],
    fail_on_regression: Annotated[
        bool,
        typer.Option(
            "--fail-on-regression",
            help="Exit 1 on a blocking regression — the bit CI keys on.",
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the comparison as JSON and nothing else.")
    ] = False,
) -> None:
    """Compare a candidate eval run against a baseline and decide whether it regressed."""
    from agentverdict.gating.compare import compare_runs

    with _open_session() as session:
        base = _require_run(session, base_run_id, "base")
        candidate = _require_run(session, candidate_run_id, "candidate")
        try:
            comparison = compare_runs(session, base, candidate)
        except ValueError as exc:
            # Comparing runs scored by different judges measures the judges, not
            # the agent. That refusal is a message worth reading, not a traceback.
            typer.echo(f"Cannot compare these runs: {exc}", err=True)
            raise typer.Exit(code=EXIT_CANNOT_COMPARE) from exc

    if as_json:
        typer.echo(comparison.model_dump_json(indent=2))
    else:
        _print_comparison(comparison, fail_on_regression=fail_on_regression)

    # `blocks_merge` is the single bit the gate acts on, and it is set only for a
    # regression — so an inconclusive or improved run can never fail CI, whatever
    # direction the raw means happened to move.
    if fail_on_regression and comparison.blocks_merge:
        raise typer.Exit(code=EXIT_REGRESSION)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Reload on code changes.")] = False,
) -> None:
    """Run the API and labeling UI with uvicorn."""
    import uvicorn

    # The same tripwire init-db runs, because nothing guarantees init-db ran with this
    # build: a drifted rubric caught here fails the deploy, not the first label
    # submission after it. `_open_session` upgrades the schema on the way in.
    with _open_session() as session:
        rubric = _ensured_rubric(session)
        name, version = rubric.name, rubric.version
    typer.echo(f"Rubric {name} v{version} is in force.")
    uvicorn.run("agentverdict.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
