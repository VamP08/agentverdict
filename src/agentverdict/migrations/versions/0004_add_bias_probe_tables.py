"""add bias_probe_runs and bias_probe_results

Revision ID: 0004_add_bias_probe_tables
Revises: 0003_add_eval_run_prompt_version
Create Date: 2026-08-12

The bias probes re-ask one judge about the same trajectories under prompts that
have been reordered or padded on purpose, and record what comes back. Those
answers get tables of their own rather than a flag column on ``judge_verdicts``:
those rows feed calibration and the regression gate, and a verdict on a
deliberately perturbed transcript is not a verdict on the run. A flag would leave
every existing query one forgotten ``WHERE`` clause away from grading a release,
or reporting a judge's agreement with humans, on padded transcripts — and would
leave the columns the probes actually need (``variant``, ``repeat``,
``prompt_sha``) null for every row that matters.

``prompt_sha`` is stored per result because it is the delivery check: an arm whose
rendered prompt hashes to the identity arm's never applied its perturbation, and
has to be reported as such rather than as an absence of bias.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004_add_bias_probe_tables"
down_revision: str | Sequence[str] | None = "0003_add_eval_run_prompt_version"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _existing_tables() -> set[str]:
    """Tables already present on the target database.

    A database predating Alembic is adopted by creating whatever tables it lacks with
    ``create_all`` and then stamping the baseline revision (see ``migrate.py``), so a
    revision that introduces tables can find them already there — and creating them a
    second time would abort the upgrade on exactly the installs adoption exists to
    rescue. Column revisions like 0002 and 0003 never had to think about this.
    """
    return set(sa.inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    """Apply this revision."""
    existing = _existing_tables()

    # New tables, so no batch mode is needed: SQLite only struggles with ALTER.
    if "bias_probe_runs" not in existing:
        _create_bias_probe_runs()
    if "bias_probe_results" not in existing:
        _create_bias_probe_results()


def _create_bias_probe_runs() -> None:
    op.create_table(
        "bias_probe_runs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("judge_id", sa.String(length=32), nullable=False),
        sa.Column("probe", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("repeats", sa.Integer(), nullable=False),
        sa.Column("trajectory_count", sa.Integer(), nullable=False),
        sa.Column("error_count", sa.Integer(), nullable=False),
        sa.Column("judge_prompt_version", sa.String(length=32), nullable=True),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["judge_id"], ["judges.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_bias_probe_runs_judge_id"), "bias_probe_runs", ["judge_id"], unique=False
    )


def _create_bias_probe_results() -> None:
    op.create_table(
        "bias_probe_results",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("probe_run_id", sa.String(length=32), nullable=False),
        sa.Column("trajectory_id", sa.String(length=32), nullable=False),
        sa.Column("variant", sa.String(length=32), nullable=False),
        sa.Column("repeat", sa.Integer(), nullable=False),
        sa.Column("verdict", sa.String(length=16), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("prompt_sha", sa.String(length=64), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["probe_run_id"], ["bias_probe_runs.id"]),
        sa.ForeignKeyConstraint(["trajectory_id"], ["trajectories.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_bias_probe_results_probe_run_id"),
        "bias_probe_results",
        ["probe_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_bias_probe_results_trajectory_id"),
        "bias_probe_results",
        ["trajectory_id"],
        unique=False,
    )


def downgrade() -> None:
    """Revert this revision."""
    # Results first: they carry the foreign key into the runs table.
    op.drop_index(op.f("ix_bias_probe_results_trajectory_id"), table_name="bias_probe_results")
    op.drop_index(op.f("ix_bias_probe_results_probe_run_id"), table_name="bias_probe_results")
    op.drop_table("bias_probe_results")

    op.drop_index(op.f("ix_bias_probe_runs_judge_id"), table_name="bias_probe_runs")
    op.drop_table("bias_probe_runs")
