"""initial schema: tasks, trajectories, labels, judges, eval runs, verdicts

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-08-07

The schema as it stood before migrations existed: everything M1 and M2 part 1
created through ``Base.metadata.create_all``. Columns added afterwards get their
own revisions, so a database built by an earlier release can be stamped with this
baseline and then upgraded forward.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision."""
    op.create_table(
        "rubrics",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("criteria", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "version", name="uq_rubric_name_version"),
    )

    op.create_table(
        "tasks",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("key", sa.String(length=200), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("tools_spec", sa.JSON(), nullable=False),
        sa.Column("expected_outcome", sa.Text(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_tasks_key"), "tasks", ["key"], unique=True)

    op.create_table(
        "trajectories",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("task_id", sa.String(length=32), nullable=False),
        sa.Column("agent_config", sa.JSON(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_trajectories_task_id"), "trajectories", ["task_id"], unique=False)

    op.create_table(
        "trajectory_steps",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("trajectory_id", sa.String(length=32), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("content", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["trajectory_id"], ["trajectories.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("trajectory_id", "index", name="uq_step_trajectory_index"),
    )
    op.create_index(
        op.f("ix_trajectory_steps_trajectory_id"),
        "trajectory_steps",
        ["trajectory_id"],
        unique=False,
    )

    op.create_table(
        "human_labels",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("trajectory_id", sa.String(length=32), nullable=False),
        sa.Column("annotator", sa.String(length=100), nullable=False),
        sa.Column("verdict", sa.String(length=16), nullable=False),
        sa.Column("rubric_id", sa.String(length=32), nullable=True),
        sa.Column("rubric_scores", sa.JSON(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("time_spent_s", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["rubric_id"], ["rubrics.id"]),
        sa.ForeignKeyConstraint(["trajectory_id"], ["trajectories.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("trajectory_id", "annotator", name="uq_label_trajectory_annotator"),
    )
    op.create_index(
        op.f("ix_human_labels_trajectory_id"), "human_labels", ["trajectory_id"], unique=False
    )

    op.create_table(
        "judges",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_judges_name"), "judges", ["name"], unique=True)

    op.create_table(
        "eval_runs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("judge_id", sa.String(length=32), nullable=False),
        sa.Column("task_key", sa.String(length=200), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("trajectory_count", sa.Integer(), nullable=False),
        sa.Column("error_count", sa.Integer(), nullable=False),
        sa.Column("verdict_counts", sa.JSON(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["judge_id"], ["judges.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_eval_runs_judge_id"), "eval_runs", ["judge_id"], unique=False)

    op.create_table(
        "judge_verdicts",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("eval_run_id", sa.String(length=32), nullable=False),
        sa.Column("judge_id", sa.String(length=32), nullable=False),
        sa.Column("trajectory_id", sa.String(length=32), nullable=False),
        sa.Column("verdict", sa.String(length=16), nullable=True),
        sa.Column("rubric_scores", sa.JSON(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("raw_response", sa.JSON(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["eval_run_id"], ["eval_runs.id"]),
        sa.ForeignKeyConstraint(["judge_id"], ["judges.id"]),
        sa.ForeignKeyConstraint(["trajectory_id"], ["trajectories.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_judge_verdicts_eval_run_id"), "judge_verdicts", ["eval_run_id"], unique=False
    )
    op.create_index(
        op.f("ix_judge_verdicts_judge_id"), "judge_verdicts", ["judge_id"], unique=False
    )
    op.create_index(
        op.f("ix_judge_verdicts_trajectory_id"), "judge_verdicts", ["trajectory_id"], unique=False
    )


def downgrade() -> None:
    """Revert this revision."""
    op.drop_index(op.f("ix_judge_verdicts_trajectory_id"), table_name="judge_verdicts")
    op.drop_index(op.f("ix_judge_verdicts_judge_id"), table_name="judge_verdicts")
    op.drop_index(op.f("ix_judge_verdicts_eval_run_id"), table_name="judge_verdicts")
    op.drop_table("judge_verdicts")

    op.drop_index(op.f("ix_eval_runs_judge_id"), table_name="eval_runs")
    op.drop_table("eval_runs")

    op.drop_index(op.f("ix_judges_name"), table_name="judges")
    op.drop_table("judges")

    op.drop_index(op.f("ix_human_labels_trajectory_id"), table_name="human_labels")
    op.drop_table("human_labels")

    op.drop_index(op.f("ix_trajectory_steps_trajectory_id"), table_name="trajectory_steps")
    op.drop_table("trajectory_steps")

    op.drop_index(op.f("ix_trajectories_task_id"), table_name="trajectories")
    op.drop_table("trajectories")

    op.drop_index(op.f("ix_tasks_key"), table_name="tasks")
    op.drop_table("tasks")

    op.drop_table("rubrics")
