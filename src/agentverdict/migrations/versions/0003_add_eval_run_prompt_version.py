"""add eval_runs.judge_prompt_version

Revision ID: 0003_add_eval_run_prompt_version
Revises: 0002_add_task_user_message
Create Date: 2026-08-10

The regression gate compares two eval runs and blames the difference on the code
under review. That only holds if the grader was the same one both times, so each
run records a fingerprint of the judge prompt it was scored with.

Nullable, with no backfill: a run recorded before this column existed was graded
by whatever ``prompts.py`` said at the time, which is not recoverable now. Stamping
those rows with today's fingerprint would make the gate confidently compare runs
that were never comparable.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_add_eval_run_prompt_version"
down_revision: str | Sequence[str] | None = "0002_add_task_user_message"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision."""
    # Batch mode so SQLite (which cannot ALTER arbitrary columns) rebuilds the table.
    with op.batch_alter_table("eval_runs") as batch_op:
        batch_op.add_column(sa.Column("judge_prompt_version", sa.String(length=32), nullable=True))


def downgrade() -> None:
    """Revert this revision."""
    with op.batch_alter_table("eval_runs") as batch_op:
        batch_op.drop_column("judge_prompt_version")
