"""add tasks.user_message

Revision ID: 0002_add_task_user_message
Revises: 0001_initial_schema
Create Date: 2026-08-07

``tasks.prompt`` describes the scenario for graders and states what the agent
*should* do, so replaying against it leaks the answer to the agent under test.
Tasks therefore carry a first-person opening line the agent sees instead.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_add_task_user_message"
down_revision: str | Sequence[str] | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision."""
    # Batch mode so SQLite (which cannot ALTER arbitrary columns) rebuilds the table.
    with op.batch_alter_table("tasks") as batch_op:
        batch_op.add_column(sa.Column("user_message", sa.Text(), nullable=True))


def downgrade() -> None:
    """Revert this revision."""
    with op.batch_alter_table("tasks") as batch_op:
        batch_op.drop_column("user_message")
