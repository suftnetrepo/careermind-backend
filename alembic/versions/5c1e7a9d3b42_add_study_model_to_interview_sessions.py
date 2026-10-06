"""add study_model to interview_sessions

Revision ID: 5c1e7a9d3b42
Revises: 27dda746261d
Create Date: 2026-10-06 09:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5c1e7a9d3b42'
down_revision: Union[str, None] = '27dda746261d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('interview_sessions', sa.Column('study_model', sa.String(), nullable=True))
    # Everything generated before this column existed was written by gpt-4o
    op.execute("UPDATE interview_sessions SET study_model = 'gpt-4o' WHERE quiz_json IS NOT NULL")


def downgrade() -> None:
    op.drop_column('interview_sessions', 'study_model')
