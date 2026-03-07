"""add agent_run metrics columns

Revision ID: e6a3b2c1d4f5
Revises: 1eda18146b78
Create Date: 2026-03-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e6a3b2c1d4f5'
down_revision: Union[str, Sequence[str], None] = '1eda18146b78'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('agent_runs', sa.Column('zones_htf_count',       sa.Integer(), nullable=True))
    op.add_column('agent_runs', sa.Column('zones_ltf_count',       sa.Integer(), nullable=True))
    op.add_column('agent_runs', sa.Column('events_detected_count', sa.Integer(), nullable=True))
    op.add_column('agent_runs', sa.Column('valid_setups_count',    sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('agent_runs', 'valid_setups_count')
    op.drop_column('agent_runs', 'events_detected_count')
    op.drop_column('agent_runs', 'zones_ltf_count')
    op.drop_column('agent_runs', 'zones_htf_count')
