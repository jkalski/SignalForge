"""add setup details columns (merge heads)

Merges two prior branch heads:
  - e6a3b2c1d4f5  (add agent_run metrics columns)
  - c7d4e5f6a8b9  (add setups table)

Then adds institutional-lite columns to the setups table.

Revision ID: f1a2b3c4d5e6
Revises: e6a3b2c1d4f5, c7d4e5f6a8b9
Create Date: 2026-03-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, Sequence[str], None] = ('e6a3b2c1d4f5', 'c7d4e5f6a8b9')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('setups', sa.Column('setup_type',       sa.String(length=20),  nullable=True))
    op.add_column('setups', sa.Column('confluence_score', sa.Numeric(6, 2),      nullable=True))
    op.add_column('setups', sa.Column('vol_spike',        sa.Boolean(),          nullable=True))
    op.add_column('setups', sa.Column('htf_aligned',      sa.Boolean(),          nullable=True))
    op.add_column('setups', sa.Column('signal_status',    sa.String(length=10),  nullable=True))
    op.add_column('setups', sa.Column('details',          sa.Text(),             nullable=True))
    op.create_index('idx_setups_signal_status', 'setups', ['signal_status'])


def downgrade() -> None:
    op.drop_index('idx_setups_signal_status', table_name='setups')
    op.drop_column('setups', 'details')
    op.drop_column('setups', 'signal_status')
    op.drop_column('setups', 'htf_aligned')
    op.drop_column('setups', 'vol_spike')
    op.drop_column('setups', 'confluence_score')
    op.drop_column('setups', 'setup_type')
