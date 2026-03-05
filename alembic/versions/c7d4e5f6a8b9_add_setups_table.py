"""add setups table

Revision ID: c7d4e5f6a8b9
Revises: b8e1f3a2d9c5
Create Date: 2026-03-04 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c7d4e5f6a8b9'
down_revision: Union[str, Sequence[str], None] = 'b8e1f3a2d9c5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'setups',
        sa.Column('id',            sa.String(length=160), nullable=False),
        sa.Column('run_id',        sa.Integer(),          nullable=False),
        sa.Column('symbol',        sa.String(length=20),  nullable=False),
        sa.Column('timeframe',     sa.String(length=5),   nullable=False),
        sa.Column('source',        sa.String(length=20),  nullable=True),
        sa.Column('ts',            sa.DateTime(),         nullable=False),
        sa.Column('close',         sa.Numeric(12, 4),     nullable=True),
        sa.Column('ema_20',        sa.Numeric(12, 4),     nullable=True),
        sa.Column('ema_50',        sa.Numeric(12, 4),     nullable=True),
        sa.Column('rsi_14',        sa.Numeric(6, 2),      nullable=True),
        sa.Column('atr_14',        sa.Numeric(12, 4),     nullable=True),
        sa.Column('score_raw',     sa.Numeric(12, 4),     nullable=True),
        sa.Column('score',         sa.Numeric(12, 6),     nullable=True),
        sa.Column('distance_pct',  sa.Numeric(8, 4),      nullable=True),
        sa.Column('current_state', sa.String(length=10),  nullable=True),
        sa.Column('trigger_type',  sa.String(length=10),  nullable=True),
        sa.Column('created_at',    sa.DateTime(),         nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_setups_run_id', 'setups', ['run_id'])
    op.create_index('idx_setups_symbol', 'setups', ['symbol'])
    op.create_index('idx_setups_ts',     'setups', ['ts'])
    op.create_index('idx_setups_score',  'setups', ['score'])


def downgrade() -> None:
    op.drop_index('idx_setups_score',  table_name='setups')
    op.drop_index('idx_setups_ts',     table_name='setups')
    op.drop_index('idx_setups_symbol', table_name='setups')
    op.drop_index('idx_setups_run_id', table_name='setups')
    op.drop_table('setups')
