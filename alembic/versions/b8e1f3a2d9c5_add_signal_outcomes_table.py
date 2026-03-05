"""add signal_outcomes table

Revision ID: b8e1f3a2d9c5
Revises: a3f9d2e1b4c7
Create Date: 2026-03-04 13:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b8e1f3a2d9c5'
down_revision: Union[str, Sequence[str], None] = 'a3f9d2e1b4c7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'signal_outcomes',
        sa.Column('id',                sa.String(length=120), nullable=False),
        sa.Column('signal_id',         sa.String(length=80),  nullable=False),
        sa.Column('symbol',            sa.String(length=20),  nullable=False),
        sa.Column('timeframe',         sa.String(length=5),   nullable=False),
        sa.Column('source',            sa.String(length=20),  nullable=True),
        sa.Column('signal_ts',         sa.DateTime(),         nullable=False),
        sa.Column('horizon_bars',      sa.Integer(),          nullable=False),
        sa.Column('entry_price',       sa.Numeric(12, 4),     nullable=True),
        sa.Column('exit_ts',           sa.DateTime(),         nullable=True),
        sa.Column('exit_price',        sa.Numeric(12, 4),     nullable=True),
        sa.Column('return_pct',        sa.Numeric(10, 6),     nullable=True),
        sa.Column('max_favorable_pct', sa.Numeric(10, 6),     nullable=True),
        sa.Column('max_adverse_pct',   sa.Numeric(10, 6),     nullable=True),
        sa.Column('created_at',        sa.DateTime(),         nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_signal_outcomes_signal_id',       'signal_outcomes', ['signal_id'])
    op.create_index('idx_signal_outcomes_symbol_tf_ts',    'signal_outcomes', ['symbol', 'timeframe', 'signal_ts'])


def downgrade() -> None:
    op.drop_index('idx_signal_outcomes_symbol_tf_ts', table_name='signal_outcomes')
    op.drop_index('idx_signal_outcomes_signal_id',    table_name='signal_outcomes')
    op.drop_table('signal_outcomes')
