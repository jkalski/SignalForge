"""add strategy to signals

Revision ID: a3f9d2e1b4c7
Revises: 1eda18146b78
Create Date: 2026-03-04 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a3f9d2e1b4c7'
down_revision: Union[str, Sequence[str], None] = '1eda18146b78'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('signals', sa.Column('strategy', sa.String(length=40), nullable=True))


def downgrade() -> None:
    op.drop_column('signals', 'strategy')
