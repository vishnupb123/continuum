"""add text embedding vector

Revision ID: 261abac430eb
Revises: 5783d8d9c3a0
Create Date: 2026-09-26 15:36:48.710090
"""

from typing import Sequence, Union

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa


revision: str = "261abac430eb"
down_revision: Union[str, None] = "5783d8d9c3a0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column(
        "text_features",
        sa.Column(
            "embedding",
            Vector(dim=768),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column(
        "text_features",
        "embedding",
    )