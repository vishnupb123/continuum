"""add audio embedding vector

Revision ID: d90f11ac06db
Revises: 50cf4d439e6f
Create Date: 2026-09-26 19:08:11.178494
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import VECTOR


revision: str = "d90f11ac06db"
down_revision: Union[str, None] = "50cf4d439e6f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "audio_features",
        sa.Column(
            "encoder_revision",
            sa.String(length=255),
            nullable=True,
        ),
    )

    op.add_column(
        "audio_features",
        sa.Column(
            "embedding",
            VECTOR(768),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column(
        "audio_features",
        "embedding",
    )

    op.drop_column(
        "audio_features",
        "encoder_revision",
    )