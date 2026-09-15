"""M0 journals and state observations"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_m0"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        "journal_entries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("entry_type", sa.String(length=20), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "state_observations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("journal_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("journal_entries.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("energy", sa.Float(), nullable=False),
        sa.Column("stress", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_journal_entries_created_at", "journal_entries", ["created_at"])

def downgrade():
    op.drop_index("ix_journal_entries_created_at", table_name="journal_entries")
    op.drop_table("state_observations")
    op.drop_table("journal_entries")
