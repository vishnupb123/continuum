import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class JournalFeatureSet(Base):
    __tablename__ = "journal_feature_sets"

    __table_args__ = (
        UniqueConstraint(
            "journal_id",
            "pipeline_version",
            "source_hash",
            name="uq_journal_feature_generation",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    journal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "journal_entries.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    pipeline_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    source_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default="PENDING",
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    journal = relationship(
        "JournalEntry",
        back_populates="feature_sets",
    )

    text_feature = relationship(
        "TextFeature",
        back_populates="feature_set",
        uselist=False,
        cascade="all, delete-orphan",
    )

    audio_feature = relationship(
        "AudioFeature",
        back_populates="feature_set",
        uselist=False,
        cascade="all, delete-orphan",
    )