import uuid
from datetime import datetime, timezone

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class JournalAudio(Base):
    __tablename__ = "journal_audio"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    journal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("journal_entries.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    storage_key: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        unique=True,
    )

    original_filename: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    mime_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    duration_seconds: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    transcription_status: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default="TRANSCRIPTION_QUEUED",
    )

    transcription_error: Mapped[str | None] = mapped_column(
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

    journal = relationship(
        "JournalEntry",
        back_populates="audio",
    )