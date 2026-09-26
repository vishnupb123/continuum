import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


AUDIO_EMBEDDING_DIMENSION = 768


class AudioFeature(Base):
    __tablename__ = "audio_features"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    feature_set_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "journal_feature_sets.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        unique=True,
        index=True,
    )

    preprocessing_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    encoder_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    encoder_version: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    encoder_revision: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    embedding_dimension: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    embedding: Mapped[list[float] | None] = mapped_column(
        VECTOR(AUDIO_EMBEDDING_DIMENSION),
        nullable=True,
    )

    duration_seconds: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    speech_ratio: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    sample_rate_hz: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    quality_status: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )

    feature_metadata: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    feature_set = relationship(
        "JournalFeatureSet",
        back_populates="audio_feature",
    )