import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class TextFeature(Base):
    __tablename__ = "text_features"

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

    source_type: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
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

    embedding_dimension: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    word_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    character_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
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
        back_populates="text_feature",
    )