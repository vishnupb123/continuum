import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from app.db.base import Base
from app.models.context_constants import (
    CONTEXT_REPRESENTATION_DIMENSION,
    CONTEXT_STATUS_PENDING,
)


class ContextInference(Base):
    __tablename__ = "context_inferences"

    __table_args__ = (
        UniqueConstraint(
            "feature_set_id",
            "architecture_version",
            "model_revision",
            name="uq_context_inference_generation",
        ),
        CheckConstraint(
            (
                "energy IS NULL OR "
                "(energy >= 0.0 AND energy <= 1.0)"
            ),
            name="ck_context_inferences_energy_range",
        ),
        CheckConstraint(
            (
                "stress IS NULL OR "
                "(stress >= 0.0 AND stress <= 1.0)"
            ),
            name="ck_context_inferences_stress_range",
        ),
        CheckConstraint(
            (
                "positive_mood IS NULL OR "
                "(positive_mood >= 0.0 "
                "AND positive_mood <= 1.0)"
            ),
            name=(
                "ck_context_inferences_"
                "positive_mood_range"
            ),
        ),
        CheckConstraint(
            (
                "social_connection IS NULL OR "
                "(social_connection >= 0.0 "
                "AND social_connection <= 1.0)"
            ),
            name=(
                "ck_context_inferences_"
                "social_connection_range"
            ),
        ),
        CheckConstraint(
            (
                "confidence_score IS NULL OR "
                "(confidence_score >= 0.0 "
                "AND confidence_score <= 1.0)"
            ),
            name=(
                "ck_context_inferences_"
                "confidence_score_range"
            ),
        ),
    )

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
        index=True,
    )

    architecture_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    model_revision: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    model_artifact_hash: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    state_capability: Mapped[str] = mapped_column(
    String(40),
    nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default=CONTEXT_STATUS_PENDING,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    representation_dimension: Mapped[int | None] = (
        mapped_column(
            Integer,
            nullable=True,
        )
    )

    representation: Mapped[
        list[float] | None
    ] = mapped_column(
        Vector(CONTEXT_REPRESENTATION_DIMENSION),
        nullable=True,
    )

    energy: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    stress: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    positive_mood: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    social_connection: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    confidence_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    confidence_calibrated: Mapped[bool] = (
        mapped_column(
            Boolean,
            nullable=False,
            default=False,
        )
    )



    inference_metadata: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
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

    completed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    feature_set = relationship(
        "JournalFeatureSet",
        back_populates="context_inferences",
    )