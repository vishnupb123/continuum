import uuid
from datetime import datetime, timezone
from sqlalchemy import DateTime, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base

class StateObservation(Base):
    __tablename__ = "state_observations"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    journal_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("journal_entries.id", ondelete="CASCADE"), unique=True, nullable=False)
    energy: Mapped[float] = mapped_column(Float, nullable=False)
    stress: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    journal = relationship("JournalEntry", back_populates="observation")
