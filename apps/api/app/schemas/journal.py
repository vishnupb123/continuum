from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class JournalCreate(BaseModel):
    entry_type: str = Field(default="TEXT", pattern="^(TEXT)$")
    text: str = Field(min_length=1, max_length=10000)


class StateObservationRead(BaseModel):
    energy: float
    stress: float
    confidence: float
    model_version: str


class JournalAudioRead(BaseModel):
    id: UUID
    original_filename: str | None = None
    mime_type: str
    size_bytes: int
    duration_seconds: float | None = None
    transcription_status: str


class JournalRead(BaseModel):
    id: UUID
    entry_type: str
    text: str | None
    status: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    state: StateObservationRead | None = None
    audio: JournalAudioRead | None = None


class JournalAccepted(BaseModel):
    id: UUID
    status: str


class JournalListResponse(BaseModel):
    items: list[JournalRead]
    total: int
    limit: int
    offset: int


class JournalUpdate(BaseModel):
    text: str = Field(min_length=1, max_length=10000)