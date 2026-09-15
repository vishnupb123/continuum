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

class JournalRead(BaseModel):
    id: UUID
    entry_type: str
    text: str
    status: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    state: StateObservationRead | None = None

class JournalAccepted(BaseModel):
    id: UUID
    status: str
