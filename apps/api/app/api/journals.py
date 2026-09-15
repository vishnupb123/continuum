from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.db.session import get_db
from app.models import JournalEntry, StateObservation
from app.schemas.journal import JournalAccepted, JournalCreate, JournalRead, StateObservationRead
from app.tasks.journal_tasks import process_journal

router = APIRouter(prefix="/v1/journals", tags=["journals"])

@router.post("", response_model=JournalAccepted, status_code=status.HTTP_202_ACCEPTED)
def create_journal(payload: JournalCreate, db: Session = Depends(get_db)):
    journal = JournalEntry(entry_type=payload.entry_type, raw_text=payload.text, status="QUEUED")
    db.add(journal)
    db.commit()
    db.refresh(journal)
    process_journal.delay(str(journal.id))
    return JournalAccepted(id=journal.id, status=journal.status)

@router.get("/{journal_id}", response_model=JournalRead)
def get_journal(journal_id: UUID, db: Session = Depends(get_db)):
    journal = db.get(JournalEntry, journal_id)
    if journal is None:
        raise HTTPException(status_code=404, detail="Journal not found")
    observation = db.scalar(select(StateObservation).where(StateObservation.journal_id == journal.id))
    state = None
    if observation:
        state = StateObservationRead(
            energy=observation.energy,
            stress=observation.stress,
            confidence=observation.confidence,
            model_version=observation.model_version,
        )
    return JournalRead(
        id=journal.id,
        entry_type=journal.entry_type,
        text=journal.raw_text,
        status=journal.status,
        error_message=journal.error_message,
        created_at=journal.created_at,
        updated_at=journal.updated_at,
        state=state,
    )
