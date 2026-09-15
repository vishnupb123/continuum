from uuid import UUID
from sqlalchemy import select
from app.db.session import SessionLocal
from app.models import JournalEntry, StateObservation
from app.services.mock_context_model import MockContextModel
from app.tasks.celery_app import celery_app

@celery_app.task(name="app.tasks.journal_tasks.process_journal", autoretry_for=(ConnectionError,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def process_journal(journal_id: str):
    db = SessionLocal()
    try:
        journal = db.get(JournalEntry, UUID(journal_id))
        if journal is None:
            return {"status": "missing"}
        if journal.status == "COMPLETED":
            return {"status": "already_completed"}

        journal.status = "PROCESSING"
        journal.error_message = None
        db.commit()

        existing = db.scalar(select(StateObservation).where(StateObservation.journal_id == journal.id))
        if existing is None:
            result = MockContextModel().predict(journal.raw_text)
            db.add(StateObservation(
                journal_id=journal.id,
                energy=result.energy,
                stress=result.stress,
                confidence=result.confidence,
                model_version=result.model_version,
            ))

        journal.status = "COMPLETED"
        db.commit()
        return {"status": "completed"}
    except Exception as exc:
        db.rollback()
        journal = db.get(JournalEntry, UUID(journal_id))
        if journal is not None:
            journal.status = "FAILED"
            journal.error_message = "Processing failed"
            db.commit()
        raise exc
    finally:
        db.close()
