from uuid import UUID

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models import JournalAudio, JournalEntry, StateObservation
from app.services.mock_context_model import MockContextModel
from app.services.storage.factory import get_audio_storage
from app.services.transcription.factory import get_transcription_provider
from app.tasks.feature_tasks import (
    generate_journal_features,
)
from app.tasks.celery_app import celery_app

def _queue_feature_generation(
    journal_id: str,
) -> None:
    """
    Hand a completed journal to the M3 parent
    orchestration task.

    The parent owns feature-generation identity and
    modality-specific dispatch.
    """

    generate_journal_features.delay(
        journal_id
    )
@celery_app.task(
    name="app.tasks.journal_tasks.process_journal",
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def process_journal(journal_id: str):
    db = SessionLocal()

    journal_uuid = UUID(journal_id)
    should_queue_features = False
    result_status = None

    try:
        journal = db.get(
            JournalEntry,
            journal_uuid,
        )

        if journal is None:
            return {
                "status": "missing"
            }

        # -------------------------------------------------
        # RECOVERY / IDEMPOTENCY PATH
        # -------------------------------------------------
        #
        # Analysis may already be complete while the
        # downstream feature dispatch previously failed.
        #
        # Re-running this task must repair that handoff.
        #
        if journal.status == "COMPLETED":
            should_queue_features = True
            result_status = (
                "already_completed"
            )

        else:
            # A VOICE journal must never enter analysis
            # until transcription has produced usable
            # text.
            if (
                journal.entry_type == "VOICE"
                and (
                    journal.raw_text is None
                    or not journal.raw_text.strip()
                )
            ):
                return {
                    "status":
                    "transcript_missing"
                }

            journal.status = "PROCESSING"
            journal.error_message = None
            db.commit()

            existing = db.scalar(
                select(
                    StateObservation
                ).where(
                    StateObservation.journal_id
                    == journal.id
                )
            )

            if existing is None:
                result = (
                    MockContextModel()
                    .predict(
                        journal.raw_text
                    )
                )

                db.add(
                    StateObservation(
                        journal_id=journal.id,
                        energy=result.energy,
                        stress=result.stress,
                        confidence=(
                            result.confidence
                        ),
                        model_version=(
                            result.model_version
                        ),
                    )
                )

            journal.status = "COMPLETED"

            # Analysis completion is committed before
            # M3 dispatch. Journal state therefore does
            # not depend on feature extraction.
            db.commit()

            should_queue_features = True
            result_status = "completed"

    except Exception:
        db.rollback()

        journal = db.get(
            JournalEntry,
            journal_uuid,
        )

        if journal is not None:
            journal.status = "FAILED"
            journal.error_message = (
                "Processing failed"
            )
            db.commit()

        raise

    finally:
        db.close()

    # -----------------------------------------------------
    # M3 DOWNSTREAM HANDOFF
    # -----------------------------------------------------
    #
    # Deliberately outside the analysis transaction.
    #
    # If Redis/Celery dispatch fails, the journal remains
    # COMPLETED. Re-running process_journal() enters the
    # recovery path above and retries this dispatch.
    #
    if should_queue_features:
        _queue_feature_generation(
          str(journal_uuid)
        )

    return {
        "status": result_status,
        "features": "queued",
    }
    
@celery_app.task(
    name="app.tasks.journal_tasks.transcribe_journal_audio",
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def transcribe_journal_audio(journal_id: str):
    db = SessionLocal()

    try:
        journal_uuid = UUID(journal_id)

        journal = db.get(
            JournalEntry,
            journal_uuid,
        )

        if journal is None:
            return {
                "status": "missing"
            }

        if journal.entry_type != "VOICE":
            return {
                "status": "not_voice"
            }

        audio = db.scalar(
            select(JournalAudio).where(
                JournalAudio.journal_id
                == journal.id
            )
        )

        if audio is None:
            return {
                "status": "audio_missing"
            }

        # -------------------------------------------------
        # RECOVERY / IDEMPOTENCY PATH
        # -------------------------------------------------
        #
        # The transcription may already have completed
        # while the downstream analysis dispatch failed.
        #
        # In that situation we must NOT transcribe again.
        # Instead, repair the handoff by queueing analysis.
        #
        if (
            audio.transcription_status
            == "TRANSCRIPTION_COMPLETED"
        ):
            if journal.status == "COMPLETED":
                return {
                    "status":
                    "already_completed"
                }

            process_journal.delay(
                str(journal.id)
            )

            return {
                "status":
                "analysis_queued"
            }

        # -------------------------------------------------
        # TRANSCRIPTION START
        # -------------------------------------------------

        audio.transcription_status = (
            "TRANSCRIBING"
        )

        audio.transcription_error = None

        journal.status = "PROCESSING"
        journal.error_message = None

        db.commit()

        storage = get_audio_storage()

        audio_bytes = storage.get(
            audio.storage_key
        )

        provider = (
            get_transcription_provider()
        )

        result = provider.transcribe(
            audio_bytes=audio_bytes,
            mime_type=audio.mime_type,
        )

        transcript = (
            result.text.strip()
        )

        if not transcript:
            raise ValueError(
                "Transcription provider returned "
                "an empty transcript"
            )

        # -------------------------------------------------
        # TRANSCRIPTION SUCCESS
        # -------------------------------------------------
        #
        # Persist the successful transcription BEFORE
        # attempting to dispatch downstream analysis.
        #
        # This transaction is the source of truth.
        #
        journal.raw_text = transcript

        audio.duration_seconds = (
            result.duration_seconds
        )

        audio.transcription_status = (
            "TRANSCRIPTION_COMPLETED"
        )

        audio.transcription_error = None

        journal.status = "QUEUED"
        journal.error_message = None

        db.commit()

    except Exception:
        # -------------------------------------------------
        # TRUE TRANSCRIPTION FAILURE
        # -------------------------------------------------
        #
        # Only failures that happen before successful
        # transcription persistence reach this block.
        #
        db.rollback()

        try:
            journal_uuid = UUID(
                journal_id
            )
        except ValueError:
            raise

        journal = db.get(
            JournalEntry,
            journal_uuid,
        )

        if journal is not None:
            audio = db.scalar(
                select(JournalAudio).where(
                    JournalAudio.journal_id
                    == journal.id
                )
            )

            journal.status = "FAILED"

            journal.error_message = (
                "Transcription failed"
            )

            if audio is not None:
                audio.transcription_status = (
                    "TRANSCRIPTION_FAILED"
                )

                audio.transcription_error = (
                    "Transcription failed"
                )

            db.commit()

        raise

    finally:
        db.close()

    # -----------------------------------------------------
    # DOWNSTREAM HANDOFF
    # -----------------------------------------------------
    #
    # Deliberately outside the transcription try/except.
    #
    # If Celery/Redis dispatch fails here, transcription
    # remains TRANSCRIPTION_COMPLETED and the journal
    # remains QUEUED.
    #
    # Re-running this task will enter the recovery path
    # above and retry the analysis dispatch without
    # retranscribing the audio.
    #
    process_journal.delay(
        str(journal_uuid)
    )

    return {
        "status": "transcribed",
        "journal_id": str(
            journal_uuid
        ),
    }