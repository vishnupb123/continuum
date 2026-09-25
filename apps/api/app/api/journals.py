import uuid
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.db.session import get_db
from app.models import JournalAudio, JournalEntry, StateObservation, User
from app.schemas.journal import (
    JournalAccepted,
    JournalAudioRead,
    JournalCreate,
    JournalListResponse,
    JournalRead,
    JournalUpdate,
    StateObservationRead,
)
from app.services.storage.factory import get_audio_storage
from app.tasks.journal_tasks import (
    process_journal,
    transcribe_journal_audio,
)


router = APIRouter(
    prefix="/v1/journals",
    tags=["journals"],
)


ALLOWED_AUDIO_TYPES = {
    "audio/webm": "webm",
    "audio/mpeg": "mp3",
    "audio/mp4": "m4a",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}

MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024


def build_state_response(
    observation: StateObservation | None,
) -> StateObservationRead | None:
    if observation is None:
        return None

    return StateObservationRead(
        energy=observation.energy,
        stress=observation.stress,
        confidence=observation.confidence,
        model_version=observation.model_version,
    )


def build_audio_response(
    audio: JournalAudio | None,
) -> JournalAudioRead | None:
    if audio is None:
        return None

    return JournalAudioRead(
        id=audio.id,
        original_filename=audio.original_filename,
        mime_type=audio.mime_type,
        size_bytes=audio.size_bytes,
        duration_seconds=audio.duration_seconds,
        transcription_status=audio.transcription_status,
    )


def build_journal_response(
    journal: JournalEntry,
    observation: StateObservation | None,
) -> JournalRead:
    return JournalRead(
        id=journal.id,
        entry_type=journal.entry_type,
        text=journal.raw_text,
        status=journal.status,
        error_message=journal.error_message,
        created_at=journal.created_at,
        updated_at=journal.updated_at,
        state=build_state_response(observation),
        audio=build_audio_response(journal.audio),
    )


@router.post(
    "",
    response_model=JournalAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_journal(
    payload: JournalCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    journal = JournalEntry(
        user_id=current_user.id,
        entry_type=payload.entry_type,
        raw_text=payload.text,
        status="QUEUED",
    )

    db.add(journal)
    db.commit()
    db.refresh(journal)

    process_journal.delay(str(journal.id))

    return JournalAccepted(
        id=journal.id,
        status=journal.status,
    )


@router.get(
    "",
    response_model=JournalListResponse,
)
def list_journals(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    total = (
        db.scalar(
            select(func.count())
            .select_from(JournalEntry)
            .where(JournalEntry.user_id == current_user.id)
        )
        or 0
    )

    journals = db.scalars(
        select(JournalEntry)
        .where(JournalEntry.user_id == current_user.id)
        .order_by(JournalEntry.created_at.desc())
        .offset(offset)
        .limit(limit)
    ).all()

    items = []

    for journal in journals:
        observation = db.scalar(
            select(StateObservation).where(
                StateObservation.journal_id == journal.id
            )
        )

        items.append(
            build_journal_response(
                journal=journal,
                observation=observation,
            )
        )

    return JournalListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "/voice",
    response_model=JournalAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_voice_journal(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    raw_content_type = (file.content_type or "").lower().strip()

    # MediaRecorder may include codec parameters, for example:
    # "audio/webm;codecs=opus".
    #
    # Validate against the base MIME type while preserving the
    # browser-reported MIME type in JournalAudio metadata.
    content_type = raw_content_type.split(";", 1)[0].strip()

    extension = ALLOWED_AUDIO_TYPES.get(content_type)

    if extension is None:
        await file.close()

        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported audio format",
        )

    audio_bytes = await file.read(MAX_AUDIO_SIZE_BYTES + 1)

    if not audio_bytes:
        await file.close()

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is empty",
        )

    if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
        await file.close()

        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Audio file exceeds the 25 MB limit",
        )

    journal_id = uuid.uuid4()

    storage_key = (
        f"users/{current_user.id}/"
        f"journals/{journal_id}/"
        f"audio.{extension}"
    )

    storage = get_audio_storage()

    journal = JournalEntry(
        id=journal_id,
        user_id=current_user.id,
        entry_type="VOICE",
        raw_text=None,
        status="QUEUED",
    )

    audio = JournalAudio(
        journal_id=journal_id,
        storage_key=storage_key,
        original_filename=file.filename,
        mime_type=raw_content_type,
        size_bytes=len(audio_bytes),
        duration_seconds=None,
        transcription_status="TRANSCRIPTION_QUEUED",
    )

    object_stored = False

    try:
        storage.put(
            storage_key,
            audio_bytes,
        )
        object_stored = True

        db.add(journal)
        db.add(audio)

        db.commit()
        db.refresh(journal)

    except Exception:
        db.rollback()

        if object_stored:
            try:
                storage.delete(storage_key)
            except Exception:
                # Storage cleanup failure should not hide
                # the original persistence failure.
                pass

        raise

    finally:
        await file.close()
    
    transcribe_journal_audio.delay(
    str(journal.id)
    )
    
    return JournalAccepted(
        id=journal.id,
        status=journal.status,
    )


@router.get(
    "/{journal_id}/audio",
    response_class=Response,
)
def get_journal_audio(
    journal_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    journal = db.scalar(
        select(JournalEntry).where(
            JournalEntry.id == journal_id,
            JournalEntry.user_id == current_user.id,
        )
    )

    if journal is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Journal not found",
        )

    if (
        journal.entry_type != "VOICE"
        or journal.audio is None
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio not found",
        )

    storage = get_audio_storage()

    try:
        audio_bytes = storage.get(
            journal.audio.storage_key
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio not found",
        )

    return Response(
        content=audio_bytes,
        media_type=journal.audio.mime_type,
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "private, no-store",
        },
    )
@router.get(
    "/{journal_id}",
    response_model=JournalRead,
)
def get_journal(
    journal_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    journal = db.scalar(
        select(JournalEntry).where(
            JournalEntry.id == journal_id,
            JournalEntry.user_id == current_user.id,
        )
    )

    if journal is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Journal not found",
        )

    observation = db.scalar(
        select(StateObservation).where(
            StateObservation.journal_id == journal.id
        )
    )

    return build_journal_response(
        journal=journal,
        observation=observation,
    )


@router.patch(
    "/{journal_id}",
    response_model=JournalAccepted,
)
def update_journal(
    journal_id: UUID,
    payload: JournalUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    journal = db.scalar(
        select(JournalEntry).where(
            JournalEntry.id == journal_id,
            JournalEntry.user_id == current_user.id,
        )
    )

    if journal is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Journal not found",
        )

    if journal.entry_type != "TEXT":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only text journals can be edited",
        )

    observation = db.scalar(
        select(StateObservation).where(
            StateObservation.journal_id == journal.id
        )
    )

    if observation is not None:
        db.delete(observation)

    journal.raw_text = payload.text
    journal.status = "QUEUED"
    journal.error_message = None

    db.commit()
    db.refresh(journal)

    process_journal.delay(str(journal.id))

    return JournalAccepted(
        id=journal.id,
        status=journal.status,
    )


@router.delete(
    "/{journal_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_journal(
    journal_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    journal = db.scalar(
        select(JournalEntry).where(
            JournalEntry.id == journal_id,
            JournalEntry.user_id == current_user.id,
        )
    )

    if journal is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Journal not found",
        )

    # Capture the private storage key before deleting the ORM objects.
    storage_key = (
        journal.audio.storage_key
        if journal.audio is not None
        else None
    )

    db.delete(journal)

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    # Only remove the physical object after the database deletion succeeds.
    #
    # This ordering prevents a failed DB transaction from leaving a journal
    # record that points to an already-deleted audio object.
    if storage_key is not None:
        storage = get_audio_storage()

        try:
            storage.delete(storage_key)
        except Exception:
            # At this stage the DB deletion has already committed.
            #
            # We deliberately do not turn a successful journal deletion into
            # an API failure. Production cleanup/retry handling will be added
            # when we introduce durable storage and observability.
            pass

    return None