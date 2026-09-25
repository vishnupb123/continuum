import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.models import JournalAudio, JournalEntry, User


def create_user(db_session) -> User:
    user = User(
        email=f"{uuid.uuid4()}@example.com",
        password_hash="test-password-hash",
        display_name="Audio Test User",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def create_voice_journal(db_session, user: User) -> JournalEntry:
    journal = JournalEntry(
        user_id=user.id,
        entry_type="VOICE",
        raw_text=None,
        status="QUEUED",
    )
    db_session.add(journal)
    db_session.commit()
    db_session.refresh(journal)
    return journal


def create_audio(
    db_session,
    journal: JournalEntry,
    storage_key: str | None = None,
) -> JournalAudio:
    audio = JournalAudio(
        journal_id=journal.id,
        storage_key=storage_key or f"users/{journal.user_id}/journals/{journal.id}/audio.webm",
        original_filename="recording.webm",
        mime_type="audio/webm",
        size_bytes=1024,
        duration_seconds=12.5,
        transcription_status="TRANSCRIPTION_QUEUED",
    )
    db_session.add(audio)
    db_session.commit()
    db_session.refresh(audio)
    return audio


def test_voice_journal_can_exist_without_transcript(db_session):
    user = create_user(db_session)
    journal = create_voice_journal(db_session, user)

    assert journal.entry_type == "VOICE"
    assert journal.raw_text is None


def test_journal_audio_can_be_persisted(db_session):
    user = create_user(db_session)
    journal = create_voice_journal(db_session, user)
    audio = create_audio(db_session, journal)

    assert audio.id is not None
    assert audio.journal_id == journal.id
    assert audio.mime_type == "audio/webm"
    assert audio.size_bytes == 1024
    assert audio.duration_seconds == 12.5
    assert audio.transcription_status == "TRANSCRIPTION_QUEUED"


def test_journal_audio_relationship_is_one_to_one(db_session):
    user = create_user(db_session)
    journal = create_voice_journal(db_session, user)
    audio = create_audio(db_session, journal)

    db_session.refresh(journal)

    assert journal.audio is not None
    assert journal.audio.id == audio.id
    assert audio.journal.id == journal.id


def test_second_audio_for_same_journal_is_rejected(db_session):
    user = create_user(db_session)
    journal = create_voice_journal(db_session, user)

    create_audio(db_session, journal)

    duplicate = JournalAudio(
        journal_id=journal.id,
        storage_key=f"duplicate/{uuid.uuid4()}.webm",
        mime_type="audio/webm",
        size_bytes=100,
        transcription_status="TRANSCRIPTION_QUEUED",
    )

    db_session.add(duplicate)

    with pytest.raises(IntegrityError):
        db_session.commit()

    db_session.rollback()


def test_storage_key_must_be_unique(db_session):
    user = create_user(db_session)

    journal_one = create_voice_journal(db_session, user)
    journal_two = create_voice_journal(db_session, user)

    shared_key = f"users/{user.id}/shared/audio.webm"

    create_audio(db_session, journal_one, shared_key)

    duplicate = JournalAudio(
        journal_id=journal_two.id,
        storage_key=shared_key,
        mime_type="audio/webm",
        size_bytes=100,
        transcription_status="TRANSCRIPTION_QUEUED",
    )

    db_session.add(duplicate)

    with pytest.raises(IntegrityError):
        db_session.commit()

    db_session.rollback()


def test_deleting_journal_cascades_audio_metadata(db_session):
    user = create_user(db_session)
    journal = create_voice_journal(db_session, user)
    audio = create_audio(db_session, journal)

    audio_id = audio.id

    db_session.delete(journal)
    db_session.commit()

    stored_audio = db_session.scalar(
        select(JournalAudio).where(JournalAudio.id == audio_id)
    )

    assert stored_audio is None