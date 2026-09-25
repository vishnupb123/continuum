from unittest.mock import Mock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.models import (
    JournalAudio,
    JournalEntry,
    StateObservation,
    User,
)
from app.services.storage.local import LocalAudioStorage
from app.services.transcription.base import TranscriptionResult
from app.tasks.journal_tasks import (
    process_journal,
    transcribe_journal_audio,
)


def task_session_factory(db_session):
    """
    Create the session factory used by the Celery task during tests.

    It uses the same database engine as the pytest db_session fixture,
    so task code sees the records created by the test.
    """
    return sessionmaker(
        bind=db_session.get_bind(),
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


def create_user(db_session) -> User:
    user = User(
        email="transcription@example.com",
        password_hash="test-password-hash",
        display_name="Transcription Test",
    )

    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    return user


def create_voice_journal(
    db_session,
    user: User,
    storage_key: str = "test/audio.webm",
):
    journal = JournalEntry(
        user_id=user.id,
        entry_type="VOICE",
        raw_text=None,
        status="QUEUED",
    )

    db_session.add(journal)
    db_session.flush()

    audio = JournalAudio(
        journal_id=journal.id,
        storage_key=storage_key,
        original_filename="recording.webm",
        mime_type="audio/webm",
        size_bytes=10,
        duration_seconds=None,
        transcription_status="TRANSCRIPTION_QUEUED",
    )

    db_session.add(audio)
    db_session.commit()

    db_session.refresh(journal)
    db_session.refresh(audio)

    return journal, audio


def test_transcription_task_persists_transcript(
    db_session,
    tmp_path,
):
    user = create_user(db_session)

    journal, audio = create_voice_journal(
        db_session,
        user,
    )

    storage = LocalAudioStorage(str(tmp_path))

    storage.put(
        audio.storage_key,
        b"fake-audio",
    )

    class FakeProvider:
        def transcribe(
            self,
            audio_bytes: bytes,
            mime_type: str,
        ):
            assert audio_bytes == b"fake-audio"
            assert mime_type == "audio/webm"

            return TranscriptionResult(
                text="Today I had a productive day.",
                duration_seconds=8.5,
                language="en",
            )

    TestSessionLocal = task_session_factory(db_session)

    with (
        patch(
            "app.tasks.journal_tasks.SessionLocal",
            TestSessionLocal,
        ),
        patch(
            "app.tasks.journal_tasks.get_audio_storage",
            return_value=storage,
        ),
        patch(
            "app.tasks.journal_tasks.get_transcription_provider",
            return_value=FakeProvider(),
        ),
        patch(
            "app.tasks.journal_tasks.process_journal.delay"
        ) as mocked_process,
    ):
        result = transcribe_journal_audio.run(
            str(journal.id)
        )

    assert result["status"] == "transcribed"
    
    mocked_process.assert_called_once_with(
    str(journal.id)
)

    journal_id = journal.id
    audio_id = audio.id

    db_session.expire_all()

    updated_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    updated_audio = db_session.get(
        JournalAudio,
        audio_id,
    )

    assert updated_journal is not None
    assert updated_audio is not None

    assert updated_journal.raw_text == (
        "Today I had a productive day."
    )

    assert updated_journal.status == "QUEUED"
    assert updated_journal.error_message is None

    assert (
        updated_audio.transcription_status
        == "TRANSCRIPTION_COMPLETED"
    )

    assert updated_audio.transcription_error is None
    assert updated_audio.duration_seconds == 8.5


def test_completed_transcription_requeues_analysis_without_retranscribing(
    db_session,
):
    user = create_user(db_session)

    journal, audio = create_voice_journal(
        db_session,
        user,
    )

    journal.raw_text = "Existing transcript"

    audio.transcription_status = (
        "TRANSCRIPTION_COMPLETED"
    )

    # Represents the important recovery state:
    #
    # transcription succeeded previously, but the
    # downstream analysis handoff was lost.
    journal.status = "QUEUED"

    db_session.commit()

    journal_id = journal.id

    TestSessionLocal = task_session_factory(
        db_session
    )

    with (
        patch(
            "app.tasks.journal_tasks.SessionLocal",
            TestSessionLocal,
        ),
        patch(
            "app.tasks.journal_tasks.get_audio_storage"
        ) as mocked_storage,
        patch(
            "app.tasks.journal_tasks.get_transcription_provider"
        ) as mocked_provider,
        patch(
            "app.tasks.journal_tasks.process_journal.delay"
        ) as mocked_process,
    ):
        result = transcribe_journal_audio.run(
            str(journal_id)
        )

    assert result["status"] == "analysis_queued"

    # Most important idempotency guarantee:
    # already-transcribed audio must not be read or
    # sent through transcription again.
    mocked_storage.assert_not_called()
    mocked_provider.assert_not_called()

    # Instead, repair the lost downstream handoff.
    mocked_process.assert_called_once_with(
        str(journal_id)
    )

    db_session.expire_all()

    refreshed_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    refreshed_audio = db_session.query(
        JournalAudio
    ).filter(
        JournalAudio.journal_id
        == journal_id
    ).one()

    # Existing durable transcription remains intact.
    assert (
        refreshed_journal.raw_text
        == "Existing transcript"
    )

    assert (
        refreshed_audio.transcription_status
        == "TRANSCRIPTION_COMPLETED"
    )

    assert (
        refreshed_journal.status
        == "QUEUED"
    )

def test_transcription_failure_marks_journal_failed(
    db_session,
    tmp_path,
):
    user = create_user(db_session)

    journal, audio = create_voice_journal(
        db_session,
        user,
    )

    journal_id = journal.id
    audio_id = audio.id

    storage = LocalAudioStorage(str(tmp_path))

    storage.put(
        audio.storage_key,
        b"fake-audio",
    )

    class FailingProvider:
        def transcribe(
            self,
            audio_bytes: bytes,
            mime_type: str,
        ):
            raise RuntimeError(
                "Provider exploded"
            )

    TestSessionLocal = task_session_factory(db_session)

    with (
        patch(
            "app.tasks.journal_tasks.SessionLocal",
            TestSessionLocal,
        ),
        patch(
            "app.tasks.journal_tasks.get_audio_storage",
            return_value=storage,
        ),
        patch(
            "app.tasks.journal_tasks.get_transcription_provider",
            return_value=FailingProvider(),
        ),
        patch(
            "app.tasks.journal_tasks.process_journal.delay"
        ) as mocked_process,
        
        
        pytest.raises(
            RuntimeError,
            match="Provider exploded",
        ),
    ):
        transcribe_journal_audio.run(
            str(journal_id)
        )

    db_session.expire_all()

    failed_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    failed_audio = db_session.get(
        JournalAudio,
        audio_id,
    )

    assert failed_journal is not None
    assert failed_audio is not None

    assert failed_journal.status == "FAILED"

    assert (
        failed_journal.error_message
        == "Transcription failed"
    )

    assert (
        failed_audio.transcription_status
        == "TRANSCRIPTION_FAILED"
    )

    assert (
        failed_audio.transcription_error
        == "Transcription failed"
    )
    
    mocked_process.assert_not_called()


def test_missing_journal_returns_missing(
    db_session,
):
    TestSessionLocal = task_session_factory(db_session)

    with patch(
        "app.tasks.journal_tasks.SessionLocal",
        TestSessionLocal,
    ):
        result = transcribe_journal_audio.run(
            "00000000-0000-0000-0000-000000000000"
        )

    assert result == {
        "status": "missing"
    }


def test_text_journal_is_not_transcribed(
    db_session,
):
    user = create_user(db_session)

    journal = JournalEntry(
        user_id=user.id,
        entry_type="TEXT",
        raw_text="Existing text journal",
        status="QUEUED",
    )

    db_session.add(journal)
    db_session.commit()
    db_session.refresh(journal)

    TestSessionLocal = task_session_factory(db_session)

    with (
        patch(
            "app.tasks.journal_tasks.SessionLocal",
            TestSessionLocal,
        ),
        patch(
            "app.tasks.journal_tasks.get_audio_storage"
        ) as mocked_storage,
        patch(
            "app.tasks.journal_tasks.get_transcription_provider"
        ) as mocked_provider,
    ):
        result = transcribe_journal_audio.run(
            str(journal.id)
        )

    assert result == {
        "status": "not_voice"
    }

    mocked_storage.assert_not_called()
    mocked_provider.assert_not_called()


def test_voice_journal_without_audio_returns_audio_missing(
    db_session,
):
    user = create_user(db_session)

    journal = JournalEntry(
        user_id=user.id,
        entry_type="VOICE",
        raw_text=None,
        status="QUEUED",
    )

    db_session.add(journal)
    db_session.commit()
    db_session.refresh(journal)

    TestSessionLocal = task_session_factory(db_session)

    with patch(
        "app.tasks.journal_tasks.SessionLocal",
        TestSessionLocal,
    ):
        result = transcribe_journal_audio.run(
            str(journal.id)
        )

    assert result == {
        "status": "audio_missing"
    }
    
def test_voice_journal_without_transcript_cannot_be_processed(
    db_session,
):
    user = create_user(db_session)

    journal = JournalEntry(
        user_id=user.id,
        entry_type="VOICE",
        raw_text=None,
        status="QUEUED",
    )

    db_session.add(journal)
    db_session.commit()
    db_session.refresh(journal)

    journal_id = journal.id

    TestSessionLocal = task_session_factory(
        db_session
    )

    with patch(
        "app.tasks.journal_tasks.SessionLocal",
        TestSessionLocal,
    ):
        result = process_journal.run(
            str(journal_id)
        )

    assert result == {
        "status": "transcript_missing"
    }

    db_session.expire_all()

    updated_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    assert updated_journal is not None
    assert updated_journal.status == "QUEUED"

    observation = db_session.scalar(
        select(StateObservation).where(
            StateObservation.journal_id
            == journal_id
        )
    )

    assert observation is None


def test_voice_transcript_enters_existing_analysis_pipeline(
    db_session,
    tmp_path,
):
    user = create_user(db_session)

    journal, audio = create_voice_journal(
        db_session,
        user,
    )

    journal_id = journal.id
    audio_id = audio.id

    storage = LocalAudioStorage(
        str(tmp_path)
    )

    storage.put(
        audio.storage_key,
        b"fake-audio",
    )

    class FakeProvider:
        def transcribe(
            self,
            audio_bytes: bytes,
            mime_type: str,
        ):
            return TranscriptionResult(
                text=(
                    "I had a busy day but I feel "
                    "positive about the progress."
                ),
                duration_seconds=12.0,
                language="en",
            )

    TestSessionLocal = task_session_factory(
        db_session
    )

    def run_analysis(journal_id_value: str):
        return process_journal.run(
            journal_id_value
        )

    with (
        patch(
            "app.tasks.journal_tasks.SessionLocal",
            TestSessionLocal,
        ),
        patch(
            "app.tasks.journal_tasks.get_audio_storage",
            return_value=storage,
        ),
        patch(
            "app.tasks.journal_tasks.get_transcription_provider",
            return_value=FakeProvider(),
        ),
        patch(
            "app.tasks.journal_tasks.process_journal.delay",
            side_effect=run_analysis,
        ) as mocked_process,
    ):
        result = transcribe_journal_audio.run(
            str(journal_id)
        )

    assert result["status"] == "transcribed"

    mocked_process.assert_called_once_with(
        str(journal_id)
    )

    db_session.expire_all()

    completed_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    completed_audio = db_session.get(
        JournalAudio,
        audio_id,
    )

    observation = db_session.scalar(
        select(StateObservation).where(
            StateObservation.journal_id
            == journal_id
        )
    )

    assert completed_journal is not None
    assert completed_audio is not None
    assert observation is not None

    assert completed_journal.raw_text == (
        "I had a busy day but I feel "
        "positive about the progress."
    )

    assert completed_journal.status == "COMPLETED"

    assert (
        completed_audio.transcription_status
        == "TRANSCRIPTION_COMPLETED"
    )

    assert completed_audio.duration_seconds == 12.0

    assert observation.model_version is not None
    assert observation.energy is not None
    assert observation.stress is not None
    assert observation.confidence is not None
    
def test_analysis_dispatch_failure_does_not_mark_transcription_failed(
    db_session,
):
    user = create_user(db_session)

    journal, audio = create_voice_journal(
        db_session,
        user,
    )

    journal_id = journal.id

    TestSessionLocal = task_session_factory(
        db_session
    )

    mock_storage = Mock()
    mock_storage.get.return_value = (
        b"fake-audio-bytes"
    )

    mock_provider = Mock()
    mock_provider.transcribe.return_value = (
        TranscriptionResult(
            text="Successfully transcribed text",
            duration_seconds=12.5,
            language="en",
        )
    )

    with (
        patch(
            "app.tasks.journal_tasks.SessionLocal",
            TestSessionLocal,
        ),
        patch(
            "app.tasks.journal_tasks.get_audio_storage",
            return_value=mock_storage,
        ),
        patch(
            "app.tasks.journal_tasks.get_transcription_provider",
            return_value=mock_provider,
        ),
        patch(
            "app.tasks.journal_tasks.process_journal.delay",
            side_effect=ConnectionError(
                "Redis unavailable"
            ),
        ),
    ):
        try:
            transcribe_journal_audio.run(
                str(journal_id)
            )
        except ConnectionError:
            pass

    db_session.expire_all()

    refreshed_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    refreshed_audio = db_session.query(
        JournalAudio
    ).filter(
        JournalAudio.journal_id
        == journal_id
    ).one()

    # Transcription itself succeeded and was durably
    # committed before downstream dispatch failed.
    assert (
        refreshed_journal.raw_text
        == "Successfully transcribed text"
    )

    assert (
        refreshed_audio.transcription_status
        == "TRANSCRIPTION_COMPLETED"
    )

    assert (
        refreshed_audio.transcription_error
        is None
    )

    assert (
        refreshed_audio.duration_seconds
        == 12.5
    )

    # Analysis has not completed, so the journal remains
    # recoverable rather than being incorrectly marked FAILED.
    assert (
        refreshed_journal.status
        == "QUEUED"
    )

    assert (
        refreshed_journal.error_message
        is None
    )
    
def test_dispatch_failure_can_be_recovered_without_retranscription(
    db_session,
):
    user = create_user(db_session)

    journal, audio = create_voice_journal(
        db_session,
        user,
    )

    journal_id = journal.id

    TestSessionLocal = task_session_factory(
        db_session
    )

    mock_storage = Mock()
    mock_storage.get.return_value = (
        b"fake-audio-bytes"
    )

    mock_provider = Mock()
    mock_provider.transcribe.return_value = (
        TranscriptionResult(
            text="Durable transcript",
            duration_seconds=8.0,
            language="en",
        )
    )

    # -------------------------------------------------
    # FIRST ATTEMPT
    # -------------------------------------------------
    #
    # Transcription succeeds, but analysis dispatch
    # fails.
    #
    with (
        patch(
            "app.tasks.journal_tasks.SessionLocal",
            TestSessionLocal,
        ),
        patch(
            "app.tasks.journal_tasks.get_audio_storage",
            return_value=mock_storage,
        ),
        patch(
            "app.tasks.journal_tasks.get_transcription_provider",
            return_value=mock_provider,
        ),
        patch(
            "app.tasks.journal_tasks.process_journal.delay",
            side_effect=ConnectionError(
                "Redis unavailable"
            ),
        ),
    ):
        try:
            transcribe_journal_audio.run(
                str(journal_id)
            )
        except ConnectionError:
            pass

    db_session.expire_all()

    journal_after_failure = db_session.get(
        JournalEntry,
        journal_id,
    )

    audio_after_failure = db_session.query(
        JournalAudio
    ).filter(
        JournalAudio.journal_id
        == journal_id
    ).one()

    assert (
        journal_after_failure.raw_text
        == "Durable transcript"
    )

    assert (
        journal_after_failure.status
        == "QUEUED"
    )

    assert (
        audio_after_failure.transcription_status
        == "TRANSCRIPTION_COMPLETED"
    )

    # The actual transcription happened exactly once.
    assert mock_storage.get.call_count == 1
    assert mock_provider.transcribe.call_count == 1

    # -------------------------------------------------
    # RECOVERY ATTEMPT
    # -------------------------------------------------
    #
    # Redis is available again.
    #
    # Running the transcription task again should
    # recognize the durable completed transcription
    # and repair only the downstream handoff.
    #
    with (
        patch(
            "app.tasks.journal_tasks.SessionLocal",
            TestSessionLocal,
        ),
        patch(
            "app.tasks.journal_tasks.get_audio_storage"
        ) as recovery_storage,
        patch(
            "app.tasks.journal_tasks.get_transcription_provider"
        ) as recovery_provider,
        patch(
            "app.tasks.journal_tasks.process_journal.delay"
        ) as recovered_process,
    ):
        result = transcribe_journal_audio.run(
            str(journal_id)
        )

    assert (
        result["status"]
        == "analysis_queued"
    )

    # Critical:
    # recovery must NOT touch the original audio or
    # invoke transcription again.
    recovery_storage.assert_not_called()
    recovery_provider.assert_not_called()

    recovered_process.assert_called_once_with(
        str(journal_id)
    )

    db_session.expire_all()

    recovered_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    recovered_audio = db_session.query(
        JournalAudio
    ).filter(
        JournalAudio.journal_id
        == journal_id
    ).one()

    # Durable transcription is unchanged.
    assert (
        recovered_journal.raw_text
        == "Durable transcript"
    )

    assert (
        recovered_journal.status
        == "QUEUED"
    )

    assert (
        recovered_audio.transcription_status
        == "TRANSCRIPTION_COMPLETED"
    )

    assert (
        recovered_audio.transcription_error
        is None
    )

    # Across both attempts, the real transcription
    # provider was invoked only once.
    assert mock_provider.transcribe.call_count == 1