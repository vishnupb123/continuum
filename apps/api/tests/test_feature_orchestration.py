from unittest.mock import patch

from app.models.journal import JournalEntry
from app.models.journal_audio import JournalAudio
from app.models.journal_feature_set import (
    JournalFeatureSet,
)
from app.models.user import User
from app.tasks.feature_tasks import (
    generate_journal_features,
)


def _make_text_journal(
    db_session,
):
    user = User(
        email="orchestration-text@example.com",
        password_hash="test-password-hash",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="TEXT",
        raw_text=(
            "This is a completed text journal "
            "for orchestration testing."
        ),
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.commit()
    db_session.refresh(journal)

    return journal


def _make_voice_journal(
    db_session,
):
    user = User(
        email="orchestration-voice@example.com",
        password_hash="test-password-hash",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="VOICE",
        raw_text=(
            "This is a completed voice "
            "journal transcript."
        ),
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.flush()

    audio = JournalAudio(
        journal_id=journal.id,
        storage_key=(
            "tests/orchestration.wav"
        ),
        original_filename=(
            "orchestration.wav"
        ),
        mime_type="audio/wav",
        size_bytes=123,
        transcription_status=(
            "TRANSCRIPTION_COMPLETED"
        ),
    )

    db_session.add(audio)
    db_session.commit()
    db_session.refresh(journal)

    return journal


@patch(
    "app.tasks.feature_tasks."
    "_dispatch_text_feature_task"
)
@patch(
    "app.tasks.feature_tasks.SessionLocal"
)
def test_text_parent_resolves_generation_and_dispatches_text(
    session_local_mock,
    text_dispatch_mock,
    db_session,
):
    session_local_mock.return_value = (
        db_session
    )

    journal = _make_text_journal(
        db_session
    )

    result = (
        generate_journal_features.run(
            str(journal.id)
        )
    )

    assert result["status"] == "queued"

    assert result["modalities"] == [
        "TEXT",
    ]

    feature_sets = (
        db_session.query(
            JournalFeatureSet
        )
        .filter(
            JournalFeatureSet.journal_id
            == journal.id
        )
        .all()
    )

    assert len(feature_sets) == 1

    feature_set = feature_sets[0]

    text_dispatch_mock.assert_called_once_with(
        str(journal.id),
        str(feature_set.id),
    )


@patch(
    "app.tasks.feature_tasks."
    "_dispatch_audio_feature_task"
)
@patch(
    "app.tasks.feature_tasks."
    "_dispatch_text_feature_task"
)
@patch(
    "app.tasks.feature_tasks."
    "_calculate_source_hash"
)
@patch(
    "app.tasks.feature_tasks.SessionLocal"
)
def test_voice_parent_dispatches_both_modalities(
    session_local_mock,
    source_hash_mock,
    text_dispatch_mock,
    audio_dispatch_mock,
    db_session,
):
    session_local_mock.return_value = (
        db_session
    )

    # Fingerprinting itself is already tested.
    # This test is specifically about orchestration.
    source_hash_mock.return_value = (
        "voice-orchestration-hash"
    )

    journal = _make_voice_journal(
        db_session
    )

    result = (
        generate_journal_features.run(
            str(journal.id)
        )
    )

    assert result["status"] == "queued"

    assert result["modalities"] == [
        "TEXT",
        "AUDIO",
    ]

    feature_sets = (
        db_session.query(
            JournalFeatureSet
        )
        .filter(
            JournalFeatureSet.journal_id
            == journal.id
        )
        .all()
    )

    assert len(feature_sets) == 1

    feature_set = feature_sets[0]

    text_dispatch_mock.assert_called_once_with(
        str(journal.id),
        str(feature_set.id),
    )

    audio_dispatch_mock.assert_called_once_with(
        str(journal.id),
        str(feature_set.id),
    )


@patch(
    "app.tasks.feature_tasks."
    "_dispatch_text_feature_task"
)
@patch(
    "app.tasks.feature_tasks.SessionLocal"
)
def test_parent_redispatches_existing_pending_generation(
    session_local_mock,
    text_dispatch_mock,
    db_session,
):
    session_local_mock.return_value = (
        db_session
    )

    journal = _make_text_journal(
        db_session
    )

    first = (
        generate_journal_features.run(
            str(journal.id)
        )
    )

    assert first["status"] == "queued"

    first_feature_set_id = (
        first["feature_set_id"]
    )

    # Simulate:
    #
    # generation was committed, but downstream
    # Celery delivery was lost.
    #
    # Running the parent again must repair the
    # handoff using the SAME generation.
    second = (
        generate_journal_features.run(
            str(journal.id)
        )
    )

    assert second["status"] == "queued"

    assert (
        second["feature_set_id"]
        == first_feature_set_id
    )

    feature_sets = (
        db_session.query(
            JournalFeatureSet
        )
        .filter(
            JournalFeatureSet.journal_id
            == journal.id
        )
        .all()
    )

    assert len(feature_sets) == 1

    assert (
        text_dispatch_mock.call_count
        == 2
    )


@patch(
    "app.tasks.feature_tasks.SessionLocal"
)
def test_parent_rejects_unfinished_journal(
    session_local_mock,
    db_session,
):
    session_local_mock.return_value = (
        db_session
    )

    user = User(
        email="orchestration-unfinished@example.com",
        password_hash="test-password-hash",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="TEXT",
        raw_text="Not ready yet.",
        status="PROCESSING",
    )

    db_session.add(journal)
    db_session.commit()

    result = (
        generate_journal_features.run(
            str(journal.id)
        )
    )

    assert (
        result["status"]
        == "journal_not_completed"
    )

    feature_sets = (
        db_session.query(
            JournalFeatureSet
        )
        .filter(
            JournalFeatureSet.journal_id
            == journal.id
        )
        .all()
    )

    assert feature_sets == []