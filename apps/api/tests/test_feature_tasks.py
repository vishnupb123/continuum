from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import func, select

import app.tasks.feature_tasks as feature_task_module

from app.models.feature_constants import (
    FEATURE_STATUS_COMPLETED,
    FEATURE_STATUS_FAILED,
    FEATURE_STATUS_PENDING,
)
from app.models.journal import JournalEntry
from app.models.journal_audio import JournalAudio
from app.models.journal_feature_set import (
    JournalFeatureSet,
)
from app.models.text_feature import TextFeature
from app.models.user import User
from app.services.features.feature_sets import (
    resolve_feature_set,
)
from app.services.features.fingerprinting import (
    calculate_journal_source_hash,
    fingerprint_text_journal,
)
from app.services.features.text_encoder.base import (
    TextEncodingResult,
)
from app.tasks.feature_tasks import (
    extract_journal_audio_features,
    extract_journal_text_features,
)
from tests.conftest import TestingSessionLocal


# ============================================================
# TEST HELPERS
# ============================================================


def make_completed_text_journal(
    db_session,
):
    user = User(
        email="feature-task-text@example.com",
        password_hash="test-password-hash",
        display_name="Feature Task Text User",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="TEXT",
        raw_text=(
            "This journal has enough text "
            "for feature extraction."
        ),
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.commit()
    db_session.refresh(journal)

    return journal


def make_completed_voice_journal(
    db_session,
):
    user = User(
        email="feature-task-voice@example.com",
        password_hash="test-password-hash",
        display_name="Feature Task Voice User",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="VOICE",
        raw_text=(
            "This is the transcript for a completed "
            "voice journal used in orchestration tests."
        ),
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.flush()

    audio = JournalAudio(
        journal_id=journal.id,
        storage_key=(
            "tests/feature-task-voice.wav"
        ),
        original_filename=(
            "feature-task-voice.wav"
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


def make_feature_set(
    db_session,
    *,
    journal,
    source_hash: str,
):
    resolution = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash=source_hash,
    )

    db_session.commit()

    return resolution.feature_set


def make_text_encoding_result():
    embedding = [0.0] * 768
    embedding[0] = 1.0

    return TextEncodingResult(
        embedding=tuple(embedding),
        dimension=768,
        encoder_name=(
            "sentence-transformers/"
            "all-mpnet-base-v2"
        ),
        encoder_version=(
            "text-encoder-v1"
        ),
        encoder_revision=(
            "e8c3b32edf5434bc2275fc9bab85f82640a19130"
        ),
        normalized=True,
    )


# ============================================================
# TEXT FAILURE ISOLATION
# ============================================================


@patch(
    "app.tasks.feature_tasks.SessionLocal"
)
@patch(
    "app.tasks.feature_tasks."
    "extract_text_features"
)
def test_text_feature_failure_does_not_fail_journal(
    extract_mock,
    session_local_mock,
    db_session,
):
    session_local_mock.return_value = (
        db_session
    )

    journal = make_completed_text_journal(
        db_session
    )

    # Capture UUID before the Celery task closes
    # the patched SQLAlchemy session.
    journal_id = journal.id

    feature_set = make_feature_set(
        db_session,
        journal=journal,
        source_hash=(
            "text-failure-source-hash"
        ),
    )

    feature_set_id = feature_set.id

    extract_mock.side_effect = RuntimeError(
        "forced encoder failure"
    )

    with pytest.raises(
        RuntimeError,
        match="forced encoder failure",
    ):
        extract_journal_text_features.run(
            str(journal_id),
            str(feature_set_id),
        )

    db_session.expire_all()

    persisted_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    assert persisted_journal is not None

    # Critical invariant:
    # M3 failure must never corrupt successful
    # M2 journal processing.
    assert (
        persisted_journal.status
        == "COMPLETED"
    )

    persisted_feature_set = db_session.get(
        JournalFeatureSet,
        feature_set_id,
    )

    assert persisted_feature_set is not None

    assert (
        persisted_feature_set.status
        == FEATURE_STATUS_FAILED
    )

    assert (
        persisted_feature_set.text_feature
        is None
    )

    assert (
        persisted_feature_set.error_message
        == (
            "Text feature extraction failed: "
            "RuntimeError"
        )
    )

    extract_mock.assert_called_once()


# ============================================================
# TEXT RETRY / GENERATION REUSE
# ============================================================


@patch(
    "app.tasks.feature_tasks.SessionLocal"
)
@patch(
    "app.services.features.text_features."
    "get_text_encoder"
)
def test_failed_text_generation_recovers_using_same_feature_set(
    get_text_encoder_mock,
    session_local_mock,
    db_session,
):
    session_local_mock.return_value = (
        db_session
    )

    journal = make_completed_text_journal(
        db_session
    )

    # Capture IDs before task execution because the
    # task owns/closes the patched session.
    journal_id = journal.id

    feature_set = make_feature_set(
        db_session,
        journal=journal,
        source_hash=(
            "text-recovery-source-hash"
        ),
    )

    feature_set_id = feature_set.id

    encoder = (
        get_text_encoder_mock.return_value
    )

    # --------------------------------------------------------
    # FIRST ATTEMPT — FORCE FAILURE
    # --------------------------------------------------------

    encoder.encode.side_effect = RuntimeError(
        "forced encoder failure"
    )

    with pytest.raises(
        RuntimeError,
        match="forced encoder failure",
    ):
        extract_journal_text_features.run(
            str(journal_id),
            str(feature_set_id),
        )

    db_session.expire_all()

    failed_feature_set = db_session.get(
        JournalFeatureSet,
        feature_set_id,
    )

    assert failed_feature_set is not None

    assert (
        failed_feature_set.status
        == FEATURE_STATUS_FAILED
    )

    assert (
        failed_feature_set.text_feature
        is None
    )

    # --------------------------------------------------------
    # PARENT-STYLE FAILED -> PENDING RECOVERY
    # --------------------------------------------------------

    resolution = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash=(
            "text-recovery-source-hash"
        ),
    )

    assert (
        resolution.feature_set.id
        == feature_set_id
    )

    assert (
        resolution.should_process
        is True
    )

    assert (
        resolution.feature_set.status
        == FEATURE_STATUS_PENDING
    )

    db_session.commit()

    # --------------------------------------------------------
    # SECOND ATTEMPT — ENCODER RECOVERS
    # --------------------------------------------------------

    encoder.encode.side_effect = None

    encoder.encode.return_value = (
        make_text_encoding_result()
    )

    result = (
        extract_journal_text_features.run(
            str(journal_id),
            str(feature_set_id),
        )
    )

    assert (
        result["status"]
        == "completed"
    )

    db_session.expire_all()

    recovered = db_session.get(
        JournalFeatureSet,
        feature_set_id,
    )

    assert recovered is not None

    # Critical invariant:
    # retry uses the exact same generation.
    assert (
        recovered.id
        == feature_set_id
    )

    assert (
        recovered.status
        == FEATURE_STATUS_COMPLETED
    )

    assert (
        recovered.error_message
        is None
    )

    assert (
        recovered.completed_at
        is not None
    )

    assert (
        recovered.text_feature
        is not None
    )

    assert (
        recovered.text_feature.embedding_dimension
        == 768
    )

    assert (
        recovered.text_feature.encoder_version
        == "text-encoder-v1"
    )

    assert (
        recovered.text_feature.encoder_revision
        == (
            "e8c3b32edf5434bc2275fc9bab85f82640a19130"
        )
    )

    persisted_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    assert persisted_journal is not None

    assert (
        persisted_journal.status
        == "COMPLETED"
    )


# ============================================================
# DUPLICATE TEXT DELIVERY
# ============================================================


@patch(
    "app.tasks.feature_tasks.SessionLocal"
)
@patch(
    "app.services.features.text_features."
    "get_text_encoder"
)
def test_duplicate_text_delivery_does_not_reextract(
    get_text_encoder_mock,
    session_local_mock,
    db_session,
):
    session_local_mock.return_value = (
        db_session
    )

    journal = make_completed_text_journal(
        db_session
    )

    journal_id = journal.id

    feature_set = make_feature_set(
        db_session,
        journal=journal,
        source_hash=(
            "duplicate-text-source-hash"
        ),
    )

    feature_set_id = feature_set.id

    encoder = (
        get_text_encoder_mock.return_value
    )

    encoder.encode.return_value = (
        make_text_encoding_result()
    )

    first = (
        extract_journal_text_features.run(
            str(journal_id),
            str(feature_set_id),
        )
    )

    assert (
        first["status"]
        == "completed"
    )

    # First task invocation closes the patched
    # session. The next invocation reuses the
    # Session object through SessionLocal mock,
    # which SQLAlchemy supports.
    second = (
        extract_journal_text_features.run(
            str(journal_id),
            str(feature_set_id),
        )
    )

    assert (
        second["status"]
        == "already_extracted"
    )

    # Most important assertion:
    # duplicate Celery delivery must not invoke
    # the expensive encoder twice.
    assert (
        encoder.encode.call_count
        == 1
    )

    db_session.expire_all()

    persisted = db_session.get(
        JournalFeatureSet,
        feature_set_id,
    )

    assert persisted is not None

    assert (
        persisted.status
        == FEATURE_STATUS_COMPLETED
    )

    assert (
        persisted.text_feature
        is not None
    )


# ============================================================
# CHILD GENERATION OWNERSHIP
# ============================================================


@patch(
    "app.tasks.feature_tasks.SessionLocal"
)
def test_text_child_rejects_feature_set_from_another_journal(
    session_local_mock,
    db_session,
):
    session_local_mock.return_value = (
        db_session
    )

    first_journal = (
        make_completed_text_journal(
            db_session
        )
    )

    first_journal_id = (
        first_journal.id
    )

    second_user = User(
        email=(
            "feature-task-other@example.com"
        ),
        password_hash=(
            "test-password-hash"
        ),
    )

    db_session.add(second_user)
    db_session.flush()

    second_journal = JournalEntry(
        user_id=second_user.id,
        entry_type="TEXT",
        raw_text=(
            "This belongs to another journal."
        ),
        status="COMPLETED",
    )

    db_session.add(second_journal)
    db_session.commit()

    second_journal_id = (
        second_journal.id
    )

    # Reacquire the first journal because the helper
    # below expects an ORM object.
    first_journal = db_session.get(
        JournalEntry,
        first_journal_id,
    )

    assert first_journal is not None

    feature_set = make_feature_set(
        db_session,
        journal=first_journal,
        source_hash=(
            "ownership-source-hash"
        ),
    )

    feature_set_id = feature_set.id

    with pytest.raises(
        ValueError,
        match=(
            "Feature set does not belong "
            "to journal"
        ),
    ):
        extract_journal_text_features.run(
            str(second_journal_id),
            str(feature_set_id),
        )


# ============================================================
# AUDIO FAILURE ISOLATION
# ============================================================


@patch(
    "app.tasks.feature_tasks.SessionLocal"
)
@patch(
    "app.tasks.feature_tasks."
    "extract_audio_features"
)
def test_audio_feature_failure_does_not_fail_journal(
    extract_mock,
    session_local_mock,
    db_session,
):
    session_local_mock.return_value = (
        db_session
    )

    journal = make_completed_voice_journal(
        db_session
    )

    # Capture before task invocation.
    journal_id = journal.id

    feature_set = make_feature_set(
        db_session,
        journal=journal,
        source_hash=(
            "audio-failure-source-hash"
        ),
    )

    feature_set_id = feature_set.id

    extract_mock.side_effect = RuntimeError(
        "forced audio encoder failure"
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "forced audio encoder failure"
        ),
    ):
        extract_journal_audio_features.run(
            str(journal_id),
            str(feature_set_id),
        )

    db_session.expire_all()

    persisted_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    persisted_feature_set = db_session.get(
        JournalFeatureSet,
        feature_set_id,
    )

    assert persisted_journal is not None
    assert persisted_feature_set is not None

    # M3 failure is isolated from the completed
    # M2 journal lifecycle.
    assert (
        persisted_journal.status
        == "COMPLETED"
    )

    assert (
        persisted_feature_set.status
        == FEATURE_STATUS_FAILED
    )

    assert (
        persisted_feature_set.audio_feature
        is None
    )

    assert (
        persisted_feature_set.error_message
        == (
            "Audio feature extraction failed: "
            "RuntimeError"
        )
    )
    
        # Internal exception details must not be persisted.
    assert (
        "forced audio encoder failure"
        not in persisted_feature_set.error_message
    )

    extract_mock.assert_called_once()



def test_corrupt_audio_fails_without_persisting_feature(
    db_session,
):
    """
    Corrupt/undecodable stored audio must fail safely.

    Unlike UNUSABLE audio, this is an execution failure during
    M3.4 preprocessing rather than a quality-policy decision.

    It must:
        - raise AudioPreprocessingError
        - not persist AudioFeature
        - mark the exact generation FAILED
        - persist only a safe failure classification
        - leave the completed M2 JournalEntry untouched
    """

    from app.services.features.audio_preprocessing import (
        AudioPreprocessingError,
    )

    journal = make_completed_voice_journal(
        db_session
    )

    journal_id = journal.id

    feature_set = make_feature_set(
        db_session,
        journal=journal,
        source_hash=(
            "m39-corrupt-audio-source-hash"
        ),
    )

    feature_set_id = feature_set.id

    storage_mock = Mock()

    # Deliberately not a valid encoded audio container.
    storage_mock.get.return_value = (
        b"this-is-not-valid-audio"
    )

    with patch.object(
        feature_task_module,
        "SessionLocal",
        side_effect=TestingSessionLocal,
    ):
        with patch(
            "app.services.features.audio_features."
            "get_audio_storage",
            return_value=storage_mock,
        ):
            with pytest.raises(
                AudioPreprocessingError,
                match="audio could not be decoded",
            ):
                extract_journal_audio_features.run(
                    str(journal_id),
                    str(feature_set_id),
                )

    storage_mock.get.assert_called_once_with(
        "tests/feature-task-voice.wav"
    )

    db_session.expire_all()

    persisted_feature_set = db_session.get(
        JournalFeatureSet,
        feature_set_id,
    )

    assert persisted_feature_set is not None

    assert (
        persisted_feature_set.status
        == FEATURE_STATUS_FAILED
    )

    assert (
        persisted_feature_set.audio_feature
        is None
    )

    # The persisted message exposes only the failure class.
    # It must not contain decoder internals or private audio data.
    assert (
        persisted_feature_set.error_message
        == (
            "Audio feature extraction failed: "
            "AudioPreprocessingError"
        )
    )
        # -----------------------------------------------------
    # PRIVACY CONTRACT — M3.9D
    # -----------------------------------------------------

    error_message = (
        persisted_feature_set.error_message
    )

    assert error_message is not None

    # Storage identifiers must not leak into persisted
    # feature-generation failure metadata.
    assert (
        "tests/feature-task-voice.wav"
        not in error_message
    )

    # Raw/corrupt audio content must not leak either.
    assert (
        "this-is-not-valid-audio"
        not in error_message
    )

    # Decoder implementation details must not be persisted.
    assert (
        "audio could not be decoded"
        not in error_message
    )

    # Only the safe exception classification survives.
    assert (
        "AudioPreprocessingError"
        in error_message
    )

    persisted_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    assert persisted_journal is not None

    # M3 remains downstream of the already-successful M2
    # journal lifecycle.
    assert (
        persisted_journal.status
        == "COMPLETED"
    )


# ============================================================
# PARTIAL VOICE FAILURE / RECOVERY
# ============================================================


@patch(
    "app.tasks.feature_tasks.SessionLocal"
)
@patch(
    "app.tasks.feature_tasks."
    "extract_audio_features"
)
@patch(
    "app.services.features.text_features."
    "get_text_encoder"
)
def test_voice_partial_failure_reuses_text_and_recovers_audio(
    get_text_encoder_mock,
    extract_audio_mock,
    session_local_mock,
    db_session,
):
    """
    Critical M3.7 recovery contract:

        text succeeds
            +
        audio fails
            ↓
        generation FAILED
        TextFeature preserved
            ↓
        parent-style FAILED -> PENDING reset
            ↓
        duplicate text child short-circuits
            +
        audio retry succeeds
            ↓
        generation COMPLETED
    """

    session_local_mock.return_value = (
        db_session
    )

    journal = make_completed_voice_journal(
        db_session
    )

    journal_id = journal.id

    feature_set = make_feature_set(
        db_session,
        journal=journal,
        source_hash=(
            "voice-partial-recovery-hash"
        ),
    )

    feature_set_id = feature_set.id

    encoder = (
        get_text_encoder_mock.return_value
    )

    encoder.encode.return_value = (
        make_text_encoding_result()
    )

    # --------------------------------------------------------
    # TEXT SUCCEEDS FIRST
    # --------------------------------------------------------

    text_result = (
        extract_journal_text_features.run(
            str(journal_id),
            str(feature_set_id),
        )
    )

    assert (
        text_result["status"]
        == "completed"
    )

    db_session.expire_all()

    after_text = db_session.get(
        JournalFeatureSet,
        feature_set_id,
    )

    assert after_text is not None

    # VOICE requires both modalities.
    assert (
        after_text.status
        != FEATURE_STATUS_COMPLETED
    )

    assert (
        after_text.text_feature
        is not None
    )

    assert (
        after_text.audio_feature
        is None
    )

    # --------------------------------------------------------
    # AUDIO FAILS
    # --------------------------------------------------------

    extract_audio_mock.side_effect = (
        RuntimeError(
            "forced audio failure"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="forced audio failure",
    ):
        extract_journal_audio_features.run(
            str(journal_id),
            str(feature_set_id),
        )

    db_session.expire_all()

    failed = db_session.get(
        JournalFeatureSet,
        feature_set_id,
    )

    assert failed is not None

    assert (
        failed.status
        == FEATURE_STATUS_FAILED
    )

    # Successful modality must survive.
    assert (
        failed.text_feature
        is not None
    )

    assert (
        failed.audio_feature
        is None
    )

    # --------------------------------------------------------
    # PARENT RECOVERS SAME GENERATION
    # --------------------------------------------------------

    resolution = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash=(
            "voice-partial-recovery-hash"
        ),
    )

    assert (
        resolution.feature_set.id
        == feature_set_id
    )

    assert (
        resolution.feature_set.status
        == FEATURE_STATUS_PENDING
    )

    db_session.commit()

    # --------------------------------------------------------
    # TEXT CHILD IS REDELIVERED
    # --------------------------------------------------------

    duplicate_text = (
        extract_journal_text_features.run(
            str(journal_id),
            str(feature_set_id),
        )
    )

    assert (
        duplicate_text["status"]
        == "already_extracted"
    )

    # Encoder ran only during original delivery.
    assert (
        encoder.encode.call_count
        == 1
    )

    # --------------------------------------------------------
    # AUDIO RECOVERS
    # --------------------------------------------------------
    #
    # This test is about orchestration semantics.
    # Real M3.4/M3.5/M3.6 audio extraction has its own
    # acceptance coverage.
    # --------------------------------------------------------

    def successful_audio_extraction(
        db,
        *,
        journal,
        feature_set,
    ):
        from app.models.audio_feature import (
            AudioFeature,
        )
        from app.services.features.feature_sets import (
            complete_feature_set_if_ready,
        )

        audio_feature = AudioFeature(
            preprocessing_version=(
                "audio-preprocess-v1"
            ),
            encoder_name=(
                "microsoft/wavlm-base-plus"
            ),
            encoder_version=(
                "audio-encoder-v1"
            ),
            encoder_revision=(
                "4c66d4806a428f2e922ccfa1a962776e232d487b"
            ),
            embedding_dimension=768,
            embedding=(
                [1.0]
                + [0.0] * 767
            ),
            duration_seconds=2.0,
            speech_ratio=None,
            sample_rate_hz=16_000,
            quality_status="GOOD",
            feature_metadata={},
        )

        feature_set.audio_feature = (
            audio_feature
        )

        db.flush()

        complete_feature_set_if_ready(
            feature_set,
            entry_type=journal.entry_type,
        )

        db.flush()

        class Result:
            pass

        result = Result()
        result.feature_set = (
            feature_set
        )

        return result

    extract_audio_mock.side_effect = (
        successful_audio_extraction
    )

    audio_result = (
        extract_journal_audio_features.run(
            str(journal_id),
            str(feature_set_id),
        )
    )

    assert (
        audio_result["status"]
        == "completed"
    )

    db_session.expire_all()

    recovered = db_session.get(
        JournalFeatureSet,
        feature_set_id,
    )

    assert recovered is not None

    assert (
        recovered.status
        == FEATURE_STATUS_COMPLETED
    )

    assert (
        recovered.text_feature
        is not None
    )

    assert (
        recovered.audio_feature
        is not None
    )

    assert (
        recovered.error_message
        is None
    )

    assert (
        recovered.completed_at
        is not None
    )

    persisted_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    assert persisted_journal is not None

    assert (
        persisted_journal.status
        == "COMPLETED"
    )


# ============================================================
# CONCURRENT TEXT CHILD DELIVERY — M3.8D
# ============================================================


def test_concurrent_text_children_converge_on_one_feature(
    db_session,
):
    """
    Two Celery workers processing the exact same text
    modality concurrently must converge on one persisted
    TextFeature.

    The encoder is mocked deliberately. This test targets
    task/database concurrency, not concurrent model loading.

    Duplicate computation is acceptable at M3.8.

    Duplicate persistence, generation corruption, or an
    escaped unique-constraint failure is not.
    """

    user = User(
        email="m38-text-concurrency@example.com",
        password_hash="test-password-hash",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="TEXT",
        raw_text=(
            "Concurrent child extraction must converge "
            "on one persisted text representation."
        ),
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.flush()

    source_hash = fingerprint_text_journal(
        journal.raw_text
    ).value

    resolution = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash=source_hash,
    )

    journal_id = journal.id
    feature_set_id = resolution.feature_set.id

    db_session.commit()

    extraction_barrier = Barrier(2)

    original_extract = (
        feature_task_module.extract_text_features
    )

    def synchronized_extract(
        db,
        *,
        journal,
        feature_set,
    ):
        # Both workers have already passed the child task's
        # "text_feature is None" guard before either is
        # allowed to enter persistence.
        extraction_barrier.wait(
            timeout=5
        )

        return original_extract(
            db,
            journal=journal,
            feature_set=feature_set,
        )

    mock_encoded = Mock()
    mock_encoded.encoder_name = "test-encoder"
    mock_encoded.encoder_version = "text-encoder-v1"
    mock_encoded.encoder_revision = "test-revision"
    mock_encoded.dimension = 768
    mock_encoded.embedding = [0.0] * 768
    mock_encoded.normalized = True

    mock_encoder = Mock()
    mock_encoder.encode.return_value = (
        mock_encoded
    )

    def worker():
        # Each simulated Celery worker receives an
        # independent SQLAlchemy session/transaction.
        worker_session = TestingSessionLocal()

        try:
            with patch.object(
                feature_task_module,
                "SessionLocal",
                return_value=worker_session,
            ):
                return (
                    extract_journal_text_features.run(
                        str(journal_id),
                        str(feature_set_id),
                    )
                )
        finally:
            worker_session.close()

    # Mock only model inference. PostgreSQL persistence,
    # SQLAlchemy flush/commit/rollback and the UNIQUE
    # constraint remain real.
    with patch(
        "app.services.features.text_features."
        "get_text_encoder",
        return_value=mock_encoder,
    ):
        with patch.object(
            feature_task_module,
            "extract_text_features",
            side_effect=synchronized_extract,
        ):
            with ThreadPoolExecutor(
                max_workers=2
            ) as executor:
                futures = [
                    executor.submit(worker)
                    for _ in range(2)
                ]

                results = [
                    future.result(timeout=15)
                    for future in futures
                ]

    db_session.expire_all()

    count = db_session.scalar(
        select(
            func.count(
                TextFeature.id
            )
        ).where(
            TextFeature.feature_set_id
            == feature_set_id
        )
    )

    assert count == 1

    assert {
        result["status"]
        for result in results
    } <= {
        "completed",
        "already_extracted",
    }

    feature_set = db_session.get(
        JournalFeatureSet,
        feature_set_id,
    )

    assert feature_set is not None

    assert (
        feature_set.status
        == FEATURE_STATUS_COMPLETED
    )
    
def test_concurrent_audio_children_converge_on_one_feature(
    db_session,
):
    """
    Two Celery workers processing the exact same audio
    modality concurrently must converge on one persisted
    AudioFeature.

    Expensive M3.4/M3.5/M3.6 computation is replaced with
    deterministic test extraction.

    PostgreSQL persistence, transaction handling and the
    UNIQUE(feature_set_id) constraint remain real.
    """

    from app.models.audio_feature import AudioFeature
    from app.services.features.feature_sets import (
        complete_feature_set_if_ready,
    )

    user = User(
        email="m38-audio-concurrency@example.com",
        password_hash="test-password-hash",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="VOICE",
        raw_text=(
            "Concurrent audio child extraction "
            "test transcript."
        ),
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.flush()

    audio = JournalAudio(
        journal_id=journal.id,
        storage_key=(
            "tests/m38-audio-concurrency.wav"
        ),
        original_filename=(
            "m38-audio-concurrency.wav"
        ),
        mime_type="audio/wav",
        size_bytes=123,
        transcription_status=(
            "TRANSCRIPTION_COMPLETED"
        ),
    )

    db_session.add(audio)
    db_session.flush()

    resolution = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash=(
            "m38-audio-concurrency-source-hash"
        ),
    )

    journal_id = journal.id
    feature_set_id = resolution.feature_set.id

    db_session.commit()

    extraction_barrier = Barrier(2)

    def synchronized_audio_extraction(
        db,
        *,
        journal,
        feature_set,
    ):
        """
        Both workers reach this function only after their
        task-level audio_feature-is-None checks.

        The barrier guarantees both workers enter the
        persistence race.
        """

        extraction_barrier.wait(
            timeout=5
        )

        audio_feature = AudioFeature(
            preprocessing_version=(
                "audio-preprocess-v1"
            ),
            encoder_name=(
                "microsoft/wavlm-base-plus"
            ),
            encoder_version=(
                "audio-encoder-v1"
            ),
            encoder_revision=(
                "4c66d4806a428f2e922ccfa1a962776e232d487b"
            ),
            embedding_dimension=768,
            embedding=(
                [1.0]
                + [0.0] * 767
            ),
            duration_seconds=2.0,
            speech_ratio=None,
            sample_rate_hz=16_000,
            quality_status="GOOD",
            feature_metadata={
                "test": (
                    "m3.8d-concurrent-audio"
                ),
            },
        )

        feature_set.audio_feature = (
            audio_feature
        )

        # This is the actual persistence race.
        db.flush()

        complete_feature_set_if_ready(
            feature_set,
            entry_type=journal.entry_type,
        )

        db.flush()

        class Result:
            pass

        result = Result()

        result.feature_set = (
            feature_set
        )

        return result

    def worker():
        worker_session = TestingSessionLocal()

        try:
            with patch.object(
                feature_task_module,
                "SessionLocal",
                return_value=worker_session,
            ):
                return (
                    extract_journal_audio_features.run(
                        str(journal_id),
                        str(feature_set_id),
                    )
                )
        finally:
            worker_session.close()

    with patch.object(
        feature_task_module,
        "extract_audio_features",
        side_effect=(
            synchronized_audio_extraction
        ),
    ):
        with ThreadPoolExecutor(
            max_workers=2
        ) as executor:
            futures = [
                executor.submit(worker)
                for _ in range(2)
            ]

            results = [
                future.result(timeout=15)
                for future in futures
            ]

    db_session.expire_all()

    count = db_session.scalar(
        select(
            func.count(
                AudioFeature.id
            )
        ).where(
            AudioFeature.feature_set_id
            == feature_set_id
        )
    )

    assert count == 1

    assert {
        result["status"]
        for result in results
    } <= {
        "completed",
        "already_extracted",
    }

    feature_set = db_session.get(
        JournalFeatureSet,
        feature_set_id,
    )

    assert feature_set is not None

    assert (
        feature_set.audio_feature
        is not None
    )
    
def test_concurrent_failing_text_child_cannot_downgrade_completed_generation(
    db_session,
):
    """
    A stale duplicate worker that fails after another worker
    successfully completes the same generation must not
    downgrade COMPLETED -> FAILED.

    Timeline:

        Worker A loads incomplete generation.
        Worker B loads incomplete generation.

        Both pass the task-level duplicate guard.

        Worker B waits inside extraction.

        Worker A persists TextFeature and commits COMPLETED.

        Worker B then raises.

        Worker B's failure handler reloads the generation.

        COMPLETED must remain terminal.
    """

    from threading import Event

    user = User(
        email="m39-concurrent-failure@example.com",
        password_hash="test-password-hash",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="TEXT",
        raw_text=(
            "Concurrent stale worker failure must never "
            "downgrade a successfully completed generation."
        ),
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.flush()

    source_hash = fingerprint_text_journal(
        journal.raw_text
    ).value

    resolution = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash=source_hash,
    )

    journal_id = journal.id
    feature_set_id = resolution.feature_set.id

    db_session.commit()

    # -----------------------------------------------------
    # SYNCHRONIZATION
    # -----------------------------------------------------
    #
    # Both workers must enter extraction before either
    # finishes.
    #
    # The successful worker then commits first.
    # Only after that commit may the stale worker fail.
    # -----------------------------------------------------

    both_inside_extraction = Barrier(2)

    winner_committed = Event()

    original_extract = (
        feature_task_module.extract_text_features
    )

    mock_encoded = Mock()
    mock_encoded.encoder_name = "test-encoder"
    mock_encoded.encoder_version = "text-encoder-v1"
    mock_encoded.encoder_revision = "test-revision"
    mock_encoded.dimension = 768
    mock_encoded.embedding = [0.0] * 768
    mock_encoded.normalized = True

    mock_encoder = Mock()
    mock_encoder.encode.return_value = (
        mock_encoded
    )

    def synchronized_extract(
        db,
        *,
        journal,
        feature_set,
    ):
        # Both workers have already passed:
        #
        #     feature_set.text_feature is None
        #
        # before either continues.
        both_inside_extraction.wait(
            timeout=5
        )

        # Assign deterministic roles using the SQLAlchemy
        # Session object itself.
        #
        # The first session reaching this branch becomes
        # the winner. The second becomes the stale failing
        # worker.
        if not hasattr(
            synchronized_extract,
            "winner_session",
        ):
            synchronized_extract.winner_session = db

        if (
            synchronized_extract.winner_session
            is db
        ):
            return original_extract(
                db,
                journal=journal,
                feature_set=feature_set,
            )

        # Loser must not fail until the winner's task has
        # committed the COMPLETED generation.
        if not winner_committed.wait(
            timeout=10
        ):
            raise TimeoutError(
                "Winning worker did not commit in time"
            )

        raise RuntimeError(
            "simulated stale worker failure"
        )

    def worker():
        worker_session = TestingSessionLocal()

        try:
            return (
                extract_journal_text_features.run(
                    str(journal_id),
                    str(feature_set_id),
                )
            )
        finally:
            worker_session.close()

    # SessionLocal is patched once for the entire concurrent
    # run. Every task invocation receives its own real
    # SQLAlchemy Session.
    #
    # This avoids overlapping thread-local patch contexts
    # mutating the same module attribute.
    with patch.object(
        feature_task_module,
        "SessionLocal",
        side_effect=TestingSessionLocal,
    ):
        with patch(
            "app.services.features.text_features."
            "get_text_encoder",
            return_value=mock_encoder,
        ):
            with patch.object(
                feature_task_module,
                "extract_text_features",
                side_effect=synchronized_extract,
            ):
                with ThreadPoolExecutor(
                    max_workers=2
                ) as executor:
                    futures = [
                        executor.submit(worker)
                        for _ in range(2)
                    ]

                    first_exception = None
                    successful_result = None

                    # One worker must complete. Once it has
                    # returned, its task-level db.commit()
                    # has already happened.
                    while (
                        successful_result is None
                    ):
                        for future in futures:
                            if not future.done():
                                continue

                            try:
                                result = future.result()
                            except RuntimeError as exc:
                                first_exception = exc
                                continue

                            successful_result = result
                            break

                    winner_committed.set()

                    results = []

                    for future in futures:
                        try:
                            results.append(
                                future.result(
                                    timeout=10
                                )
                            )
                        except RuntimeError as exc:
                            first_exception = exc

    # -----------------------------------------------------
    # EXPECTED WORKER OUTCOMES
    # -----------------------------------------------------

    assert successful_result is not None

    assert (
        successful_result["status"]
        == "completed"
    )

    assert first_exception is not None

    assert (
        str(first_exception)
        == "simulated stale worker failure"
    )

    # -----------------------------------------------------
    # FINAL DATABASE INVARIANT
    # -----------------------------------------------------

    db_session.expire_all()

    final_feature_set = db_session.get(
        JournalFeatureSet,
        feature_set_id,
    )

    assert final_feature_set is not None

    # THIS is the M3.9 invariant we expect to be RED with
    # the current production implementation.
    assert (
        final_feature_set.status
        == FEATURE_STATUS_COMPLETED
    )

    assert (
        final_feature_set.error_message
        is None
    )

    assert (
        final_feature_set.completed_at
        is not None
    )

    assert (
        final_feature_set.text_feature
        is not None
    )

    count = db_session.scalar(
        select(
            func.count(
                TextFeature.id
            )
        ).where(
            TextFeature.feature_set_id
            == feature_set_id
        )
    )

    assert count == 1
    

def test_concurrent_failing_audio_child_cannot_downgrade_completed_generation(
    db_session,
):
    """
    A stale duplicate AUDIO worker that fails after another
    worker completes the same VOICE generation must not
    downgrade COMPLETED -> FAILED.

    The TextFeature is pre-created because VOICE requires
    both TEXT and AUDIO before the generation can complete.
    """

    from threading import Event

    from app.models.audio_feature import AudioFeature
    from app.services.features.feature_sets import (
        complete_feature_set_if_ready,
    )

    user = User(
        email="m39-concurrent-audio-failure@example.com",
        password_hash="test-password-hash",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="VOICE",
        raw_text=(
            "This transcript already has its text feature "
            "before the concurrent audio race begins."
        ),
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.flush()

    audio = JournalAudio(
        journal_id=journal.id,
        storage_key=(
            "tests/m39-concurrent-audio-failure.wav"
        ),
        original_filename=(
            "m39-concurrent-audio-failure.wav"
        ),
        mime_type="audio/wav",
        size_bytes=123,
        transcription_status=(
            "TRANSCRIPTION_COMPLETED"
        ),
    )

    db_session.add(audio)
    db_session.flush()

    resolution = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash=(
            "m39-concurrent-audio-failure-source-hash"
        ),
    )

    feature_set = resolution.feature_set

    # VOICE needs both modalities. Pre-create TEXT so the
    # winning AUDIO worker can complete the generation.
    text_feature = TextFeature(
        feature_set_id=feature_set.id,
        source_type="TRANSCRIPT",
        preprocessing_version="text-preprocess-v1",
        encoder_name="test-encoder",
        encoder_revision="test-revision",
        encoder_version="text-encoder-v1",
        embedding_dimension=768,
        embedding=[0.0] * 768,
        word_count=12,
        character_count=80,
        quality_status="GOOD",
        feature_metadata={
            "test": "m3.9-audio-stale-failure",
        },
    )

    db_session.add(text_feature)
    db_session.flush()

    journal_id = journal.id
    feature_set_id = feature_set.id

    db_session.commit()

    both_inside_extraction = Barrier(2)
    winner_committed = Event()

    def synchronized_audio_extraction(
        db,
        *,
        journal,
        feature_set,
    ):
        # Both tasks have already observed that AudioFeature
        # is absent before either worker continues.
        both_inside_extraction.wait(
            timeout=5
        )

        if not hasattr(
            synchronized_audio_extraction,
            "winner_session",
        ):
            synchronized_audio_extraction.winner_session = db

        if (
            synchronized_audio_extraction.winner_session
            is db
        ):
            audio_feature = AudioFeature(
                preprocessing_version=(
                    "audio-preprocess-v1"
                ),
                encoder_name=(
                    "microsoft/wavlm-base-plus"
                ),
                encoder_version=(
                    "audio-encoder-v1"
                ),
                encoder_revision=(
                    "4c66d4806a428f2e922ccfa1a962776e232d487b"
                ),
                embedding_dimension=768,
                embedding=(
                    [1.0]
                    + [0.0] * 767
                ),
                duration_seconds=2.0,
                speech_ratio=None,
                sample_rate_hz=16_000,
                quality_status="GOOD",
                feature_metadata={
                    "test": (
                        "m3.9-concurrent-audio-failure"
                    ),
                },
            )

            feature_set.audio_feature = (
                audio_feature
            )

            db.flush()

            completed = (
                complete_feature_set_if_ready(
                    feature_set,
                    entry_type=journal.entry_type,
                )
            )

            assert completed is True

            db.flush()

            class Result:
                pass

            result = Result()
            result.feature_set = feature_set

            return result

        # The stale worker may fail only after the winning
        # task has committed COMPLETED.
        if not winner_committed.wait(
            timeout=10
        ):
            raise TimeoutError(
                "Winning audio worker did not commit in time"
            )

        raise RuntimeError(
            "simulated stale audio worker failure"
        )

    def worker():
        return (
            extract_journal_audio_features.run(
                str(journal_id),
                str(feature_set_id),
            )
        )

    with patch.object(
        feature_task_module,
        "SessionLocal",
        side_effect=TestingSessionLocal,
    ):
        with patch.object(
            feature_task_module,
            "extract_audio_features",
            side_effect=synchronized_audio_extraction,
        ):
            with ThreadPoolExecutor(
                max_workers=2
            ) as executor:
                futures = [
                    executor.submit(worker)
                    for _ in range(2)
                ]

                first_exception = None
                successful_result = None

                while successful_result is None:
                    for future in futures:
                        if not future.done():
                            continue

                        try:
                            result = future.result()
                        except RuntimeError as exc:
                            first_exception = exc
                            continue

                        successful_result = result
                        break

                # Successful task has returned only after
                # its transaction committed.
                winner_committed.set()

                for future in futures:
                    try:
                        future.result(
                            timeout=10
                        )
                    except RuntimeError as exc:
                        first_exception = exc

    assert successful_result is not None

    assert (
        successful_result["status"]
        == "completed"
    )

    assert first_exception is not None

    assert (
        str(first_exception)
        == "simulated stale audio worker failure"
    )

    # -----------------------------------------------------
    # FINAL DATABASE INVARIANT
    # -----------------------------------------------------

    db_session.expire_all()

    final_feature_set = db_session.get(
        JournalFeatureSet,
        feature_set_id,
    )

    assert final_feature_set is not None

    assert (
        final_feature_set.status
        == FEATURE_STATUS_COMPLETED
    )

    assert (
        final_feature_set.error_message
        is None
    )

    assert (
        final_feature_set.completed_at
        is not None
    )

    assert (
        final_feature_set.text_feature
        is not None
    )

    assert (
        final_feature_set.audio_feature
        is not None
    )

    count = db_session.scalar(
        select(
            func.count(
                AudioFeature.id
            )
        ).where(
            AudioFeature.feature_set_id
            == feature_set_id
        )
    )

    assert count == 1
    
def test_degraded_text_is_persisted_and_generation_completes(
    db_session,
):
    """
    Very short text is weak signal, not invalid input.

    DEGRADED text must:
        - still be encoded
        - persist a TextFeature
        - preserve DEGRADED quality
        - complete a TEXT generation
    """

    user = User(
        email="m39-degraded-text@example.com",
        password_hash="test-password-hash",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="TEXT",
        raw_text="Bad day",
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.flush()

    source_hash = fingerprint_text_journal(
        journal.raw_text
    ).value

    resolution = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash=source_hash,
    )

    journal_id = journal.id
    feature_set_id = resolution.feature_set.id

    db_session.commit()

    mock_encoded = Mock()
    mock_encoded.encoder_name = "test-encoder"
    mock_encoded.encoder_version = "text-encoder-v1"
    mock_encoded.encoder_revision = "test-revision"
    mock_encoded.dimension = 768
    mock_encoded.embedding = [0.0] * 768
    mock_encoded.normalized = True

    mock_encoder = Mock()
    mock_encoder.encode.return_value = (
        mock_encoded
    )

    with patch.object(
        feature_task_module,
        "SessionLocal",
        side_effect=TestingSessionLocal,
    ):
        with patch(
            "app.services.features.text_features."
            "get_text_encoder",
            return_value=mock_encoder,
        ):
            result = (
                extract_journal_text_features.run(
                    str(journal_id),
                    str(feature_set_id),
                )
            )

    assert result["status"] == "completed"

    mock_encoder.encode.assert_called_once_with(
        "Bad day"
    )

    db_session.expire_all()

    persisted = db_session.get(
        JournalFeatureSet,
        feature_set_id,
    )

    assert persisted is not None

    assert (
        persisted.status
        == FEATURE_STATUS_COMPLETED
    )

    assert persisted.text_feature is not None

    assert (
        persisted.text_feature.quality_status
        == "DEGRADED"
    )

    assert (
        persisted.text_feature.word_count
        == 2
    )

    assert (
        persisted.text_feature.feature_metadata[
            "quality_reasons"
        ]
        == ["VERY_SHORT_TEXT"]
    )

    # M3 quality must not affect the already-completed
    # M2 journal lifecycle.
    persisted_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    assert persisted_journal is not None
    assert persisted_journal.status == "COMPLETED"
    
def test_unusable_text_fails_without_persisting_feature(
    db_session,
):
    """
    Text that becomes empty after preprocessing must not
    produce a semantic embedding.

    The generation fails safely while the M2 journal remains
    COMPLETED.
    """

    user = User(
        email="m39-unusable-text@example.com",
        password_hash="test-password-hash",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="TEXT",
        raw_text="   \t   \n   ",
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.flush()

    source_hash = fingerprint_text_journal(
        journal.raw_text
    ).value

    resolution = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash=source_hash,
    )

    journal_id = journal.id
    feature_set_id = resolution.feature_set.id

    db_session.commit()

    mock_encoder = Mock()

    with patch.object(
        feature_task_module,
        "SessionLocal",
        side_effect=TestingSessionLocal,
    ):
        with patch(
            "app.services.features.text_features."
            "get_text_encoder",
            return_value=mock_encoder,
        ):
            with pytest.raises(
                ValueError,
                match=(
                    "Journal text is unusable "
                    "after preprocessing"
                ),
            ):
                extract_journal_text_features.run(
                    str(journal_id),
                    str(feature_set_id),
                )

    # Critical quality invariant:
    # meaningless input never reaches the expensive model.
    mock_encoder.encode.assert_not_called()

    db_session.expire_all()

    persisted = db_session.get(
        JournalFeatureSet,
        feature_set_id,
    )

    assert persisted is not None

    assert (
        persisted.status
        == FEATURE_STATUS_FAILED
    )

    assert persisted.text_feature is None

    # Persisted failure exposes the failure class,
    # not raw private journal content.
    assert (
        persisted.error_message
        == (
            "Text feature extraction failed: "
            "ValueError"
        )
    )
    
    # -----------------------------------------------------
    # PRIVACY CONTRACT — M3.9D
    # -----------------------------------------------------

    error_message = persisted.error_message

    assert error_message is not None

    # Raw journal content must never be persisted in
    # feature-generation failure metadata.
    assert journal.raw_text not in error_message

    # Only the safe failure classification should survive.
    assert "ValueError" in error_message

    persisted_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    assert persisted_journal is not None

    assert (
        persisted_journal.status
        == "COMPLETED"
    )
    
    
def test_unusable_audio_fails_before_feature_extraction(
    db_session,
):
    """
    Structurally decodable audio may still be semantically
    unusable according to the audio quality policy.

    UNUSABLE audio must:
        - not reach acoustic feature extraction
        - not reach WavLM encoding
        - not persist AudioFeature
        - fail the feature generation safely
        - leave the M2 journal COMPLETED
    """

    user = User(
        email="m39-unusable-audio@example.com",
        password_hash="test-password-hash",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="VOICE",
        raw_text=(
            "Transcript remains valid even though "
            "the associated audio is unusable."
        ),
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.flush()

    audio = JournalAudio(
        journal_id=journal.id,
        storage_key="tests/m39-unusable-audio.wav",
        original_filename="m39-unusable-audio.wav",
        mime_type="audio/wav",
        size_bytes=123,
        transcription_status="TRANSCRIPTION_COMPLETED",
    )

    db_session.add(audio)
    db_session.flush()

    resolution = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash="m39-unusable-audio-source-hash",
    )

    journal_id = journal.id
    feature_set_id = resolution.feature_set.id

    db_session.commit()

    # -----------------------------------------------------
    # Deterministic UNUSABLE pipeline result
    # -----------------------------------------------------

    import numpy as np

    from app.services.features.audio_pipeline import (
        AudioPipelineResult,
    )
    from app.services.features.audio_quality import (
        AudioQualityMeasurements,
    )
    from app.services.features.audio_quality_policy import (
        AudioQualityAssessment,
    )

    pipeline_result = AudioPipelineResult(
        waveform=np.zeros(
            16_000,
            dtype=np.float32,
        ),
        sample_rate_hz=16_000,
        sample_count=16_000,
        duration_seconds=1.0,
        original_sample_rate_hz=16_000,
        original_channels=1,
        measurements=AudioQualityMeasurements(
            peak_amplitude=0.0,
            rms_amplitude=0.0,
            silence_ratio=1.0,
            clipping_ratio=0.0,
        ),
        assessment=AudioQualityAssessment(
            status="UNUSABLE",
            reasons=(
                "effectively_silent",
                "almost_entirely_silent",
            ),
        ),
    )

    storage_mock = Mock()
    storage_mock.get.return_value = (
        b"fake-audio-bytes"
    )

    acoustic_mock = Mock()
    encoder_factory_mock = Mock()

    with patch.object(
        feature_task_module,
        "SessionLocal",
        side_effect=TestingSessionLocal,
    ):
        with patch(
            "app.services.features.audio_features."
            "get_audio_storage",
            return_value=storage_mock,
        ):
            with patch(
                "app.services.features.audio_features."
                "process_audio",
                return_value=pipeline_result,
            ):
                with patch(
                    "app.services.features.audio_features."
                    "extract_acoustic_features",
                    acoustic_mock,
                ):
                    with patch(
                        "app.services.features.audio_features."
                        "get_audio_encoder",
                        encoder_factory_mock,
                    ):
                        with pytest.raises(
                            ValueError,
                            match=(
                                "Journal audio is unusable "
                                "after quality assessment"
                            ),
                        ):
                            extract_journal_audio_features.run(
                                str(journal_id),
                                str(feature_set_id),
                            )

    # -----------------------------------------------------
    # Expensive downstream processing must never occur.
    # -----------------------------------------------------

    acoustic_mock.assert_not_called()
    encoder_factory_mock.assert_not_called()

    db_session.expire_all()

    persisted = db_session.get(
        JournalFeatureSet,
        feature_set_id,
    )

    assert persisted is not None

    assert (
        persisted.status
        == FEATURE_STATUS_FAILED
    )

    assert persisted.audio_feature is None

    assert (
        persisted.error_message
        == (
            "Audio feature extraction failed: "
            "ValueError"
        )
    )

    # M3 failure remains isolated from M2.
    persisted_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    assert persisted_journal is not None

    assert (
        persisted_journal.status
        == "COMPLETED"
    )


def test_degraded_audio_is_persisted_and_generation_completes(
    db_session,
):
    """
    DEGRADED audio remains usable.

    It must:
        - continue through acoustic extraction
        - continue through the learned audio encoder
        - persist AudioFeature
        - preserve quality status and reasons
        - complete the VOICE generation when text already exists
        - leave the M2 journal COMPLETED
    """

    import numpy as np

    from app.services.features.acoustic_features import (
        AcousticFeatureResult,
    )
    from app.services.features.audio_encoder import (
        AudioEncodingResult,
    )
    from app.services.features.audio_pipeline import (
        AudioPipelineResult,
    )
    from app.services.features.audio_quality import (
        AudioQualityMeasurements,
    )
    from app.services.features.audio_quality_policy import (
        AudioQualityAssessment,
    )

    user = User(
        email="m39-degraded-audio@example.com",
        password_hash="test-password-hash",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="VOICE",
        raw_text=(
            "Valid transcript for degraded audio."
        ),
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.flush()

    audio = JournalAudio(
        journal_id=journal.id,
        storage_key="tests/m39-degraded-audio.wav",
        original_filename="m39-degraded-audio.wav",
        mime_type="audio/wav",
        size_bytes=123,
        transcription_status="TRANSCRIPTION_COMPLETED",
    )

    db_session.add(audio)
    db_session.flush()

    resolution = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash="m39-degraded-audio-source-hash",
    )

    feature_set = resolution.feature_set

    # VOICE requires both modalities before the generation can
    # become COMPLETED. Pretend the text child already won.
    feature_set.text_feature = TextFeature(
        source_type="TRANSCRIPT",
        preprocessing_version="test-text-preprocess",
        encoder_name="test-text-encoder",
        encoder_version="test-text-encoder-v1",
        encoder_revision="test-revision",
        embedding_dimension=768,
        embedding=[0.0] * 768,
        word_count=5,
        character_count=32,
        quality_status="GOOD",
        feature_metadata={},
    )

    db_session.flush()

    journal_id = journal.id
    feature_set_id = feature_set.id

    db_session.commit()

    waveform = np.full(
        16_000,
        0.005,
        dtype=np.float32,
    )

    pipeline_result = AudioPipelineResult(
        waveform=waveform,
        sample_rate_hz=16_000,
        sample_count=16_000,
        duration_seconds=1.0,
        original_sample_rate_hz=16_000,
        original_channels=1,
        
        measurements=AudioQualityMeasurements(
            peak_amplitude=0.005,
            rms_amplitude=0.005,
            silence_ratio=0.50,
            clipping_ratio=0.0,
        ),
        assessment=AudioQualityAssessment(
            status="DEGRADED",
            reasons=(
                "short_duration",
                "low_signal_level",
            ),
        ),
    )

    acoustic_result = AcousticFeatureResult(
        feature_version="test-acoustic-v1",
        duration_seconds=1.0,
        sample_rate_hz=16_000,
        frame_count=10,
        signal_activity_ratio=0.5,
        signal_inactivity_ratio=0.5,
        rms_mean=0.1,
        rms_std=0.01,
        spectral_centroid_mean_hz=1000.0,
        spectral_centroid_std_hz=100.0,
        spectral_bandwidth_mean_hz=500.0,
        spectral_bandwidth_std_hz=50.0,
        spectral_rolloff_mean_hz=2000.0,
        spectral_rolloff_std_hz=200.0,
        zero_crossing_rate_mean=0.1,
        zero_crossing_rate_std=0.01,
        mfcc_mean=tuple([0.0] * 13),
        mfcc_std=tuple([0.0] * 13),
    )
    
    audio_embedding = np.zeros(
    768,
    dtype=np.float32,
    )
    audio_embedding[0] = 1.0

    audio_encoding = AudioEncodingResult(
        encoder_name="test-audio-encoder",
        encoder_version="test-audio-encoder-v1",
        encoder_revision="test-revision",
        embedding_dimension=768,
        embedding=audio_embedding,
        sample_rate_hz=16_000,
        pooling_strategy="masked_temporal_mean",
        normalization="l2",
    )

    storage_mock = Mock()
    storage_mock.get.return_value = b"fake-audio"

    encoder_mock = Mock()
    encoder_mock.encode.return_value = audio_encoding

    with patch.object(
        feature_task_module,
        "SessionLocal",
        side_effect=TestingSessionLocal,
    ):
        with patch(
            "app.services.features.audio_features."
            "get_audio_storage",
            return_value=storage_mock,
        ):
            with patch(
                "app.services.features.audio_features."
                "process_audio",
                return_value=pipeline_result,
            ):
                with patch(
                    "app.services.features.audio_features."
                    "extract_acoustic_features",
                    return_value=acoustic_result,
                ) as acoustic_mock:
                    with patch(
                        "app.services.features.audio_features."
                        "get_audio_encoder",
                        return_value=encoder_mock,
                    ):
                        result = (
                            extract_journal_audio_features.run(
                                str(journal_id),
                                str(feature_set_id),
                            )
                        )

    assert result["status"] == "completed"

    acoustic_mock.assert_called_once()

    encoder_mock.encode.assert_called_once()

    db_session.expire_all()

    persisted = db_session.get(
        JournalFeatureSet,
        feature_set_id,
    )

    assert persisted is not None
    assert persisted.status == FEATURE_STATUS_COMPLETED

    assert persisted.audio_feature is not None

    assert (
        persisted.audio_feature.quality_status
        == "DEGRADED"
    )

    assert (
        persisted.audio_feature.feature_metadata[
            "quality_reasons"
        ]
        == [
            "short_duration",
            "low_signal_level",
        ]
    )

    persisted_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    assert persisted_journal is not None
    assert persisted_journal.status == "COMPLETED"