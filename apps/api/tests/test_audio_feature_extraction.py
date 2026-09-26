import uuid

import numpy as np
import pytest

from app.models.feature_constants import (
    FEATURE_QUALITY_GOOD,
    FEATURE_QUALITY_UNUSABLE,
    FEATURE_STATUS_COMPLETED,
    FEATURE_STATUS_PROCESSING,
)
from app.models.journal import JournalEntry
from app.models.journal_audio import JournalAudio
from app.models.journal_feature_set import JournalFeatureSet
from app.models.text_feature import TextFeature
from app.models.user import User
from app.services.features.audio_features import (
    extract_audio_features,
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


def make_user() -> User:
    unique_id = uuid.uuid4()

    return User(
        id=unique_id,
        email=f"audio-test-{unique_id}@example.com",
        password_hash="test-password-hash",
        display_name="Audio Test User",
        is_active=True,
    )


def make_pipeline_result(
    *,
    quality_status: str = FEATURE_QUALITY_GOOD,
    quality_reasons: tuple[str, ...] = tuple(),
) -> AudioPipelineResult:
    waveform = np.zeros(
        48_000,
        dtype=np.float32,
    )

    measurements = AudioQualityMeasurements(
        peak_amplitude=0.5,
        rms_amplitude=0.2,
        silence_ratio=0.1,
        clipping_ratio=0.0,
    )

    assessment = AudioQualityAssessment(
        status=quality_status,
        reasons=quality_reasons,
    )

    return AudioPipelineResult(
        waveform=waveform,
        sample_rate_hz=16_000,
        sample_count=48_000,
        duration_seconds=3.0,
        original_sample_rate_hz=48_000,
        original_channels=1,
        measurements=measurements,
        assessment=assessment,
    )


def make_voice_journal(
    user: User,
) -> JournalEntry:
    journal = JournalEntry(
        id=uuid.uuid4(),
        user_id=user.id,
        entry_type="VOICE",
        raw_text="This is the transcript.",
        status="COMPLETED",
    )

    journal.audio = JournalAudio(
        id=uuid.uuid4(),
        storage_key=(
            f"journals/{journal.id}/audio.webm"
        ),
        original_filename="audio.webm",
        mime_type="audio/webm",
        size_bytes=1234,
        transcription_status="TRANSCRIBED",
    )

    return journal


def make_text_journal(
    user: User,
) -> JournalEntry:
    return JournalEntry(
        id=uuid.uuid4(),
        user_id=user.id,
        entry_type="TEXT",
        raw_text="Text only journal.",
        status="COMPLETED",
    )


def make_voice_journal_without_audio(
    user: User,
) -> JournalEntry:
    return JournalEntry(
        id=uuid.uuid4(),
        user_id=user.id,
        entry_type="VOICE",
        raw_text="Transcript exists.",
        status="COMPLETED",
    )


def make_feature_set(
    journal: JournalEntry,
) -> JournalFeatureSet:
    return JournalFeatureSet(
        id=uuid.uuid4(),
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash=(
            f"test-source-hash-{uuid.uuid4()}"
        ),
        status="PENDING",
    )


def install_fake_audio_dependencies(
    monkeypatch,
    *,
    journal: JournalEntry,
    pipeline_result: AudioPipelineResult,
):
    class FakeStorage:
        def get(self, key: str) -> bytes:
            assert (
                key
                == journal.audio.storage_key
            )

            return b"encoded-audio"

    monkeypatch.setattr(
        "app.services.features.audio_features."
        "get_audio_storage",
        lambda: FakeStorage(),
    )

    monkeypatch.setattr(
        "app.services.features.audio_features."
        "process_audio",
        lambda audio_bytes: pipeline_result,
    )


def test_audio_features_are_persisted(
    db_session,
    monkeypatch,
):
    user = make_user()

    db_session.add(user)
    db_session.flush()

    journal = make_voice_journal(
        user
    )

    feature_set = make_feature_set(
        journal
    )

    db_session.add(journal)
    db_session.add(feature_set)
    db_session.flush()

    expected = make_pipeline_result()

    install_fake_audio_dependencies(
        monkeypatch,
        journal=journal,
        pipeline_result=expected,
    )

    result = extract_audio_features(
        db_session,
        journal=journal,
        feature_set=feature_set,
    )

    audio_feature = (
        result.audio_feature
    )

    assert (
        result.feature_set.id
        == feature_set.id
    )

    assert (
        result.pipeline_result
        == expected
    )

    assert (
        audio_feature.feature_set_id
        == feature_set.id
    )

    assert (
        feature_set.audio_feature.id
        == audio_feature.id
    )

    assert (
        audio_feature.preprocessing_version
        == "audio-preprocess-v1"
    )

    # Acoustic encoder has not been introduced
    # yet. These remain NULL until M3.6.
    assert audio_feature.encoder_name is None

    assert (
        audio_feature.encoder_version
        is None
    )

    assert (
        audio_feature.embedding_dimension
        is None
    )

    assert (
        audio_feature.duration_seconds
        == pytest.approx(3.0)
    )

    # Silence ratio is NOT speech/VAD.
    assert (
        audio_feature.speech_ratio
        is None
    )

    assert (
        audio_feature.sample_rate_hz
        == 16_000
    )

    assert (
        audio_feature.quality_status
        == FEATURE_QUALITY_GOOD
    )

    metadata = (
        audio_feature.feature_metadata
    )

    assert (
        metadata[
            "audio_pipeline_version"
        ]
        == "audio-pipeline-v1"
    )

    assert (
        metadata[
            "audio_quality_version"
        ]
        == "audio-quality-v1"
    )

    assert (
        metadata[
            "audio_quality_policy_version"
        ]
        == "audio-quality-policy-v1"
    )

    assert (
        metadata["quality_reasons"]
        == []
    )

    assert (
        metadata["sample_count"]
        == 48_000
    )

    assert (
        metadata[
            "original_sample_rate_hz"
        ]
        == 48_000
    )

    assert (
        metadata[
            "original_channels"
        ]
        == 1
    )

    assert (
        metadata[
            "peak_amplitude"
        ]
        == pytest.approx(0.5)
    )

    assert (
        metadata[
            "rms_amplitude"
        ]
        == pytest.approx(0.2)
    )

    assert (
        metadata[
            "silence_ratio"
        ]
        == pytest.approx(0.1)
    )

    assert (
        metadata[
            "clipping_ratio"
        ]
        == pytest.approx(0.0)
    )

    # A VOICE feature generation requires
    # both modalities. Audio alone must not
    # complete the feature generation.
    assert (
        feature_set.status
        == FEATURE_STATUS_PROCESSING
    )


def test_audio_completes_generation_when_text_exists(
    db_session,
    monkeypatch,
):
    user = make_user()

    db_session.add(user)
    db_session.flush()

    journal = make_voice_journal(
        user
    )

    feature_set = make_feature_set(
        journal
    )

    db_session.add(journal)
    db_session.add(feature_set)
    db_session.flush()

    text_feature = TextFeature(
        feature_set_id=feature_set.id,
        source_type="TRANSCRIPT",
        preprocessing_version=(
            "text-preprocess-v1"
        ),
        encoder_name="test-encoder",
        encoder_revision="test-revision",
        encoder_version="text-encoder-v1",
        embedding_dimension=768,
        embedding=[0.0] * 768,
        word_count=4,
        character_count=23,
        quality_status=FEATURE_QUALITY_GOOD,
        feature_metadata={},
    )

    feature_set.text_feature = (
        text_feature
    )

    db_session.flush()

    expected = make_pipeline_result()

    install_fake_audio_dependencies(
        monkeypatch,
        journal=journal,
        pipeline_result=expected,
    )

    extract_audio_features(
        db_session,
        journal=journal,
        feature_set=feature_set,
    )

    assert (
        feature_set.status
        == FEATURE_STATUS_COMPLETED
    )

    assert (
        feature_set.audio_feature
        is not None
    )

    assert (
        feature_set.text_feature
        is not None
    )


def test_unusable_audio_is_still_persisted(
    db_session,
    monkeypatch,
):
    user = make_user()

    db_session.add(user)
    db_session.flush()

    journal = make_voice_journal(
        user
    )

    feature_set = make_feature_set(
        journal
    )

    db_session.add(journal)
    db_session.add(feature_set)
    db_session.flush()

    expected = make_pipeline_result(
        quality_status=(
            FEATURE_QUALITY_UNUSABLE
        ),
        quality_reasons=(
            "effectively_silent",
            "almost_entirely_silent",
        ),
    )

    install_fake_audio_dependencies(
        monkeypatch,
        journal=journal,
        pipeline_result=expected,
    )

    result = extract_audio_features(
        db_session,
        journal=journal,
        feature_set=feature_set,
    )

    assert (
        result.audio_feature.quality_status
        == FEATURE_QUALITY_UNUSABLE
    )

    assert (
        result.audio_feature.feature_metadata[
            "quality_reasons"
        ]
        == [
            "effectively_silent",
            "almost_entirely_silent",
        ]
    )

    # An unusable signal is still valuable
    # provenance. We persist it rather than
    # pretending extraction never occurred.
    assert (
        feature_set.audio_feature
        is not None
    )

    assert (
        feature_set.status
        == FEATURE_STATUS_PROCESSING
    )


def test_text_journal_rejects_audio_extraction(
    db_session,
):
    user = make_user()

    db_session.add(user)
    db_session.flush()

    journal = make_text_journal(
        user
    )

    feature_set = make_feature_set(
        journal
    )

    db_session.add(journal)
    db_session.add(feature_set)
    db_session.flush()

    with pytest.raises(
        ValueError,
        match="VOICE",
    ):
        extract_audio_features(
            db_session,
            journal=journal,
            feature_set=feature_set,
        )


def test_voice_journal_without_audio_is_rejected(
    db_session,
):
    user = make_user()

    db_session.add(user)
    db_session.flush()

    journal = (
        make_voice_journal_without_audio(
            user
        )
    )

    feature_set = make_feature_set(
        journal
    )

    db_session.add(journal)
    db_session.add(feature_set)
    db_session.flush()

    with pytest.raises(
        ValueError,
        match="no audio",
    ):
        extract_audio_features(
            db_session,
            journal=journal,
            feature_set=feature_set,
        )


def test_feature_set_must_belong_to_journal(
    db_session,
):
    user = make_user()

    db_session.add(user)
    db_session.flush()

    journal = make_voice_journal(
        user
    )

    db_session.add(journal)
    db_session.flush()

    feature_set = JournalFeatureSet(
        id=uuid.uuid4(),

        # Deliberately points somewhere else.
        journal_id=uuid.uuid4(),

        pipeline_version="m3-v1",
        source_hash=(
            f"different-journal-"
            f"{uuid.uuid4()}"
        ),
        status="PENDING",
    )

    # Do NOT persist this deliberately invalid
    # feature set. PostgreSQL would correctly
    # reject its journal FK before our service
    # invariant could be tested.
    with pytest.raises(
        ValueError,
        match="does not belong",
    ):
        extract_audio_features(
            db_session,
            journal=journal,
            feature_set=feature_set,
        )