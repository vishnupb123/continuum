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
from app.services.features.mock_audio_encoder import (
    MockAudioEncoder,
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
    """
    Install deterministic test doubles around the audio feature
    extraction service.

    M3.4 preprocessing is replaced with the supplied pipeline result,
    while M3.6 uses the deterministic MockAudioEncoder.

    This keeps the service tests fast and prevents WavLM from being
    loaded during ordinary regression tests.
    """

    class FakeStorage:
        def get(self, key: str) -> bytes:
            assert (
                key
                == journal.audio.storage_key
            )

            return b"encoded-audio"

    mock_audio_encoder = (
        MockAudioEncoder()
    )

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

    monkeypatch.setattr(
        "app.services.features.audio_features."
        "get_audio_encoder",
        lambda: mock_audio_encoder,
    )

    return mock_audio_encoder


def assert_valid_mock_audio_embedding(
    audio_feature,
) -> None:
    """
    Assert the frozen M3.6 learned-audio persistence contract.
    """

    assert (
        audio_feature.encoder_name
        == "mock-audio-encoder"
    )

    assert (
        audio_feature.encoder_version
        == "audio-encoder-v1"
    )

    assert (
        audio_feature.encoder_revision
        == "deterministic-v1"
    )

    assert (
        audio_feature.embedding_dimension
        == 768
    )

    assert (
        audio_feature.embedding
        is not None
    )

    embedding = np.asarray(
        audio_feature.embedding,
        dtype=np.float32,
    )

    assert embedding.shape == (
        768,
    )

    assert np.all(
        np.isfinite(embedding)
    )

    assert np.linalg.norm(
        embedding.astype(np.float64)
    ) == pytest.approx(
        1.0,
        abs=1e-6,
    )


def assert_valid_audio_embedding_metadata(
    metadata: dict,
) -> None:
    """
    Assert provenance for the M3.6 learned representation.
    """

    assert (
        "audio_embedding"
        in metadata
    )

    embedding_metadata = metadata[
        "audio_embedding"
    ]

    assert (
        embedding_metadata["encoder_name"]
        == "mock-audio-encoder"
    )

    assert (
        embedding_metadata["encoder_version"]
        == "audio-encoder-v1"
    )

    assert (
        embedding_metadata["encoder_revision"]
        == "deterministic-v1"
    )

    assert (
        embedding_metadata["embedding_dimension"]
        == 768
    )

    assert (
        embedding_metadata["sample_rate_hz"]
        == 16_000
    )

    assert (
        embedding_metadata["pooling_strategy"]
        == "masked_temporal_mean"
    )

    assert (
        embedding_metadata["normalization"]
        == "l2"
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

    # M3.6 learned audio representation.
    assert_valid_mock_audio_embedding(
        audio_feature
    )

    assert (
        result.audio_encoding.encoder_name
        == "mock-audio-encoder"
    )

    assert (
        result.audio_encoding.encoder_version
        == "audio-encoder-v1"
    )

    assert (
        result.audio_encoding.encoder_revision
        == "deterministic-v1"
    )

    assert (
        result.audio_encoding.embedding_dimension
        == 768
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

    # M3.6 representation provenance.
    assert_valid_audio_embedding_metadata(
        metadata
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

    result = extract_audio_features(
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

    assert_valid_mock_audio_embedding(
        result.audio_feature
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

    # M3.6 currently preserves the representation
    # even when engineering quality is UNUSABLE.
    # Downstream consumers must inspect quality_status.
    assert_valid_mock_audio_embedding(
        result.audio_feature
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


def test_audio_extraction_persists_acoustic_features(
    db_session,
    monkeypatch,
):
    import io
    import wave

    sample_rate = 16_000
    duration_seconds = 1.0

    time = (
        np.arange(
            int(
                sample_rate
                * duration_seconds
            ),
            dtype=np.float64,
        )
        / sample_rate
    )

    waveform = (
        0.25
        * np.sin(
            2.0
            * np.pi
            * 440.0
            * time
        )
    )

    pcm = (
        waveform
        * 32767.0
    ).astype(
        np.int16
    )

    buffer = io.BytesIO()

    with wave.open(
        buffer,
        "wb",
    ) as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(
            sample_rate
        )
        wav_file.writeframes(
            pcm.tobytes()
        )

    audio_bytes = buffer.getvalue()

    user = User(
        email="acoustic-persistence@example.com",
        password_hash="test-password-hash",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="VOICE",
        raw_text="Test transcript.",
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.flush()

    journal_audio = JournalAudio(
        journal_id=journal.id,
        storage_key="test/acoustic.wav",
        original_filename="acoustic.wav",
        mime_type="audio/wav",
        size_bytes=len(audio_bytes),
        transcription_status="TRANSCRIBED",
    )

    db_session.add(journal_audio)
    db_session.flush()

    feature_set = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash="acoustic-test-hash",
        status="PENDING",
    )

    db_session.add(feature_set)
    db_session.flush()

    class FakeStorage:
        def get(
            self,
            key: str,
        ) -> bytes:
            assert (
                key
                == "test/acoustic.wav"
            )

            return audio_bytes

    monkeypatch.setattr(
        "app.services.features.audio_features."
        "get_audio_storage",
        lambda: FakeStorage(),
    )

    mock_audio_encoder = (
        MockAudioEncoder()
    )

    monkeypatch.setattr(
        "app.services.features.audio_features."
        "get_audio_encoder",
        lambda: mock_audio_encoder,
    )

    result = extract_audio_features(
        db_session,
        journal=journal,
        feature_set=feature_set,
    )

    audio_feature = (
        result.audio_feature
    )

    metadata = (
        audio_feature.feature_metadata
    )

    assert (
        "acoustic_features"
        in metadata
    )

    acoustic = metadata[
        "acoustic_features"
    ]

    assert (
        acoustic["feature_version"]
        == "acoustic-features-v1"
    )

    assert acoustic["frame_count"] > 0

    assert (
        0.0
        <= acoustic[
            "signal_activity_ratio"
        ]
        <= 1.0
    )

    assert (
        acoustic[
            "signal_activity_ratio"
        ]
        + acoustic[
            "signal_inactivity_ratio"
        ]
        == pytest.approx(1.0)
    )

    assert np.isfinite(
        acoustic["rms_mean"]
    )

    assert np.isfinite(
        acoustic[
            "spectral_centroid_mean_hz"
        ]
    )

    assert np.isfinite(
        acoustic[
            "spectral_bandwidth_mean_hz"
        ]
    )

    assert np.isfinite(
        acoustic[
            "spectral_rolloff_mean_hz"
        ]
    )

    assert np.isfinite(
        acoustic[
            "zero_crossing_rate_mean"
        ]
    )

    assert len(
        acoustic["mfcc_mean"]
    ) == 13

    assert len(
        acoustic["mfcc_std"]
    ) == 13

    assert np.all(
        np.isfinite(
            acoustic["mfcc_mean"]
        )
    )

    assert np.all(
        np.isfinite(
            acoustic["mfcc_std"]
        )
    )

    # M3.5 signal activity is NOT VAD.
    assert (
        audio_feature.speech_ratio
        is None
    )

    # M3.6 learned representation now coexists
    # with the deterministic M3.5 representation.
    assert_valid_mock_audio_embedding(
        audio_feature
    )

    assert_valid_audio_embedding_metadata(
        metadata
    )

    # Returned domain representation must match
    # persistence.
    assert (
        result.acoustic_features.feature_version
        == acoustic["feature_version"]
    )

    assert (
        list(
            result.acoustic_features.mfcc_mean
        )
        == acoustic["mfcc_mean"]
    )

    # Returned M3.6 representation must also
    # match persistence.
    persisted_embedding = np.asarray(
        audio_feature.embedding,
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        persisted_embedding,
        result.audio_encoding.embedding,
        rtol=0.0,
        atol=1e-7,
    )