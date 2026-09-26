import io
import wave

import numpy as np
import pytest
from sqlalchemy import select

from app.models.audio_feature import AudioFeature
from app.models.journal import JournalEntry
from app.models.journal_audio import JournalAudio
from app.models.journal_feature_set import JournalFeatureSet
from app.models.user import User
from app.services.features.audio_encoder_factory import (
    get_audio_encoder,
)
from app.services.features.audio_features import (
    extract_audio_features,
)
from app.services.features.wavlm_audio_encoder import (
    WAVLM_MODEL_NAME,
    WAVLM_MODEL_REVISION,
)


def _make_real_wav_bytes(
    *,
    sample_rate_hz: int = 16_000,
    duration_seconds: float = 1.0,
    frequency_hz: float = 440.0,
) -> bytes:
    """
    Create a real PCM WAV payload.

    This deliberately passes through the actual M3.4 audio
    preprocessing pipeline rather than mocking the canonical
    waveform.
    """

    sample_count = int(
        sample_rate_hz
        * duration_seconds
    )

    time = (
        np.arange(
            sample_count,
            dtype=np.float64,
        )
        / sample_rate_hz
    )

    waveform = (
        0.25
        * np.sin(
            2.0
            * np.pi
            * frequency_hz
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
            sample_rate_hz
        )
        wav_file.writeframes(
            pcm.tobytes()
        )

    return buffer.getvalue()


class RealPipelineAudioStorage:
    def __init__(
        self,
        audio_bytes: bytes,
    ) -> None:
        self._audio_bytes = audio_bytes

    def get(
        self,
        key: str,
    ) -> bytes:
        assert (
            key
            == "tests/m3-6-real-wavlm.wav"
        )

        return self._audio_bytes


@pytest.mark.integration
def test_real_wavlm_audio_feature_pipeline_persists_embedding(
    db_session,
    monkeypatch,
):
    """
    M3.6 acceptance test.

    Proves the real production representation path:

        real WAV bytes
            ->
        real M3.4 preprocessing
            ->
        real M3.5 acoustic extraction
            ->
        configured real WavLM Base+
            ->
        AudioFeature
            ->
        PostgreSQL VECTOR(768)
            ->
        database read-back

    WavLM itself is NOT mocked.
    """

    # ---------------------------------------------------------
    # Arrange
    # ---------------------------------------------------------

    audio_bytes = (
        _make_real_wav_bytes()
    )

    user = User(
        email=(
            "m3-6-real-wavlm@example.com"
        ),
        password_hash=(
            "test-password-hash"
        ),
        display_name=(
            "M3.6 Real WavLM Test"
        ),
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="VOICE",
        raw_text=(
            "Real WavLM integration test."
        ),
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.flush()

    journal_audio = JournalAudio(
        journal_id=journal.id,
        storage_key=(
            "tests/m3-6-real-wavlm.wav"
        ),
        original_filename=(
            "m3-6-real-wavlm.wav"
        ),
        mime_type="audio/wav",
        size_bytes=len(
            audio_bytes
        ),
        transcription_status=(
            "TRANSCRIBED"
        ),
    )

    db_session.add(
        journal_audio
    )

    feature_set = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash=(
            "m3-6-real-wavlm-source"
        ),
        status="PENDING",
    )

    db_session.add(
        feature_set
    )

    db_session.flush()

    storage = RealPipelineAudioStorage(
        audio_bytes
    )

    monkeypatch.setattr(
        "app.services.features.audio_features."
        "get_audio_storage",
        lambda: storage,
    )

    # ---------------------------------------------------------
    # Verify that this acceptance test really uses WavLM.
    # ---------------------------------------------------------

    get_audio_encoder.cache_clear()

    encoder = get_audio_encoder()

    assert (
        encoder.encoder_name
        == WAVLM_MODEL_NAME
    )

    assert (
        encoder.encoder_revision
        == WAVLM_MODEL_REVISION
    )

    assert (
        encoder.encoder_version
        == "audio-encoder-v1"
    )

    assert (
        encoder.embedding_dimension
        == 768
    )

    # ---------------------------------------------------------
    # Act
    # ---------------------------------------------------------

    result = extract_audio_features(
        db_session,
        journal=journal,
        feature_set=feature_set,
    )

    audio_feature = (
        result.audio_feature
    )

    # ---------------------------------------------------------
    # Verify M3.4 canonical preprocessing
    # ---------------------------------------------------------

    assert (
        result.pipeline_result.sample_rate_hz
        == 16_000
    )

    assert (
        result.pipeline_result.waveform.dtype
        == np.float32
    )

    assert (
        result.pipeline_result.waveform.ndim
        == 1
    )

    assert np.all(
        np.isfinite(
            result.pipeline_result.waveform
        )
    )

    # ---------------------------------------------------------
    # Verify M3.5 deterministic acoustics survived
    # ---------------------------------------------------------

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

    assert (
        acoustic["frame_count"]
        > 0
    )

    assert (
        len(
            acoustic["mfcc_mean"]
        )
        == 13
    )

    assert (
        len(
            acoustic["mfcc_std"]
        )
        == 13
    )

    # ---------------------------------------------------------
    # Verify real M3.6 WavLM representation
    # ---------------------------------------------------------

    assert (
        audio_feature.encoder_name
        == WAVLM_MODEL_NAME
    )

    assert (
        audio_feature.encoder_version
        == "audio-encoder-v1"
    )

    assert (
        audio_feature.encoder_revision
        == WAVLM_MODEL_REVISION
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
        np.isfinite(
            embedding
        )
    )

    assert np.linalg.norm(
        embedding.astype(
            np.float64
        )
    ) == pytest.approx(
        1.0,
        abs=1e-5,
    )

    # The domain result returned by the real
    # WavLM adapter must match persistence.
    np.testing.assert_allclose(
        embedding,
        result.audio_encoding.embedding,
        rtol=0.0,
        atol=1e-6,
    )

    # ---------------------------------------------------------
    # Verify representation provenance
    # ---------------------------------------------------------

    assert (
        "audio_embedding"
        in metadata
    )

    embedding_metadata = metadata[
        "audio_embedding"
    ]

    assert (
        embedding_metadata[
            "encoder_name"
        ]
        == WAVLM_MODEL_NAME
    )

    assert (
        embedding_metadata[
            "encoder_version"
        ]
        == "audio-encoder-v1"
    )

    assert (
        embedding_metadata[
            "encoder_revision"
        ]
        == WAVLM_MODEL_REVISION
    )

    assert (
        embedding_metadata[
            "embedding_dimension"
        ]
        == 768
    )

    assert (
        embedding_metadata[
            "sample_rate_hz"
        ]
        == 16_000
    )

    assert (
        embedding_metadata[
            "pooling_strategy"
        ]
        == "masked_temporal_mean"
    )

    assert (
        embedding_metadata[
            "normalization"
        ]
        == "l2"
    )

    # ---------------------------------------------------------
    # Force PostgreSQL round-trip
    # ---------------------------------------------------------

    audio_feature_id = (
        audio_feature.id
    )

    db_session.commit()
    db_session.expire_all()

    persisted = db_session.scalar(
        select(
            AudioFeature
        ).where(
            AudioFeature.id
            == audio_feature_id
        )
    )

    assert persisted is not None

    assert (
        persisted.encoder_name
        == WAVLM_MODEL_NAME
    )

    assert (
        persisted.encoder_revision
        == WAVLM_MODEL_REVISION
    )

    assert (
        persisted.embedding_dimension
        == 768
    )

    assert (
        persisted.embedding
        is not None
    )

    persisted_embedding = np.asarray(
        persisted.embedding,
        dtype=np.float32,
    )

    assert persisted_embedding.shape == (
        768,
    )

    assert np.all(
        np.isfinite(
            persisted_embedding
        )
    )

    assert np.linalg.norm(
        persisted_embedding.astype(
            np.float64
        )
    ) == pytest.approx(
        1.0,
        abs=1e-5,
    )

    np.testing.assert_allclose(
        persisted_embedding,
        result.audio_encoding.embedding,
        rtol=0.0,
        atol=1e-6,
    )