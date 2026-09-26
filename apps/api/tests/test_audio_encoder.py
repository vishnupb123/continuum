import numpy as np
import pytest

from app.services.features.audio_encoder import (
    AUDIO_ENCODER_DIMENSION,
    AUDIO_ENCODER_SAMPLE_RATE_HZ,
    AUDIO_ENCODER_VERSION,
    AUDIO_NORMALIZATION,
    AUDIO_POOLING_STRATEGY,
    AudioEncodingResult,
    l2_normalize_embedding,
    validate_audio_encoder_input,
)
from app.services.features.mock_audio_encoder import (
    MOCK_AUDIO_ENCODER_NAME,
    MOCK_AUDIO_ENCODER_REVISION,
    MockAudioEncoder,
)


def _waveform() -> np.ndarray:
    time = (
        np.arange(
            16_000,
            dtype=np.float64,
        )
        / 16_000.0
    )

    return (
        0.25
        * np.sin(
            2.0
            * np.pi
            * 440.0
            * time
        )
    ).astype(
        np.float32
    )


def test_audio_encoder_contract_constants():
    assert (
        AUDIO_ENCODER_VERSION
        == "audio-encoder-v1"
    )

    assert (
        AUDIO_ENCODER_DIMENSION
        == 768
    )

    assert (
        AUDIO_ENCODER_SAMPLE_RATE_HZ
        == 16_000
    )

    assert (
        AUDIO_POOLING_STRATEGY
        == "masked_temporal_mean"
    )

    assert (
        AUDIO_NORMALIZATION
        == "l2"
    )


def test_validates_canonical_waveform():
    validate_audio_encoder_input(
        _waveform(),
        sample_rate_hz=16_000,
    )


def test_rejects_wrong_waveform_dtype():
    waveform = np.zeros(
        16_000,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="float32",
    ):
        validate_audio_encoder_input(
            waveform,
            sample_rate_hz=16_000,
        )


def test_rejects_non_1d_waveform():
    waveform = np.zeros(
        (1, 16_000),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        validate_audio_encoder_input(
            waveform,
            sample_rate_hz=16_000,
        )


def test_rejects_empty_waveform():
    waveform = np.array(
        [],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        validate_audio_encoder_input(
            waveform,
            sample_rate_hz=16_000,
        )


def test_rejects_nonfinite_waveform():
    waveform = _waveform()
    waveform[100] = np.nan

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        validate_audio_encoder_input(
            waveform,
            sample_rate_hz=16_000,
        )


def test_rejects_wrong_sample_rate():
    with pytest.raises(
        ValueError,
        match="16 kHz",
    ):
        validate_audio_encoder_input(
            _waveform(),
            sample_rate_hz=44_100,
        )


def test_l2_normalization():
    embedding = np.array(
        [3.0, 4.0],
        dtype=np.float32,
    )

    result = l2_normalize_embedding(
        embedding
    )

    assert result.dtype == np.float32

    assert float(
        np.linalg.norm(
            result.astype(
                np.float64
            )
        )
    ) == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_l2_normalization_does_not_modify_input():
    embedding = np.array(
        [3.0, 4.0],
        dtype=np.float32,
    )

    original = embedding.copy()

    l2_normalize_embedding(
        embedding
    )

    np.testing.assert_array_equal(
        embedding,
        original,
    )


def test_l2_normalization_rejects_zero_vector():
    with pytest.raises(
        ValueError,
        match="zero",
    ):
        l2_normalize_embedding(
            np.zeros(
                768,
                dtype=np.float32,
            )
        )


def test_mock_encoder_metadata():
    encoder = MockAudioEncoder()

    assert (
        encoder.encoder_name
        == MOCK_AUDIO_ENCODER_NAME
    )

    assert (
        encoder.encoder_version
        == AUDIO_ENCODER_VERSION
    )

    assert (
        encoder.encoder_revision
        == MOCK_AUDIO_ENCODER_REVISION
    )

    assert (
        encoder.embedding_dimension
        == 768
    )


def test_mock_encoder_returns_contract():
    encoder = MockAudioEncoder()

    result = encoder.encode(
        _waveform(),
        sample_rate_hz=16_000,
    )

    assert isinstance(
        result,
        AudioEncodingResult,
    )

    assert result.embedding.shape == (
        768,
    )

    assert (
        result.embedding.dtype
        == np.float32
    )

    assert (
        result.encoder_name
        == MOCK_AUDIO_ENCODER_NAME
    )

    assert (
        result.encoder_version
        == AUDIO_ENCODER_VERSION
    )

    assert (
        result.encoder_revision
        == MOCK_AUDIO_ENCODER_REVISION
    )

    assert (
        result.pooling_strategy
        == "masked_temporal_mean"
    )

    assert (
        result.normalization
        == "l2"
    )


def test_mock_embedding_is_l2_normalized():
    result = MockAudioEncoder().encode(
        _waveform(),
        sample_rate_hz=16_000,
    )

    norm = np.linalg.norm(
        result.embedding.astype(
            np.float64
        )
    )

    assert norm == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_mock_encoder_is_deterministic():
    encoder = MockAudioEncoder()

    first = encoder.encode(
        _waveform(),
        sample_rate_hz=16_000,
    )

    second = encoder.encode(
        _waveform(),
        sample_rate_hz=16_000,
    )

    np.testing.assert_array_equal(
        first.embedding,
        second.embedding,
    )


def test_different_waveforms_produce_different_embeddings():
    encoder = MockAudioEncoder()

    first_waveform = _waveform()

    second_waveform = (
        first_waveform * 0.5
    ).astype(
        np.float32
    )

    first = encoder.encode(
        first_waveform,
        sample_rate_hz=16_000,
    )

    second = encoder.encode(
        second_waveform,
        sample_rate_hz=16_000,
    )

    assert not np.array_equal(
        first.embedding,
        second.embedding,
    )


def test_mock_encoder_does_not_modify_waveform():
    waveform = _waveform()
    original = waveform.copy()

    MockAudioEncoder().encode(
        waveform,
        sample_rate_hz=16_000,
    )

    np.testing.assert_array_equal(
        waveform,
        original,
    )


def test_audio_encoding_result_rejects_wrong_dimension():
    embedding = np.ones(
        10,
        dtype=np.float32,
    )

    embedding /= np.linalg.norm(
        embedding
    )

    with pytest.raises(
        ValueError,
        match="768",
    ):
        AudioEncodingResult(
            embedding=embedding,
            encoder_name="test",
            encoder_version=(
                AUDIO_ENCODER_VERSION
            ),
            encoder_revision=None,
            embedding_dimension=10,
            sample_rate_hz=16_000,
            pooling_strategy=(
                AUDIO_POOLING_STRATEGY
            ),
            normalization=(
                AUDIO_NORMALIZATION
            ),
        )


def test_audio_encoding_result_rejects_unnormalized_embedding():
    embedding = np.ones(
        768,
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="L2 normalized",
    ):
        AudioEncodingResult(
            embedding=embedding,
            encoder_name="test",
            encoder_version=(
                AUDIO_ENCODER_VERSION
            ),
            encoder_revision=None,
            embedding_dimension=768,
            sample_rate_hz=16_000,
            pooling_strategy=(
                AUDIO_POOLING_STRATEGY
            ),
            normalization=(
                AUDIO_NORMALIZATION
            ),
        )