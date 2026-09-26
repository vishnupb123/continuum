import numpy as np
import pytest

from app.services.features.audio_encoder import (
    AUDIO_ENCODER_DIMENSION,
    AUDIO_ENCODER_VERSION,
    AUDIO_NORMALIZATION,
    AUDIO_POOLING_STRATEGY,
)
from app.services.features.wavlm_audio_encoder import (
    WAVLM_DEVICE_CPU,
    WAVLM_HIDDEN_SIZE,
    WAVLM_MODEL_NAME,
    WAVLM_MODEL_REVISION,
    WavLMAudioEncoder,
    _load_wavlm_components,
)


def _waveform(
    *,
    duration_seconds: float = 1.0,
    frequency_hz: float = 440.0,
) -> np.ndarray:
    sample_rate = 16_000

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

    return (
        0.25
        * np.sin(
            2.0
            * np.pi
            * frequency_hz
            * time
        )
    ).astype(
        np.float32
    )


@pytest.fixture(scope="module")
def encoder():
    return WavLMAudioEncoder()


def test_wavlm_contract_constants():
    assert (
        WAVLM_MODEL_NAME
        == "microsoft/wavlm-base-plus"
    )

    assert (
        WAVLM_MODEL_REVISION
        == (
            "4c66d4806a428f2e922ccfa1a962776e232d487b"
        )
    )

    assert WAVLM_HIDDEN_SIZE == 768

    assert WAVLM_DEVICE_CPU == "cpu"


def test_wavlm_encoder_metadata(
    encoder,
):
    assert (
        encoder.encoder_name
        == WAVLM_MODEL_NAME
    )

    assert (
        encoder.encoder_version
        == AUDIO_ENCODER_VERSION
    )

    assert (
        encoder.encoder_revision
        == WAVLM_MODEL_REVISION
    )

    assert (
        encoder.embedding_dimension
        == AUDIO_ENCODER_DIMENSION
    )


def test_wavlm_real_model_configuration(
    encoder,
):
    (
        feature_extractor,
        model,
    ) = _load_wavlm_components(
        WAVLM_MODEL_NAME,
        WAVLM_MODEL_REVISION,
        WAVLM_DEVICE_CPU,
        None,
    )

    assert (
        model.config.hidden_size
        == 768
    )

    assert (
        model.config.num_hidden_layers
        == 12
    )

    assert (
        feature_extractor.sampling_rate
        == 16_000
    )

    assert (
        feature_extractor.do_normalize
        is False
    )


def test_wavlm_returns_audio_encoding_contract(
    encoder,
):
    result = encoder.encode(
        _waveform(),
        sample_rate_hz=16_000,
    )

    assert result.embedding.shape == (
        768,
    )

    assert (
        result.embedding.dtype
        == np.float32
    )

    assert (
        result.embedding_dimension
        == 768
    )

    assert (
        result.encoder_name
        == WAVLM_MODEL_NAME
    )

    assert (
        result.encoder_revision
        == WAVLM_MODEL_REVISION
    )

    assert (
        result.pooling_strategy
        == AUDIO_POOLING_STRATEGY
    )

    assert (
        result.normalization
        == AUDIO_NORMALIZATION
    )


def test_wavlm_embedding_is_finite(
    encoder,
):
    result = encoder.encode(
        _waveform(),
        sample_rate_hz=16_000,
    )

    assert np.all(
        np.isfinite(
            result.embedding
        )
    )


def test_wavlm_embedding_is_l2_normalized(
    encoder,
):
    result = encoder.encode(
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


def test_wavlm_is_deterministic(
    encoder,
):
    waveform = _waveform()

    first = encoder.encode(
        waveform,
        sample_rate_hz=16_000,
    )

    second = encoder.encode(
        waveform,
        sample_rate_hz=16_000,
    )

    np.testing.assert_allclose(
        first.embedding,
        second.embedding,
        rtol=0.0,
        atol=1e-6,
    )


def test_wavlm_different_audio_changes_representation(
    encoder,
):
    first = encoder.encode(
        _waveform(
            frequency_hz=440.0
        ),
        sample_rate_hz=16_000,
    )

    second = encoder.encode(
        _waveform(
            frequency_hz=880.0
        ),
        sample_rate_hz=16_000,
    )

    assert not np.allclose(
        first.embedding,
        second.embedding,
        rtol=0.0,
        atol=1e-5,
    )


def test_wavlm_does_not_modify_waveform(
    encoder,
):
    waveform = _waveform()

    original = waveform.copy()

    encoder.encode(
        waveform,
        sample_rate_hz=16_000,
    )

    np.testing.assert_array_equal(
        waveform,
        original,
    )


def test_wavlm_rejects_wrong_sample_rate(
    encoder,
):
    with pytest.raises(
        ValueError,
        match="16 kHz",
    ):
        encoder.encode(
            _waveform(),
            sample_rate_hz=44_100,
        )


def test_wavlm_components_are_process_cached():
    first = _load_wavlm_components(
        WAVLM_MODEL_NAME,
        WAVLM_MODEL_REVISION,
        WAVLM_DEVICE_CPU,
        None,
    )

    second = _load_wavlm_components(
        WAVLM_MODEL_NAME,
        WAVLM_MODEL_REVISION,
        WAVLM_DEVICE_CPU,
        None,
    )

    assert first[0] is second[0]
    assert first[1] is second[1]