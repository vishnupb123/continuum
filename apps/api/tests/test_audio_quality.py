import numpy as np
import pytest

from app.services.features.audio_quality import (
    AUDIO_QUALITY_VERSION,
    CLIPPING_THRESHOLD,
    SILENCE_THRESHOLD,
    measure_audio_quality,
)


def test_quality_version_is_frozen():
    assert AUDIO_QUALITY_VERSION == (
        "audio-quality-v1"
    )


def test_silence_is_measured_correctly():
    waveform = np.zeros(
        16_000,
        dtype=np.float32,
    )

    result = measure_audio_quality(
        waveform
    )

    assert result.peak_amplitude == 0.0
    assert result.rms_amplitude == 0.0
    assert result.silence_ratio == 1.0
    assert result.clipping_ratio == 0.0


def test_constant_signal_is_measured_correctly():
    waveform = np.full(
        16_000,
        0.5,
        dtype=np.float32,
    )

    result = measure_audio_quality(
        waveform
    )

    assert result.peak_amplitude == pytest.approx(
        0.5
    )

    assert result.rms_amplitude == pytest.approx(
        0.5
    )

    assert result.silence_ratio == 0.0
    assert result.clipping_ratio == 0.0


def test_known_silence_ratio():
    waveform = np.concatenate(
        [
            np.zeros(
                8_000,
                dtype=np.float32,
            ),
            np.full(
                8_000,
                0.25,
                dtype=np.float32,
            ),
        ]
    )

    result = measure_audio_quality(
        waveform
    )

    assert result.silence_ratio == pytest.approx(
        0.5
    )


def test_known_clipping_ratio():
    waveform = np.concatenate(
        [
            np.full(
                4_000,
                1.0,
                dtype=np.float32,
            ),
            np.full(
                12_000,
                0.25,
                dtype=np.float32,
            ),
        ]
    )

    result = measure_audio_quality(
        waveform
    )

    assert result.clipping_ratio == pytest.approx(
        0.25
    )


def test_threshold_contract():
    waveform = np.array(
        [
            0.0,
            SILENCE_THRESHOLD / 2,
            SILENCE_THRESHOLD,
            CLIPPING_THRESHOLD - 0.01,
            CLIPPING_THRESHOLD,
            1.0,
        ],
        dtype=np.float32,
    )

    result = measure_audio_quality(
        waveform
    )

    # Strictly below silence threshold.
    assert result.silence_ratio == pytest.approx(
        2 / 6
    )

    # Greater than or equal to clipping threshold.
    assert result.clipping_ratio == pytest.approx(
        2 / 6
    )


def test_measurement_is_deterministic():
    rng = np.random.default_rng(
        seed=42
    )

    waveform = rng.uniform(
        -0.5,
        0.5,
        size=16_000,
    ).astype(
        np.float32
    )

    first = measure_audio_quality(
        waveform
    )

    second = measure_audio_quality(
        waveform
    )

    assert first == second


def test_empty_waveform_is_rejected():
    waveform = np.array(
        [],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        measure_audio_quality(
            waveform
        )


def test_non_finite_waveform_is_rejected():
    waveform = np.array(
        [0.0, np.nan, 0.5],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        measure_audio_quality(
            waveform
        )


def test_wrong_dtype_is_rejected():
    waveform = np.zeros(
        100,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="float32",
    ):
        measure_audio_quality(
            waveform
        )