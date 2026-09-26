import numpy as np
import pytest

from app.services.features.acoustic_spectral import (
    ACOUSTIC_SPECTRAL_VERSION,
    ROLLOFF_PERCENT,
    SAMPLE_RATE_HZ,
    calculate_frequency_bins,
    calculate_magnitude_spectrum,
    calculate_spectral_bandwidth,
    calculate_spectral_centroid,
    calculate_spectral_rolloff,
    extract_spectral_features,
)


def test_spectral_contract_is_frozen():
    assert (
        ACOUSTIC_SPECTRAL_VERSION
        == "acoustic-spectral-v1"
    )

    assert SAMPLE_RATE_HZ == 16_000

    assert ROLLOFF_PERCENT == 0.85


def test_frequency_bins_for_400_sample_frame():
    frequencies = (
        calculate_frequency_bins(
            400
        )
    )

    assert frequencies.shape == (201,)

    assert frequencies[0] == 0.0

    assert frequencies[-1] == 8000.0

    assert (
        frequencies[1]
        - frequencies[0]
    ) == pytest.approx(40.0)


def test_magnitude_spectrum_shape():
    frames = np.zeros(
        (3, 400),
        dtype=np.float32,
    )

    spectrum = (
        calculate_magnitude_spectrum(
            frames
        )
    )

    assert spectrum.shape == (
        3,
        201,
    )

    assert spectrum.dtype == np.float64


def test_silent_frame_has_zero_spectrum():
    frames = np.zeros(
        (1, 400),
        dtype=np.float32,
    )

    spectrum = (
        calculate_magnitude_spectrum(
            frames
        )
    )

    np.testing.assert_array_equal(
        spectrum,
        np.zeros(
            (1, 201),
            dtype=np.float64,
        ),
    )


def test_centroid_for_single_frequency_bin():
    spectrum = np.array(
        [
            [
                0.0,
                0.0,
                5.0,
                0.0,
            ],
        ],
        dtype=np.float64,
    )

    frequencies = np.array(
        [
            0.0,
            100.0,
            200.0,
            300.0,
        ],
        dtype=np.float64,
    )

    centroid = (
        calculate_spectral_centroid(
            spectrum,
            frequencies,
        )
    )

    assert centroid[0] == pytest.approx(
        200.0
    )


def test_centroid_for_equal_two_bin_energy():
    spectrum = np.array(
        [
            [
                0.0,
                1.0,
                1.0,
                0.0,
            ],
        ],
        dtype=np.float64,
    )

    frequencies = np.array(
        [
            0.0,
            100.0,
            300.0,
            500.0,
        ],
        dtype=np.float64,
    )

    centroid = (
        calculate_spectral_centroid(
            spectrum,
            frequencies,
        )
    )

    assert centroid[0] == pytest.approx(
        200.0
    )


def test_silent_spectrum_has_zero_centroid():
    spectrum = np.zeros(
        (1, 4),
        dtype=np.float64,
    )

    frequencies = np.array(
        [
            0.0,
            100.0,
            200.0,
            300.0,
        ],
        dtype=np.float64,
    )

    centroid = (
        calculate_spectral_centroid(
            spectrum,
            frequencies,
        )
    )

    assert centroid[0] == 0.0


def test_bandwidth_for_single_frequency_bin_is_zero():
    spectrum = np.array(
        [
            [
                0.0,
                0.0,
                5.0,
                0.0,
            ],
        ],
        dtype=np.float64,
    )

    frequencies = np.array(
        [
            0.0,
            100.0,
            200.0,
            300.0,
        ],
        dtype=np.float64,
    )

    centroids = np.array(
        [200.0],
        dtype=np.float64,
    )

    bandwidth = (
        calculate_spectral_bandwidth(
            spectrum,
            frequencies,
            centroids,
        )
    )

    assert bandwidth[0] == 0.0


def test_bandwidth_for_symmetric_two_bin_spectrum():
    spectrum = np.array(
        [
            [
                1.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )

    frequencies = np.array(
        [
            100.0,
            300.0,
        ],
        dtype=np.float64,
    )

    centroids = np.array(
        [200.0],
        dtype=np.float64,
    )

    bandwidth = (
        calculate_spectral_bandwidth(
            spectrum,
            frequencies,
            centroids,
        )
    )

    assert bandwidth[0] == pytest.approx(
        100.0
    )


def test_rolloff_uses_first_bin_reaching_threshold():
    spectrum = np.array(
        [
            [
                1.0,
                1.0,
                8.0,
            ],
        ],
        dtype=np.float64,
    )

    frequencies = np.array(
        [
            0.0,
            100.0,
            200.0,
        ],
        dtype=np.float64,
    )

    rolloff = (
        calculate_spectral_rolloff(
            spectrum,
            frequencies,
            rolloff_percent=0.85,
        )
    )

    # Total = 10.
    # 85% = 8.5.
    #
    # cumulative:
    # 1, 2, 10
    #
    # First bin reaching 8.5 = 200 Hz.
    assert rolloff[0] == pytest.approx(
        200.0
    )


def test_silent_spectrum_has_zero_rolloff():
    spectrum = np.zeros(
        (1, 4),
        dtype=np.float64,
    )

    frequencies = np.array(
        [
            0.0,
            100.0,
            200.0,
            300.0,
        ],
        dtype=np.float64,
    )

    rolloff = (
        calculate_spectral_rolloff(
            spectrum,
            frequencies,
        )
    )

    assert rolloff[0] == 0.0


def test_extract_spectral_features_for_silence():
    frames = np.zeros(
        (2, 400),
        dtype=np.float32,
    )

    result = (
        extract_spectral_features(
            frames
        )
    )

    assert (
        result.spectral_centroid_mean_hz
        == 0.0
    )

    assert (
        result.spectral_centroid_std_hz
        == 0.0
    )

    assert (
        result.spectral_bandwidth_mean_hz
        == 0.0
    )

    assert (
        result.spectral_bandwidth_std_hz
        == 0.0
    )

    assert (
        result.spectral_rolloff_mean_hz
        == 0.0
    )

    assert (
        result.spectral_rolloff_std_hz
        == 0.0
    )


def test_empty_frame_collection_returns_zero_statistics():
    frames = np.empty(
        (0, 400),
        dtype=np.float32,
    )

    result = (
        extract_spectral_features(
            frames
        )
    )

    assert (
        result.spectral_centroid_mean_hz
        == 0.0
    )

    assert (
        result.spectral_bandwidth_mean_hz
        == 0.0
    )

    assert (
        result.spectral_rolloff_mean_hz
        == 0.0
    )


def test_sine_wave_centroid_is_near_expected_frequency():
    sample_count = 400

    time = (
        np.arange(
            sample_count,
            dtype=np.float64,
        )
        / SAMPLE_RATE_HZ
    )

    waveform = (
        np.sin(
            2.0
            * np.pi
            * 1000.0
            * time
        )
        .astype(np.float32)
    )

    frames = waveform.reshape(
        1,
        sample_count,
    )

    result = (
        extract_spectral_features(
            frames
        )
    )

    assert (
        result.spectral_centroid_mean_hz
        == pytest.approx(
            1000.0,
            abs=10.0,
        )
    )


def test_spectral_extraction_is_deterministic():
    rng = np.random.default_rng(
        42
    )

    frames = rng.normal(
        0.0,
        0.1,
        size=(5, 400),
    ).astype(
        np.float32
    )

    first = (
        extract_spectral_features(
            frames
        )
    )

    second = (
        extract_spectral_features(
            frames
        )
    )

    assert first == second


def test_rejects_wrong_sample_rate():
    frames = np.zeros(
        (1, 400),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="16 kHz",
    ):
        extract_spectral_features(
            frames,
            sample_rate_hz=44_100,
        )


def test_rejects_invalid_rolloff_percent():
    spectrum = np.ones(
        (1, 4),
        dtype=np.float64,
    )

    frequencies = np.array(
        [
            0.0,
            100.0,
            200.0,
            300.0,
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="rolloff_percent",
    ):
        calculate_spectral_rolloff(
            spectrum,
            frequencies,
            rolloff_percent=0.0,
        )


def test_rejects_negative_spectrum():
    spectrum = np.array(
        [
            [
                0.0,
                -1.0,
                2.0,
            ],
        ],
        dtype=np.float64,
    )

    frequencies = np.array(
        [
            0.0,
            100.0,
            200.0,
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        calculate_spectral_centroid(
            spectrum,
            frequencies,
        )