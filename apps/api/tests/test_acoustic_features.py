import numpy as np
import pytest

from app.services.features.acoustic_features import (
    ACOUSTIC_FEATURE_VERSION,
    ACOUSTIC_SAMPLE_RATE_HZ,
    FRAME_LENGTH_MS,
    FRAME_LENGTH_SAMPLES,
    HOP_LENGTH_MS,
    HOP_LENGTH_SAMPLES,
    AcousticFeatureResult,
    frame_waveform,
)


def make_result(
    **overrides,
) -> AcousticFeatureResult:
    values = {
        "duration_seconds": 3.0,
        "sample_rate_hz": 16_000,
        "frame_count": 10,
        "signal_activity_ratio": 0.8,
        "signal_inactivity_ratio": 0.2,
        "rms_mean": 0.1,
        "rms_std": 0.02,
        "spectral_centroid_mean_hz": 1500.0,
        "spectral_centroid_std_hz": 200.0,
        "spectral_bandwidth_mean_hz": 1200.0,
        "spectral_bandwidth_std_hz": 150.0,
        "spectral_rolloff_mean_hz": 3000.0,
        "spectral_rolloff_std_hz": 300.0,
        "zero_crossing_rate_mean": 0.1,
        "zero_crossing_rate_std": 0.02,
        "mfcc_mean": tuple(
            0.0 for _ in range(13)
        ),
        "mfcc_std": tuple(
            0.0 for _ in range(13)
        ),
    }

    values.update(overrides)

    return AcousticFeatureResult(
        **values
    )


def test_acoustic_contract_versions():
    assert (
        ACOUSTIC_FEATURE_VERSION
        == "acoustic-features-v1"
    )

    assert (
        ACOUSTIC_SAMPLE_RATE_HZ
        == 16_000
    )

    assert FRAME_LENGTH_MS == 25.0
    assert HOP_LENGTH_MS == 10.0

    assert FRAME_LENGTH_SAMPLES == 400
    assert HOP_LENGTH_SAMPLES == 160


def test_exact_frame_produces_one_frame():
    waveform = np.arange(
        400,
        dtype=np.float32,
    )

    frames = frame_waveform(
        waveform
    )

    assert frames.shape == (1, 400)

    np.testing.assert_array_equal(
        frames[0],
        waveform,
    )


def test_short_waveform_produces_zero_frames():
    waveform = np.arange(
        399,
        dtype=np.float32,
    )

    frames = frame_waveform(
        waveform
    )

    assert frames.shape == (0, 400)


def test_partial_final_frame_is_not_emitted():
    waveform = np.arange(
        559,
        dtype=np.float32,
    )

    frames = frame_waveform(
        waveform
    )

    assert frames.shape == (1, 400)

    np.testing.assert_array_equal(
        frames[0],
        waveform[:400],
    )


def test_second_frame_starts_at_hop_boundary():
    waveform = np.arange(
        560,
        dtype=np.float32,
    )

    frames = frame_waveform(
        waveform
    )

    assert frames.shape == (2, 400)

    np.testing.assert_array_equal(
        frames[0],
        waveform[0:400],
    )

    np.testing.assert_array_equal(
        frames[1],
        waveform[160:560],
    )


def test_multiple_frames_follow_exact_formula():
    waveform = np.arange(
        720,
        dtype=np.float32,
    )

    frames = frame_waveform(
        waveform
    )

    assert frames.shape == (3, 400)

    np.testing.assert_array_equal(
        frames[0],
        waveform[0:400],
    )

    np.testing.assert_array_equal(
        frames[1],
        waveform[160:560],
    )

    np.testing.assert_array_equal(
        frames[2],
        waveform[320:720],
    )


def test_framing_is_deterministic():
    waveform = np.linspace(
        -1.0,
        1.0,
        4000,
        dtype=np.float32,
    )

    first = frame_waveform(
        waveform
    )

    second = frame_waveform(
        waveform
    )

    np.testing.assert_array_equal(
        first,
        second,
    )


def test_framing_does_not_modify_input():
    waveform = np.linspace(
        -1.0,
        1.0,
        1000,
        dtype=np.float32,
    )

    original = waveform.copy()

    frame_waveform(
        waveform
    )

    np.testing.assert_array_equal(
        waveform,
        original,
    )


def test_rejects_non_float32_waveform():
    waveform = np.zeros(
        400,
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="float32",
    ):
        frame_waveform(
            waveform
        )


def test_rejects_multidimensional_waveform():
    waveform = np.zeros(
        (1, 400),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        frame_waveform(
            waveform
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
        frame_waveform(
            waveform
        )


def test_rejects_nonfinite_waveform():
    waveform = np.zeros(
        400,
        dtype=np.float32,
    )

    waveform[10] = np.nan

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        frame_waveform(
            waveform
        )


def test_rejects_invalid_frame_length():
    waveform = np.zeros(
        400,
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="frame_length_samples",
    ):
        frame_waveform(
            waveform,
            frame_length_samples=0,
        )


def test_rejects_invalid_hop_length():
    waveform = np.zeros(
        400,
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="hop_length_samples",
    ):
        frame_waveform(
            waveform,
            hop_length_samples=0,
        )


def test_result_accepts_valid_contract():
    result = make_result()

    assert (
        result.feature_version
        == ACOUSTIC_FEATURE_VERSION
    )

    assert (
        result.sample_rate_hz
        == 16_000
    )

    assert len(result.mfcc_mean) == 13
    assert len(result.mfcc_std) == 13


def test_result_rejects_wrong_sample_rate():
    with pytest.raises(
        ValueError,
        match="16 kHz",
    ):
        make_result(
            sample_rate_hz=44_100
        )


def test_result_rejects_invalid_activity_ratio():
    with pytest.raises(
        ValueError,
        match="signal_activity_ratio",
    ):
        make_result(
            signal_activity_ratio=1.1,
            signal_inactivity_ratio=-0.1,
        )


def test_result_requires_activity_ratios_to_sum_to_one():
    with pytest.raises(
        ValueError,
        match="sum to 1",
    ):
        make_result(
            signal_activity_ratio=0.7,
            signal_inactivity_ratio=0.2,
        )


def test_result_requires_13_mfcc_means():
    with pytest.raises(
        ValueError,
        match="mfcc_mean",
    ):
        make_result(
            mfcc_mean=(0.0,) * 12
        )


def test_result_requires_13_mfcc_stds():
    with pytest.raises(
        ValueError,
        match="mfcc_std",
    ):
        make_result(
            mfcc_std=(0.0,) * 14
        )
        
def test_complete_acoustic_extraction_contract():
    from app.services.features.acoustic_features import (
        extract_acoustic_features,
    )

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
    ).astype(
        np.float32
    )

    result = extract_acoustic_features(
        waveform
    )

    assert (
        result.feature_version
        == ACOUSTIC_FEATURE_VERSION
    )

    assert (
        result.sample_rate_hz
        == 16_000
    )

    assert result.duration_seconds == pytest.approx(
        1.0
    )

    expected_frame_count = (
        1
        + (
            waveform.size
            - FRAME_LENGTH_SAMPLES
        )
        // HOP_LENGTH_SAMPLES
    )

    assert (
        result.frame_count
        == expected_frame_count
    )

    assert (
        0.0
        <= result.signal_activity_ratio
        <= 1.0
    )

    assert (
        result.signal_activity_ratio
        + result.signal_inactivity_ratio
        == pytest.approx(1.0)
    )

    assert np.isfinite(
        result.rms_mean
    )

    assert np.isfinite(
        result.rms_std
    )

    assert np.isfinite(
        result.spectral_centroid_mean_hz
    )

    assert np.isfinite(
        result.spectral_bandwidth_mean_hz
    )

    assert np.isfinite(
        result.spectral_rolloff_mean_hz
    )

    assert np.isfinite(
        result.zero_crossing_rate_mean
    )

    assert len(
        result.mfcc_mean
    ) == 13

    assert len(
        result.mfcc_std
    ) == 13

    assert np.all(
        np.isfinite(
            result.mfcc_mean
        )
    )

    assert np.all(
        np.isfinite(
            result.mfcc_std
        )
    )


def test_complete_acoustic_extraction_is_deterministic():
    from app.services.features.acoustic_features import (
        extract_acoustic_features,
    )

    rng = np.random.default_rng(
        42
    )

    waveform = rng.normal(
        0.0,
        0.1,
        size=16_000,
    ).astype(
        np.float32
    )

    first = extract_acoustic_features(
        waveform
    )

    second = extract_acoustic_features(
        waveform
    )

    assert first == second


def test_complete_acoustic_extraction_does_not_modify_waveform():
    from app.services.features.acoustic_features import (
        extract_acoustic_features,
    )

    waveform = np.linspace(
        -0.5,
        0.5,
        16_000,
        dtype=np.float32,
    )

    original = waveform.copy()

    extract_acoustic_features(
        waveform
    )

    np.testing.assert_array_equal(
        waveform,
        original,
    )


def test_complete_acoustic_extraction_handles_short_waveform():
    from app.services.features.acoustic_features import (
        extract_acoustic_features,
    )

    waveform = np.zeros(
        399,
        dtype=np.float32,
    )

    result = extract_acoustic_features(
        waveform
    )

    assert result.frame_count == 0

    assert (
        result.signal_activity_ratio
        == 0.0
    )

    assert (
        result.signal_inactivity_ratio
        == 1.0
    )

    assert result.rms_mean == 0.0
    assert result.rms_std == 0.0

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

    assert (
        result.zero_crossing_rate_mean
        == 0.0
    )

    assert result.mfcc_mean == (
        0.0,
    ) * 13

    assert result.mfcc_std == (
        0.0,
    ) * 13


def test_complete_acoustic_extraction_rejects_wrong_sample_rate():
    from app.services.features.acoustic_features import (
        extract_acoustic_features,
    )

    waveform = np.zeros(
        16_000,
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="16 kHz",
    ):
        extract_acoustic_features(
            waveform,
            sample_rate_hz=44_100,
        )