import numpy as np
import pytest

from app.services.features.acoustic_temporal import (
    ACOUSTIC_TEMPORAL_VERSION,
    SIGNAL_ACTIVITY_RMS_THRESHOLD,
    SIGNAL_ACTIVITY_THRESHOLD_ATOL,
    calculate_frame_rms,
    calculate_zero_crossing_rate,
    extract_temporal_features,
)


def test_temporal_version_is_frozen():
    assert (
        ACOUSTIC_TEMPORAL_VERSION
        == "acoustic-temporal-v1"
    )

    assert (
        SIGNAL_ACTIVITY_RMS_THRESHOLD
        == 0.01
    )

    assert (
        SIGNAL_ACTIVITY_THRESHOLD_ATOL
        == 1e-8
    )


def test_frame_rms_for_constant_frames():
    frames = np.array(
        [
            [
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            [
                -0.25,
                -0.25,
                -0.25,
                -0.25,
            ],
        ],
        dtype=np.float32,
    )

    result = calculate_frame_rms(
        frames
    )

    np.testing.assert_allclose(
        result,
        [
            0.5,
            0.25,
        ],
        rtol=1e-7,
        atol=1e-7,
    )


def test_frame_rms_for_silent_frame():
    frames = np.zeros(
        (1, 400),
        dtype=np.float32,
    )

    result = calculate_frame_rms(
        frames
    )

    assert result.shape == (1,)
    assert result[0] == 0.0


def test_zero_crossing_rate_alternating_signal():
    frames = np.array(
        [
            [
                -1.0,
                1.0,
                -1.0,
                1.0,
            ],
        ],
        dtype=np.float32,
    )

    result = (
        calculate_zero_crossing_rate(
            frames
        )
    )

    assert result.shape == (1,)

    assert result[0] == pytest.approx(
        1.0
    )


def test_zero_crossing_rate_constant_signal():
    frames = np.ones(
        (1, 400),
        dtype=np.float32,
    )

    result = (
        calculate_zero_crossing_rate(
            frames
        )
    )

    assert result[0] == 0.0


def test_zero_is_treated_as_nonnegative():
    frames = np.array(
        [
            [
                -1.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float32,
    )

    result = (
        calculate_zero_crossing_rate(
            frames
        )
    )

    # -1 -> 0 crosses because zero is non-negative.
    #  0 -> +1 does not cross.
    #
    # Therefore:
    #     1 crossing / 2 transitions = 0.5
    assert result[0] == pytest.approx(
        0.5
    )


def test_activity_ratio_all_active():
    waveform = np.full(
        1000,
        0.5,
        dtype=np.float32,
    )

    result = (
        extract_temporal_features(
            waveform
        )
    )

    assert (
        result.signal_activity_ratio
        == pytest.approx(1.0)
    )

    assert (
        result.signal_inactivity_ratio
        == pytest.approx(0.0)
    )


def test_activity_ratio_all_inactive():
    waveform = np.zeros(
        1000,
        dtype=np.float32,
    )

    result = (
        extract_temporal_features(
            waveform
        )
    )

    assert (
        result.signal_activity_ratio
        == pytest.approx(0.0)
    )

    assert (
        result.signal_inactivity_ratio
        == pytest.approx(1.0)
    )


def test_threshold_is_inclusive():
    """
    A signal mathematically equal to the configured RMS threshold
    must be considered active.

    This is also a regression test for float32 representation
    around the 0.01 boundary.
    """

    waveform = np.full(
        400,
        SIGNAL_ACTIVITY_RMS_THRESHOLD,
        dtype=np.float32,
    )

    result = (
        extract_temporal_features(
            waveform
        )
    )

    assert (
        result.signal_activity_ratio
        == pytest.approx(1.0)
    )

    assert (
        result.signal_inactivity_ratio
        == pytest.approx(0.0)
    )


def test_value_meaningfully_below_threshold_is_inactive():
    """
    The numerical tolerance must not materially lower the
    configured activity threshold.
    """

    waveform = np.full(
        400,
        0.009,
        dtype=np.float32,
    )

    result = (
        extract_temporal_features(
            waveform
        )
    )

    assert (
        result.signal_activity_ratio
        == pytest.approx(0.0)
    )

    assert (
        result.signal_inactivity_ratio
        == pytest.approx(1.0)
    )


def test_short_waveform_has_zero_frames():
    waveform = np.zeros(
        399,
        dtype=np.float32,
    )

    result = (
        extract_temporal_features(
            waveform
        )
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
        result.zero_crossing_rate_mean
        == 0.0
    )

    assert (
        result.zero_crossing_rate_std
        == 0.0
    )


def test_temporal_extraction_is_deterministic():
    waveform = np.linspace(
        -0.5,
        0.5,
        5000,
        dtype=np.float32,
    )

    first = (
        extract_temporal_features(
            waveform
        )
    )

    second = (
        extract_temporal_features(
            waveform
        )
    )

    assert first == second


def test_temporal_extraction_does_not_modify_waveform():
    waveform = np.linspace(
        -1.0,
        1.0,
        5000,
        dtype=np.float32,
    )

    original = waveform.copy()

    extract_temporal_features(
        waveform
    )

    np.testing.assert_array_equal(
        waveform,
        original,
    )


def test_rejects_invalid_frame_dtype():
    frames = np.zeros(
        (2, 400),
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="float32",
    ):
        calculate_frame_rms(
            frames
        )


def test_rejects_nonfinite_frames():
    frames = np.zeros(
        (2, 400),
        dtype=np.float32,
    )

    frames[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        calculate_zero_crossing_rate(
            frames
        )