import numpy as np
import pytest

from app.services.features.acoustic_mfcc import (
    ACOUSTIC_MFCC_VERSION,
    LOG_EPSILON,
    MEL_FILTER_COUNT,
    MFCC_COUNT,
    calculate_mfcc,
    create_mel_filterbank,
    extract_mfcc_features,
    hz_to_mel,
    mel_to_hz,
)


def test_mfcc_contract_is_frozen():
    assert (
        ACOUSTIC_MFCC_VERSION
        == "acoustic-mfcc-v1"
    )

    assert MFCC_COUNT == 13
    assert MEL_FILTER_COUNT == 40

    assert LOG_EPSILON == 1e-10


def test_zero_hz_maps_to_zero_mel():
    result = hz_to_mel(
        np.array(
            [0.0],
            dtype=np.float64,
        )
    )

    assert result[0] == pytest.approx(
        0.0
    )


def test_hz_mel_round_trip():
    frequencies = np.array(
        [
            0.0,
            100.0,
            1000.0,
            4000.0,
            8000.0,
        ],
        dtype=np.float64,
    )

    reconstructed = mel_to_hz(
        hz_to_mel(
            frequencies
        )
    )

    np.testing.assert_allclose(
        reconstructed,
        frequencies,
        rtol=1e-10,
        atol=1e-10,
    )


def test_rejects_negative_frequency():
    with pytest.raises(
        ValueError,
        match="negative",
    ):
        hz_to_mel(
            np.array(
                [-1.0],
                dtype=np.float64,
            )
        )


def test_filterbank_shape():
    filterbank = (
        create_mel_filterbank(
            frame_length=400
        )
    )

    assert filterbank.shape == (
        40,
        201,
    )


def test_filterbank_is_nonnegative():
    filterbank = (
        create_mel_filterbank(
            frame_length=400
        )
    )

    assert np.all(
        filterbank >= 0.0
    )


def test_filterbank_values_do_not_exceed_one():
    filterbank = (
        create_mel_filterbank(
            frame_length=400
        )
    )

    assert np.all(
        filterbank <= 1.0
    )


def test_each_filter_has_energy():
    filterbank = (
        create_mel_filterbank(
            frame_length=400
        )
    )

    assert np.all(
        np.sum(
            filterbank,
            axis=1,
        )
        > 0.0
    )


def test_mfcc_shape():
    frames = np.zeros(
        (3, 400),
        dtype=np.float32,
    )

    coefficients = (
        calculate_mfcc(
            frames
        )
    )

    assert coefficients.shape == (
        3,
        13,
    )

    assert (
        coefficients.dtype
        == np.float64
    )


def test_silent_mfcc_is_finite():
    frames = np.zeros(
        (2, 400),
        dtype=np.float32,
    )

    coefficients = (
        calculate_mfcc(
            frames
        )
    )

    assert np.all(
        np.isfinite(
            coefficients
        )
    )


def test_identical_frames_have_identical_mfccs():
    frame = np.linspace(
        -0.5,
        0.5,
        400,
        dtype=np.float32,
    )

    frames = np.stack(
        [
            frame,
            frame,
        ]
    )

    coefficients = (
        calculate_mfcc(
            frames
        )
    )

    np.testing.assert_allclose(
        coefficients[0],
        coefficients[1],
        rtol=0.0,
        atol=0.0,
    )


def test_single_frame_has_zero_mfcc_std():
    frame = np.linspace(
        -0.5,
        0.5,
        400,
        dtype=np.float32,
    )

    frames = frame.reshape(
        1,
        400,
    )

    result = (
        extract_mfcc_features(
            frames
        )
    )

    assert len(
        result.mfcc_mean
    ) == 13

    assert len(
        result.mfcc_std
    ) == 13

    np.testing.assert_allclose(
        result.mfcc_std,
        np.zeros(13),
        rtol=0.0,
        atol=0.0,
    )


def test_empty_frames_return_zero_descriptors():
    frames = np.empty(
        (0, 400),
        dtype=np.float32,
    )

    result = (
        extract_mfcc_features(
            frames
        )
    )

    assert result.mfcc_mean == (
        0.0,
    ) * 13

    assert result.mfcc_std == (
        0.0,
    ) * 13


def test_mfcc_extraction_is_deterministic():
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

    first = extract_mfcc_features(
        frames
    )

    second = extract_mfcc_features(
        frames
    )

    assert first == second


def test_mfcc_does_not_modify_frames():
    rng = np.random.default_rng(
        42
    )

    frames = rng.normal(
        0.0,
        0.1,
        size=(3, 400),
    ).astype(
        np.float32
    )

    original = frames.copy()

    calculate_mfcc(
        frames
    )

    np.testing.assert_array_equal(
        frames,
        original,
    )


def test_rejects_wrong_sample_rate():
    frames = np.zeros(
        (1, 400),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="16 kHz",
    ):
        calculate_mfcc(
            frames,
            sample_rate_hz=44_100,
        )


def test_rejects_invalid_frame_dtype():
    frames = np.zeros(
        (1, 400),
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="float32",
    ):
        calculate_mfcc(
            frames
        )


def test_rejects_nonfinite_frames():
    frames = np.zeros(
        (1, 400),
        dtype=np.float32,
    )

    frames[0, 10] = np.nan

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        calculate_mfcc(
            frames
        )


def test_rejects_mel_max_above_nyquist():
    with pytest.raises(
        ValueError,
        match="Nyquist",
    ):
        create_mel_filterbank(
            frame_length=400,
            sample_rate_hz=16_000,
            max_hz=9_000.0,
        )