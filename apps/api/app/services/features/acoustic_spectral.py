from dataclasses import dataclass

import numpy as np


ACOUSTIC_SPECTRAL_VERSION = "acoustic-spectral-v1"

SAMPLE_RATE_HZ = 16_000
ROLLOFF_PERCENT = 0.85


@dataclass(frozen=True)
class AcousticSpectralResult:
    spectral_centroid_mean_hz: float
    spectral_centroid_std_hz: float

    spectral_bandwidth_mean_hz: float
    spectral_bandwidth_std_hz: float

    spectral_rolloff_mean_hz: float
    spectral_rolloff_std_hz: float

    spectral_version: str = ACOUSTIC_SPECTRAL_VERSION


def calculate_magnitude_spectrum(
    frames: np.ndarray,
) -> np.ndarray:
    """
    Calculate the one-sided magnitude spectrum for every frame.

    A Hann window is applied before the real FFT.

    Input:
        float32 array:
        (frame_count, frame_length)

    Output:
        float64 array:
        (frame_count, frequency_bins)
    """

    _validate_frames(frames)

    frame_count = frames.shape[0]
    frame_length = frames.shape[1]

    if frame_count == 0:
        return np.empty(
            (
                0,
                frame_length // 2 + 1,
            ),
            dtype=np.float64,
        )

    window = np.hanning(
        frame_length
    ).astype(
        np.float64,
        copy=False,
    )

    windowed = (
        frames.astype(
            np.float64,
            copy=False,
        )
        * window
    )

    spectrum = np.fft.rfft(
        windowed,
        axis=1,
    )

    return np.abs(
        spectrum
    )


def calculate_frequency_bins(
    frame_length: int,
    *,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> np.ndarray:
    if frame_length <= 0:
        raise ValueError(
            "frame_length must be greater than zero"
        )

    if sample_rate_hz <= 0:
        raise ValueError(
            "sample_rate_hz must be greater than zero"
        )

    return np.fft.rfftfreq(
        frame_length,
        d=1.0 / sample_rate_hz,
    )


def calculate_spectral_centroid(
    magnitude_spectrum: np.ndarray,
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    """
    Spectral centroid:

        sum(frequency * magnitude)
        --------------------------
             sum(magnitude)

    Silent spectra receive centroid 0 Hz.
    """

    _validate_spectrum(
        magnitude_spectrum,
        frequencies_hz,
    )

    if magnitude_spectrum.shape[0] == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    magnitude_sum = np.sum(
        magnitude_spectrum,
        axis=1,
    )

    weighted_sum = np.sum(
        magnitude_spectrum
        * frequencies_hz[None, :],
        axis=1,
    )

    centroid = np.zeros(
        magnitude_spectrum.shape[0],
        dtype=np.float64,
    )

    np.divide(
        weighted_sum,
        magnitude_sum,
        out=centroid,
        where=magnitude_sum > 0.0,
    )

    return centroid


def calculate_spectral_bandwidth(
    magnitude_spectrum: np.ndarray,
    frequencies_hz: np.ndarray,
    centroids_hz: np.ndarray,
) -> np.ndarray:
    """
    Spectral bandwidth:

        sqrt(
            sum(
                magnitude
                * (frequency - centroid)^2
            )
            / sum(magnitude)
        )

    Silent spectra receive bandwidth 0 Hz.
    """

    _validate_spectrum(
        magnitude_spectrum,
        frequencies_hz,
    )

    if centroids_hz.ndim != 1:
        raise ValueError(
            "centroids_hz must be one-dimensional"
        )

    if (
        centroids_hz.shape[0]
        != magnitude_spectrum.shape[0]
    ):
        raise ValueError(
            "centroid count must match frame count"
        )

    if not np.all(
        np.isfinite(centroids_hz)
    ):
        raise ValueError(
            "centroids_hz must contain only finite values"
        )

    if magnitude_spectrum.shape[0] == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    magnitude_sum = np.sum(
        magnitude_spectrum,
        axis=1,
    )

    frequency_delta = (
        frequencies_hz[None, :]
        - centroids_hz[:, None]
    )

    weighted_variance = np.sum(
        magnitude_spectrum
        * np.square(frequency_delta),
        axis=1,
    )

    variance = np.zeros(
        magnitude_spectrum.shape[0],
        dtype=np.float64,
    )

    np.divide(
        weighted_variance,
        magnitude_sum,
        out=variance,
        where=magnitude_sum > 0.0,
    )

    return np.sqrt(
        variance
    )


def calculate_spectral_rolloff(
    magnitude_spectrum: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    rolloff_percent: float = ROLLOFF_PERCENT,
) -> np.ndarray:
    """
    Return the lowest frequency whose cumulative magnitude reaches
    the configured percentage of total frame magnitude.

    Silent spectra receive rolloff 0 Hz.
    """

    _validate_spectrum(
        magnitude_spectrum,
        frequencies_hz,
    )

    if not (
        0.0 < rolloff_percent <= 1.0
    ):
        raise ValueError(
            "rolloff_percent must be in (0, 1]"
        )

    frame_count = (
        magnitude_spectrum.shape[0]
    )

    if frame_count == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    cumulative = np.cumsum(
        magnitude_spectrum,
        axis=1,
    )

    totals = cumulative[:, -1]

    thresholds = (
        totals
        * rolloff_percent
    )

    reached = (
        cumulative
        >= thresholds[:, None]
    )

    indices = np.argmax(
        reached,
        axis=1,
    )

    rolloff = frequencies_hz[
        indices
    ].astype(
        np.float64,
        copy=True,
    )

    rolloff[
        totals <= 0.0
    ] = 0.0

    return rolloff


def extract_spectral_features(
    frames: np.ndarray,
    *,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> AcousticSpectralResult:
    """
    Extract deterministic aggregate spectral descriptors.
    """

    _validate_frames(frames)

    if sample_rate_hz != SAMPLE_RATE_HZ:
        raise ValueError(
            "spectral features require 16 kHz audio"
        )

    if frames.shape[0] == 0:
        return AcousticSpectralResult(
            spectral_centroid_mean_hz=0.0,
            spectral_centroid_std_hz=0.0,
            spectral_bandwidth_mean_hz=0.0,
            spectral_bandwidth_std_hz=0.0,
            spectral_rolloff_mean_hz=0.0,
            spectral_rolloff_std_hz=0.0,
        )

    magnitude_spectrum = (
        calculate_magnitude_spectrum(
            frames
        )
    )

    frequencies_hz = (
        calculate_frequency_bins(
            frames.shape[1],
            sample_rate_hz=sample_rate_hz,
        )
    )

    centroids = (
        calculate_spectral_centroid(
            magnitude_spectrum,
            frequencies_hz,
        )
    )

    bandwidths = (
        calculate_spectral_bandwidth(
            magnitude_spectrum,
            frequencies_hz,
            centroids,
        )
    )

    rolloffs = (
        calculate_spectral_rolloff(
            magnitude_spectrum,
            frequencies_hz,
        )
    )

    return AcousticSpectralResult(
        spectral_centroid_mean_hz=float(
            np.mean(centroids)
        ),
        spectral_centroid_std_hz=float(
            np.std(centroids)
        ),
        spectral_bandwidth_mean_hz=float(
            np.mean(bandwidths)
        ),
        spectral_bandwidth_std_hz=float(
            np.std(bandwidths)
        ),
        spectral_rolloff_mean_hz=float(
            np.mean(rolloffs)
        ),
        spectral_rolloff_std_hz=float(
            np.std(rolloffs)
        ),
    )


def _validate_frames(
    frames: np.ndarray,
) -> None:
    if not isinstance(frames, np.ndarray):
        raise TypeError(
            "frames must be a numpy array"
        )

    if frames.dtype != np.float32:
        raise ValueError(
            "frames must have dtype float32"
        )

    if frames.ndim != 2:
        raise ValueError(
            "frames must be two-dimensional"
        )

    if frames.shape[1] == 0:
        raise ValueError(
            "frames cannot have zero length"
        )

    if not np.all(
        np.isfinite(frames)
    ):
        raise ValueError(
            "frames must contain only finite values"
        )


def _validate_spectrum(
    magnitude_spectrum: np.ndarray,
    frequencies_hz: np.ndarray,
) -> None:
    if magnitude_spectrum.ndim != 2:
        raise ValueError(
            "magnitude_spectrum must be two-dimensional"
        )

    if frequencies_hz.ndim != 1:
        raise ValueError(
            "frequencies_hz must be one-dimensional"
        )

    if (
        magnitude_spectrum.shape[1]
        != frequencies_hz.shape[0]
    ):
        raise ValueError(
            "frequency count must match spectrum bins"
        )

    if not np.all(
        np.isfinite(magnitude_spectrum)
    ):
        raise ValueError(
            "magnitude_spectrum must contain only finite values"
        )

    if np.any(
        magnitude_spectrum < 0.0
    ):
        raise ValueError(
            "magnitude_spectrum cannot contain negative values"
        )

    if not np.all(
        np.isfinite(frequencies_hz)
    ):
        raise ValueError(
            "frequencies_hz must contain only finite values"
        )