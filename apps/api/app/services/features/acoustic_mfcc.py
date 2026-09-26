from dataclasses import dataclass

import numpy as np
from scipy.fft import dct

from app.services.features.acoustic_spectral import (
    SAMPLE_RATE_HZ,
    calculate_magnitude_spectrum,
)


ACOUSTIC_MFCC_VERSION = "acoustic-mfcc-v1"

MFCC_COUNT = 13
MEL_FILTER_COUNT = 40

MEL_MIN_HZ = 0.0
MEL_MAX_HZ = 8_000.0

LOG_EPSILON = 1e-10


@dataclass(frozen=True)
class AcousticMFCCResult:
    mfcc_mean: tuple[float, ...]
    mfcc_std: tuple[float, ...]

    mfcc_version: str = ACOUSTIC_MFCC_VERSION

    def __post_init__(self) -> None:
        if len(self.mfcc_mean) != MFCC_COUNT:
            raise ValueError(
                f"mfcc_mean must contain "
                f"{MFCC_COUNT} coefficients"
            )

        if len(self.mfcc_std) != MFCC_COUNT:
            raise ValueError(
                f"mfcc_std must contain "
                f"{MFCC_COUNT} coefficients"
            )


def hz_to_mel(
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    """
    Convert frequencies from Hz to the Mel scale.

    Formula:

        mel = 2595 * log10(
            1 + hz / 700
        )
    """

    frequencies_hz = np.asarray(
        frequencies_hz,
        dtype=np.float64,
    )

    if np.any(frequencies_hz < 0.0):
        raise ValueError(
            "frequencies_hz cannot be negative"
        )

    return (
        2595.0
        * np.log10(
            1.0
            + frequencies_hz / 700.0
        )
    )


def mel_to_hz(
    mel_values: np.ndarray,
) -> np.ndarray:
    """
    Convert Mel values back to Hz.
    """

    mel_values = np.asarray(
        mel_values,
        dtype=np.float64,
    )

    if np.any(mel_values < 0.0):
        raise ValueError(
            "mel_values cannot be negative"
        )

    return (
        700.0
        * (
            np.power(
                10.0,
                mel_values / 2595.0,
            )
            - 1.0
        )
    )


def create_mel_filterbank(
    *,
    frame_length: int,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    filter_count: int = MEL_FILTER_COUNT,
    min_hz: float = MEL_MIN_HZ,
    max_hz: float = MEL_MAX_HZ,
) -> np.ndarray:
    """
    Create deterministic triangular Mel filters.

    Output shape:

        (
            filter_count,
            frame_length // 2 + 1,
        )
    """

    if frame_length <= 0:
        raise ValueError(
            "frame_length must be greater than zero"
        )

    if sample_rate_hz <= 0:
        raise ValueError(
            "sample_rate_hz must be greater than zero"
        )

    if filter_count <= 0:
        raise ValueError(
            "filter_count must be greater than zero"
        )

    nyquist_hz = (
        sample_rate_hz / 2.0
    )

    if min_hz < 0.0:
        raise ValueError(
            "min_hz cannot be negative"
        )

    if max_hz <= min_hz:
        raise ValueError(
            "max_hz must be greater than min_hz"
        )

    if max_hz > nyquist_hz:
        raise ValueError(
            "max_hz cannot exceed Nyquist frequency"
        )

    min_mel = hz_to_mel(
        np.array(
            [min_hz],
            dtype=np.float64,
        )
    )[0]

    max_mel = hz_to_mel(
        np.array(
            [max_hz],
            dtype=np.float64,
        )
    )[0]

    mel_points = np.linspace(
        min_mel,
        max_mel,
        filter_count + 2,
        dtype=np.float64,
    )

    hz_points = mel_to_hz(
        mel_points
    )

    frequencies_hz = np.fft.rfftfreq(
        frame_length,
        d=1.0 / sample_rate_hz,
    )

    filterbank = np.zeros(
        (
            filter_count,
            frequencies_hz.shape[0],
        ),
        dtype=np.float64,
    )

    for filter_index in range(
        filter_count
    ):
        left_hz = hz_points[
            filter_index
        ]

        center_hz = hz_points[
            filter_index + 1
        ]

        right_hz = hz_points[
            filter_index + 2
        ]

        left_denominator = (
            center_hz - left_hz
        )

        right_denominator = (
            right_hz - center_hz
        )

        rising = (
            frequencies_hz - left_hz
        ) / left_denominator

        falling = (
            right_hz - frequencies_hz
        ) / right_denominator

        filterbank[filter_index] = (
            np.maximum(
                0.0,
                np.minimum(
                    rising,
                    falling,
                ),
            )
        )

    return filterbank


def calculate_mfcc(
    frames: np.ndarray,
    *,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """
    Calculate 13 MFCC coefficients per frame.

    Pipeline:

        Hann-windowed magnitude spectrum
                |
                v
        power spectrum
                |
                v
        40 triangular Mel filters
                |
                v
        log Mel energies
                |
                v
        orthonormal DCT-II
                |
                v
        first 13 coefficients

    Output:
        float64 array:
        (frame_count, 13)
    """

    _validate_frames(frames)

    if sample_rate_hz != SAMPLE_RATE_HZ:
        raise ValueError(
            "MFCC features require 16 kHz audio"
        )

    frame_count = frames.shape[0]

    if frame_count == 0:
        return np.empty(
            (
                0,
                MFCC_COUNT,
            ),
            dtype=np.float64,
        )

    magnitude_spectrum = (
        calculate_magnitude_spectrum(
            frames
        )
    )

    power_spectrum = (
        np.square(
            magnitude_spectrum
        )
        / float(frames.shape[1])
    )

    filterbank = (
        create_mel_filterbank(
            frame_length=frames.shape[1],
            sample_rate_hz=sample_rate_hz,
        )
    )

    mel_energies = (
        power_spectrum
        @ filterbank.T
    )

    log_mel_energies = np.log(
        np.maximum(
            mel_energies,
            LOG_EPSILON,
        )
    )

    coefficients = dct(
        log_mel_energies,
        type=2,
        axis=1,
        norm="ortho",
    )

    return coefficients[
        :,
        :MFCC_COUNT,
    ].astype(
        np.float64,
        copy=False,
    )


def extract_mfcc_features(
    frames: np.ndarray,
    *,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> AcousticMFCCResult:
    """
    Aggregate frame-level MFCCs into mean/std descriptors.
    """

    coefficients = calculate_mfcc(
        frames,
        sample_rate_hz=sample_rate_hz,
    )

    if coefficients.shape[0] == 0:
        zeros = tuple(
            0.0
            for _ in range(MFCC_COUNT)
        )

        return AcousticMFCCResult(
            mfcc_mean=zeros,
            mfcc_std=zeros,
        )

    means = np.mean(
        coefficients,
        axis=0,
    )

    stds = np.std(
        coefficients,
        axis=0,
    )

    return AcousticMFCCResult(
        mfcc_mean=tuple(
            float(value)
            for value in means
        ),
        mfcc_std=tuple(
            float(value)
            for value in stds
        ),
    )


def _validate_frames(
    frames: np.ndarray,
) -> None:
    if not isinstance(
        frames,
        np.ndarray,
    ):
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