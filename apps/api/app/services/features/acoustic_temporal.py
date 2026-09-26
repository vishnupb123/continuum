from dataclasses import dataclass

import numpy as np

from app.services.features.acoustic_features import (
    frame_waveform,
)


ACOUSTIC_TEMPORAL_VERSION = "acoustic-temporal-v1"

# Engineering signal-activity threshold.
#
# IMPORTANT:
# This is NOT voice activity detection (VAD) and must not be
# interpreted as a speech ratio.
SIGNAL_ACTIVITY_RMS_THRESHOLD = 0.01

# Numerical tolerance used only for the inclusive threshold
# comparison.
#
# A float32 waveform containing exactly 0.01 may produce an RMS
# value microscopically below 0.01 after numerical operations.
# This tolerance protects the intended >= threshold contract
# without materially changing the threshold itself.
SIGNAL_ACTIVITY_THRESHOLD_ATOL = 1e-8


@dataclass(frozen=True)
class AcousticTemporalResult:
    """
    Deterministic temporal and energy descriptors extracted from
    canonical audio.

    These values describe measurable properties of the signal.
    They do not represent speech detection, emotion inference,
    psychological state, or diagnostic conclusions.
    """

    frame_count: int

    signal_activity_ratio: float
    signal_inactivity_ratio: float

    rms_mean: float
    rms_std: float

    zero_crossing_rate_mean: float
    zero_crossing_rate_std: float

    temporal_version: str = ACOUSTIC_TEMPORAL_VERSION


def calculate_frame_rms(
    frames: np.ndarray,
) -> np.ndarray:
    """
    Calculate RMS amplitude independently for each frame.

    Expected input shape:
        (frame_count, frame_length)

    Returns:
        float64 array with shape:
        (frame_count,)
    """

    _validate_frames(frames)

    if frames.shape[0] == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    # Perform RMS calculations in float64 so that squaring and
    # accumulation are numerically more stable than float32.
    frames_float64 = frames.astype(
        np.float64,
        copy=False,
    )

    return np.sqrt(
        np.mean(
            np.square(frames_float64),
            axis=1,
        )
    )


def calculate_zero_crossing_rate(
    frames: np.ndarray,
) -> np.ndarray:
    """
    Calculate zero-crossing rate independently for each frame.

    A crossing occurs when consecutive samples move between
    negative and non-negative values.

    Zero is treated as non-negative.

    Rate denominator:
        frame_length - 1
    """

    _validate_frames(frames)

    frame_count = frames.shape[0]
    frame_length = frames.shape[1]

    if frame_count == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    if frame_length < 2:
        return np.zeros(
            frame_count,
            dtype=np.float64,
        )

    # True  -> sample >= 0
    # False -> sample < 0
    signs = frames >= 0.0

    crossings = np.count_nonzero(
        signs[:, 1:]
        != signs[:, :-1],
        axis=1,
    )

    return (
        crossings.astype(np.float64)
        / float(frame_length - 1)
    )


def extract_temporal_features(
    waveform: np.ndarray,
) -> AcousticTemporalResult:
    """
    Extract deterministic temporal and energy descriptors from a
    canonical M3.4 waveform.

    Pipeline:

        canonical waveform
                |
                v
        deterministic framing
                |
                +--> frame RMS
                |
                +--> signal activity ratio
                |
                +--> zero-crossing rate
                |
                v
        aggregated temporal descriptors

    This performs signal measurement only.

    It does NOT perform:
    - voice activity detection
    - speech detection
    - speaker identification
    - emotion inference
    - psychological inference
    - diagnostic inference
    """

    frames = frame_waveform(
        waveform
    )

    frame_count = frames.shape[0]

    # A canonical waveform can be valid while still being shorter
    # than one 25 ms acoustic frame.
    #
    # In that case we return deterministic neutral measurements
    # rather than inventing a padded frame.
    if frame_count == 0:
        return AcousticTemporalResult(
            frame_count=0,
            signal_activity_ratio=0.0,
            signal_inactivity_ratio=1.0,
            rms_mean=0.0,
            rms_std=0.0,
            zero_crossing_rate_mean=0.0,
            zero_crossing_rate_std=0.0,
        )

    frame_rms = calculate_frame_rms(
        frames
    )

    zero_crossing_rates = (
        calculate_zero_crossing_rate(
            frames
        )
    )

    # The conceptual contract is:
    #
    #     RMS >= 0.01 -> active
    #
    # The tiny tolerance accounts only for floating-point
    # representation error at the exact threshold boundary.
    activity_boundary = (
        SIGNAL_ACTIVITY_RMS_THRESHOLD
        - SIGNAL_ACTIVITY_THRESHOLD_ATOL
    )

    active_mask = (
        frame_rms
        >= activity_boundary
    )

    signal_activity_ratio = float(
        np.mean(active_mask)
    )

    signal_inactivity_ratio = (
        1.0
        - signal_activity_ratio
    )

    return AcousticTemporalResult(
        frame_count=frame_count,
        signal_activity_ratio=(
            signal_activity_ratio
        ),
        signal_inactivity_ratio=(
            signal_inactivity_ratio
        ),
        rms_mean=float(
            np.mean(frame_rms)
        ),
        rms_std=float(
            np.std(frame_rms)
        ),
        zero_crossing_rate_mean=float(
            np.mean(
                zero_crossing_rates
            )
        ),
        zero_crossing_rate_std=float(
            np.std(
                zero_crossing_rates
            )
        ),
    )


def _validate_frames(
    frames: np.ndarray,
) -> None:
    """
    Validate the internal acoustic frame contract.
    """

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