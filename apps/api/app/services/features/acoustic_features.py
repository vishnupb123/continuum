from dataclasses import dataclass

import numpy as np


ACOUSTIC_FEATURE_VERSION = "acoustic-features-v1"

ACOUSTIC_SAMPLE_RATE_HZ = 16_000

FRAME_LENGTH_MS = 25.0
HOP_LENGTH_MS = 10.0

FRAME_LENGTH_SAMPLES = 400
HOP_LENGTH_SAMPLES = 160

MFCC_COUNT = 13


@dataclass(frozen=True)
class AcousticFeatureResult:
    """
    Deterministic hand-engineered acoustic descriptors.

    This object describes measurable properties of an audio signal.
    It does not contain psychological, emotional, diagnostic, or
    speaker-identity conclusions.
    """

    duration_seconds: float
    sample_rate_hz: int
    frame_count: int

    signal_activity_ratio: float
    signal_inactivity_ratio: float

    rms_mean: float
    rms_std: float

    spectral_centroid_mean_hz: float
    spectral_centroid_std_hz: float

    spectral_bandwidth_mean_hz: float
    spectral_bandwidth_std_hz: float

    spectral_rolloff_mean_hz: float
    spectral_rolloff_std_hz: float

    zero_crossing_rate_mean: float
    zero_crossing_rate_std: float

    mfcc_mean: tuple[float, ...]
    mfcc_std: tuple[float, ...]

    feature_version: str = ACOUSTIC_FEATURE_VERSION

    def __post_init__(self) -> None:
        if self.sample_rate_hz != ACOUSTIC_SAMPLE_RATE_HZ:
            raise ValueError(
                "acoustic features require 16 kHz canonical audio"
            )

        if self.duration_seconds < 0:
            raise ValueError(
                "duration_seconds cannot be negative"
            )

        if self.frame_count < 0:
            raise ValueError(
                "frame_count cannot be negative"
            )

        if not (
            0.0 <= self.signal_activity_ratio <= 1.0
        ):
            raise ValueError(
                "signal_activity_ratio must be between 0 and 1"
            )

        if not (
            0.0 <= self.signal_inactivity_ratio <= 1.0
        ):
            raise ValueError(
                "signal_inactivity_ratio must be between 0 and 1"
            )

        if not np.isclose(
            self.signal_activity_ratio
            + self.signal_inactivity_ratio,
            1.0,
            atol=1e-6,
        ):
            raise ValueError(
                "signal activity and inactivity ratios must sum to 1"
            )

        if len(self.mfcc_mean) != MFCC_COUNT:
            raise ValueError(
                f"mfcc_mean must contain {MFCC_COUNT} coefficients"
            )

        if len(self.mfcc_std) != MFCC_COUNT:
            raise ValueError(
                f"mfcc_std must contain {MFCC_COUNT} coefficients"
            )

        scalar_values = (
            self.rms_mean,
            self.rms_std,
            self.spectral_centroid_mean_hz,
            self.spectral_centroid_std_hz,
            self.spectral_bandwidth_mean_hz,
            self.spectral_bandwidth_std_hz,
            self.spectral_rolloff_mean_hz,
            self.spectral_rolloff_std_hz,
            self.zero_crossing_rate_mean,
            self.zero_crossing_rate_std,
        )

        if not all(
            np.isfinite(value)
            for value in scalar_values
        ):
            raise ValueError(
                "acoustic scalar features must be finite"
            )

        if not all(
            np.isfinite(value)
            for value in self.mfcc_mean
        ):
            raise ValueError(
                "mfcc_mean must contain only finite values"
            )

        if not all(
            np.isfinite(value)
            for value in self.mfcc_std
        ):
            raise ValueError(
                "mfcc_std must contain only finite values"
            )


def frame_waveform(
    waveform: np.ndarray,
    *,
    frame_length_samples: int = FRAME_LENGTH_SAMPLES,
    hop_length_samples: int = HOP_LENGTH_SAMPLES,
) -> np.ndarray:
    """
    Split a canonical waveform into deterministic overlapping frames.

    Short final fragments are not emitted.
    No zero-padding is performed.
    """

    if not isinstance(waveform, np.ndarray):
        raise TypeError(
            "waveform must be a numpy array"
        )

    if waveform.dtype != np.float32:
        raise ValueError(
            "waveform must have dtype float32"
        )

    if waveform.ndim != 1:
        raise ValueError(
            "waveform must be one-dimensional"
        )

    if waveform.size == 0:
        raise ValueError(
            "waveform cannot be empty"
        )

    if not np.all(
        np.isfinite(waveform)
    ):
        raise ValueError(
            "waveform must contain only finite values"
        )

    if frame_length_samples <= 0:
        raise ValueError(
            "frame_length_samples must be greater than zero"
        )

    if hop_length_samples <= 0:
        raise ValueError(
            "hop_length_samples must be greater than zero"
        )

    if waveform.size < frame_length_samples:
        return np.empty(
            (0, frame_length_samples),
            dtype=np.float32,
        )

    frame_count = (
        1
        + (
            waveform.size
            - frame_length_samples
        )
        // hop_length_samples
    )

    frames = np.empty(
        (
            frame_count,
            frame_length_samples,
        ),
        dtype=np.float32,
    )

    for frame_index in range(
        frame_count
    ):
        start = (
            frame_index
            * hop_length_samples
        )

        end = (
            start
            + frame_length_samples
        )

        frames[frame_index] = (
            waveform[start:end]
        )

    return frames


def extract_acoustic_features(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int = ACOUSTIC_SAMPLE_RATE_HZ,
) -> AcousticFeatureResult:
    """
    Execute the complete deterministic acoustic-features-v1
    pipeline.

    canonical waveform
            |
            +--> deterministic framing
            |
            +--> temporal / energy descriptors
            |
            +--> spectral descriptors
            |
            +--> MFCC descriptors
            |
            v
    AcousticFeatureResult

    This function performs acoustic measurement only.

    It does not perform:
    - VAD / speech detection
    - speaker identification
    - emotion classification
    - psychological inference
    - diagnostic inference
    - learned neural embedding extraction
    """

    if sample_rate_hz != ACOUSTIC_SAMPLE_RATE_HZ:
        raise ValueError(
            "acoustic features require 16 kHz canonical audio"
        )

    # Local imports intentionally avoid circular imports:
    #
    # acoustic_temporal -> acoustic_features.frame_waveform
    #
    # Keeping them here allows acoustic_features to remain the
    # public composition boundary.
    from app.services.features.acoustic_mfcc import (
        extract_mfcc_features,
    )
    from app.services.features.acoustic_spectral import (
        extract_spectral_features,
    )
    from app.services.features.acoustic_temporal import (
        extract_temporal_features,
    )

    frames = frame_waveform(
        waveform
    )

    temporal = (
        extract_temporal_features(
            waveform
        )
    )

    spectral = (
        extract_spectral_features(
            frames,
            sample_rate_hz=sample_rate_hz,
        )
    )

    mfcc = (
        extract_mfcc_features(
            frames,
            sample_rate_hz=sample_rate_hz,
        )
    )

    duration_seconds = (
        waveform.shape[0]
        / float(sample_rate_hz)
    )

    if temporal.frame_count != frames.shape[0]:
        raise RuntimeError(
            "temporal frame count does not match acoustic framing"
        )

    return AcousticFeatureResult(
        duration_seconds=float(
            duration_seconds
        ),
        sample_rate_hz=sample_rate_hz,
        frame_count=int(
            frames.shape[0]
        ),
        signal_activity_ratio=(
            temporal.signal_activity_ratio
        ),
        signal_inactivity_ratio=(
            temporal.signal_inactivity_ratio
        ),
        rms_mean=temporal.rms_mean,
        rms_std=temporal.rms_std,
        spectral_centroid_mean_hz=(
            spectral.spectral_centroid_mean_hz
        ),
        spectral_centroid_std_hz=(
            spectral.spectral_centroid_std_hz
        ),
        spectral_bandwidth_mean_hz=(
            spectral.spectral_bandwidth_mean_hz
        ),
        spectral_bandwidth_std_hz=(
            spectral.spectral_bandwidth_std_hz
        ),
        spectral_rolloff_mean_hz=(
            spectral.spectral_rolloff_mean_hz
        ),
        spectral_rolloff_std_hz=(
            spectral.spectral_rolloff_std_hz
        ),
        zero_crossing_rate_mean=(
            temporal.zero_crossing_rate_mean
        ),
        zero_crossing_rate_std=(
            temporal.zero_crossing_rate_std
        ),
        mfcc_mean=mfcc.mfcc_mean,
        mfcc_std=mfcc.mfcc_std,
    )