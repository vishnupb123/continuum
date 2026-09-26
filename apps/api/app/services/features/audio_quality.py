from dataclasses import dataclass

import numpy as np


AUDIO_QUALITY_VERSION = "audio-quality-v1"

SILENCE_THRESHOLD = 0.01
CLIPPING_THRESHOLD = 0.99


@dataclass(frozen=True)
class AudioQualityMeasurements:
    peak_amplitude: float
    rms_amplitude: float
    silence_ratio: float
    clipping_ratio: float
    quality_version: str = AUDIO_QUALITY_VERSION


def measure_audio_quality(
    waveform: np.ndarray,
) -> AudioQualityMeasurements:
    if waveform.dtype != np.float32:
        raise ValueError(
            "waveform must be float32"
        )

    if waveform.ndim != 1:
        raise ValueError(
            "waveform must be one-dimensional"
        )

    if waveform.size == 0:
        raise ValueError(
            "waveform must not be empty"
        )

    if not np.all(np.isfinite(waveform)):
        raise ValueError(
            "waveform contains non-finite samples"
        )

    absolute = np.abs(waveform)

    peak_amplitude = float(
        np.max(absolute)
    )

    # Compute RMS using float64 internally to avoid
    # unnecessary precision loss during squaring
    # and averaging.
    waveform_64 = waveform.astype(
        np.float64,
        copy=False,
    )

    rms_amplitude = float(
        np.sqrt(
            np.mean(
                np.square(waveform_64)
            )
        )
    )

    silence_ratio = float(
        np.mean(
            absolute < SILENCE_THRESHOLD
        )
    )

    clipping_ratio = float(
        np.mean(
            absolute >= CLIPPING_THRESHOLD
        )
    )

    return AudioQualityMeasurements(
        peak_amplitude=peak_amplitude,
        rms_amplitude=rms_amplitude,
        silence_ratio=silence_ratio,
        clipping_ratio=clipping_ratio,
    )