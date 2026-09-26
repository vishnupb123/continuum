from dataclasses import dataclass

from app.models.feature_constants import (
    FEATURE_QUALITY_DEGRADED,
    FEATURE_QUALITY_GOOD,
    FEATURE_QUALITY_UNUSABLE,
)
from app.services.features.audio_quality import (
    AudioQualityMeasurements,
)


AUDIO_QUALITY_POLICY_VERSION = (
    "audio-quality-policy-v1"
)

MINIMUM_USABLE_DURATION_SECONDS = 0.5
SHORT_DURATION_SECONDS = 2.0

EFFECTIVELY_SILENT_RMS = 0.001
LOW_SIGNAL_RMS = 0.01

UNUSABLE_SILENCE_RATIO = 0.995
DEGRADED_SILENCE_RATIO = 0.90

DEGRADED_CLIPPING_RATIO = 0.05


@dataclass(frozen=True)
class AudioQualityAssessment:
    status: str
    reasons: tuple[str, ...]
    policy_version: str = (
        AUDIO_QUALITY_POLICY_VERSION
    )


def assess_audio_quality(
    *,
    duration_seconds: float,
    measurements: AudioQualityMeasurements,
) -> AudioQualityAssessment:
    if duration_seconds < 0:
        raise ValueError(
            "duration_seconds must not "
            "be negative"
        )

    unusable_reasons: list[str] = []
    degraded_reasons: list[str] = []

    if (
        duration_seconds
        < MINIMUM_USABLE_DURATION_SECONDS
    ):
        unusable_reasons.append(
            "duration_too_short"
        )

    if (
        measurements.rms_amplitude
        < EFFECTIVELY_SILENT_RMS
    ):
        unusable_reasons.append(
            "effectively_silent"
        )

    if (
        measurements.silence_ratio
        >= UNUSABLE_SILENCE_RATIO
    ):
        unusable_reasons.append(
            "almost_entirely_silent"
        )

    # UNUSABLE takes precedence over every
    # degraded condition.
    if unusable_reasons:
        return AudioQualityAssessment(
            status=FEATURE_QUALITY_UNUSABLE,
            reasons=tuple(
                unusable_reasons
            ),
        )

    if (
        duration_seconds
        < SHORT_DURATION_SECONDS
    ):
        degraded_reasons.append(
            "short_duration"
        )

    if (
        measurements.rms_amplitude
        < LOW_SIGNAL_RMS
    ):
        degraded_reasons.append(
            "low_signal_level"
        )

    if (
        measurements.silence_ratio
        >= DEGRADED_SILENCE_RATIO
    ):
        degraded_reasons.append(
            "high_silence_ratio"
        )

    if (
        measurements.clipping_ratio
        >= DEGRADED_CLIPPING_RATIO
    ):
        degraded_reasons.append(
            "high_clipping_ratio"
        )

    if degraded_reasons:
        return AudioQualityAssessment(
            status=FEATURE_QUALITY_DEGRADED,
            reasons=tuple(
                degraded_reasons
            ),
        )

    return AudioQualityAssessment(
        status=FEATURE_QUALITY_GOOD,
        reasons=tuple(),
    )