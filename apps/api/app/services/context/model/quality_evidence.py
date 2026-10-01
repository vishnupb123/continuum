from app.models.feature_constants import (
    FEATURE_QUALITY_DEGRADED,
    FEATURE_QUALITY_GOOD,
)


QUALITY_FEATURE_VALUES = {
    FEATURE_QUALITY_GOOD: 1.0,
    FEATURE_QUALITY_DEGRADED: 0.5,
}


class QualityEvidenceError(ValueError):
    """Raised when M3 quality cannot be used as M4 evidence."""


def encode_quality_evidence(
    quality: str,
) -> float:
    """
    Convert an accepted M3 quality label into an ordinal
    M4 evidence feature.

    Mapping:
        GOOD      -> 1.0
        DEGRADED  -> 0.5

    These values are model features.

    They are not calibrated probabilities, reliability
    percentages, or confidence scores.
    """

    try:
        return QUALITY_FEATURE_VALUES[
            quality
        ]
    except KeyError as exc:
        raise QualityEvidenceError(
            "Evidence quality must be GOOD or DEGRADED"
        ) from exc