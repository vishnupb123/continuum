import pytest

from app.services.context.model.quality_evidence import (
    QUALITY_FEATURE_VALUES,
    QualityEvidenceError,
    encode_quality_evidence,
)


def test_good_quality_maps_to_one():
    assert encode_quality_evidence(
        "GOOD"
    ) == 1.0


def test_degraded_quality_maps_to_half():
    assert encode_quality_evidence(
        "DEGRADED"
    ) == 0.5


def test_quality_mapping_is_frozen():
    assert QUALITY_FEATURE_VALUES == {
        "GOOD": 1.0,
        "DEGRADED": 0.5,
    }


@pytest.mark.parametrize(
    "quality",
    [
        "UNUSABLE",
        "UNKNOWN",
        "",
    ],
)
def test_unsupported_quality_is_rejected(
    quality,
):
    with pytest.raises(
        QualityEvidenceError,
        match="GOOD or DEGRADED",
    ):
        encode_quality_evidence(
            quality
        )
        