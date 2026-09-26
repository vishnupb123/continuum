import pytest

from app.models.feature_constants import (
    FEATURE_QUALITY_DEGRADED,
    FEATURE_QUALITY_GOOD,
    FEATURE_QUALITY_UNUSABLE,
)
from app.services.features.audio_quality import (
    AudioQualityMeasurements,
)
from app.services.features.audio_quality_policy import (
    AUDIO_QUALITY_POLICY_VERSION,
    assess_audio_quality,
)


def measurements(
    *,
    rms: float = 0.1,
    silence: float = 0.1,
    clipping: float = 0.0,
) -> AudioQualityMeasurements:
    return AudioQualityMeasurements(
        peak_amplitude=0.5,
        rms_amplitude=rms,
        silence_ratio=silence,
        clipping_ratio=clipping,
    )


def test_policy_version_is_frozen():
    assert (
        AUDIO_QUALITY_POLICY_VERSION
        == "audio-quality-policy-v1"
    )


def test_good_recording():
    result = assess_audio_quality(
        duration_seconds=10.0,
        measurements=measurements(),
    )

    assert (
        result.status
        == FEATURE_QUALITY_GOOD
    )
    assert result.reasons == tuple()


def test_too_short_is_unusable():
    result = assess_audio_quality(
        duration_seconds=0.49,
        measurements=measurements(),
    )

    assert (
        result.status
        == FEATURE_QUALITY_UNUSABLE
    )

    assert (
        "duration_too_short"
        in result.reasons
    )


def test_effectively_silent_is_unusable():
    result = assess_audio_quality(
        duration_seconds=10.0,
        measurements=measurements(
            rms=0.0005,
        ),
    )

    assert (
        result.status
        == FEATURE_QUALITY_UNUSABLE
    )

    assert (
        "effectively_silent"
        in result.reasons
    )


def test_almost_entirely_silent_is_unusable():
    result = assess_audio_quality(
        duration_seconds=10.0,
        measurements=measurements(
            silence=0.995,
        ),
    )

    assert (
        result.status
        == FEATURE_QUALITY_UNUSABLE
    )

    assert (
        "almost_entirely_silent"
        in result.reasons
    )


def test_short_recording_is_degraded():
    result = assess_audio_quality(
        duration_seconds=1.5,
        measurements=measurements(),
    )

    assert (
        result.status
        == FEATURE_QUALITY_DEGRADED
    )

    assert (
        "short_duration"
        in result.reasons
    )


def test_low_signal_is_degraded():
    result = assess_audio_quality(
        duration_seconds=10.0,
        measurements=measurements(
            rms=0.005,
        ),
    )

    assert (
        result.status
        == FEATURE_QUALITY_DEGRADED
    )

    assert (
        "low_signal_level"
        in result.reasons
    )


def test_high_silence_is_degraded():
    result = assess_audio_quality(
        duration_seconds=10.0,
        measurements=measurements(
            silence=0.90,
        ),
    )

    assert (
        result.status
        == FEATURE_QUALITY_DEGRADED
    )

    assert (
        "high_silence_ratio"
        in result.reasons
    )


def test_high_clipping_is_degraded():
    result = assess_audio_quality(
        duration_seconds=10.0,
        measurements=measurements(
            clipping=0.05,
        ),
    )

    assert (
        result.status
        == FEATURE_QUALITY_DEGRADED
    )

    assert (
        "high_clipping_ratio"
        in result.reasons
    )


def test_unusable_takes_precedence():
    result = assess_audio_quality(
        duration_seconds=0.2,
        measurements=measurements(
            rms=0.0001,
            silence=1.0,
            clipping=0.50,
        ),
    )

    assert (
        result.status
        == FEATURE_QUALITY_UNUSABLE
    )

    assert (
        "duration_too_short"
        in result.reasons
    )

    assert (
        "effectively_silent"
        in result.reasons
    )

    assert (
        "almost_entirely_silent"
        in result.reasons
    )

    # DEGRADED reasons should not be mixed into
    # an UNUSABLE assessment.
    assert (
        "high_clipping_ratio"
        not in result.reasons
    )


@pytest.mark.parametrize(
    (
        "duration",
        "rms",
        "silence",
        "clipping",
        "expected",
    ),
    [
        # Exactly 0.50 seconds is usable,
        # but still considered short.
        (
            0.50,
            0.1,
            0.1,
            0.0,
            FEATURE_QUALITY_DEGRADED,
        ),

        # Exactly 2.00 seconds is no longer short.
        (
            2.00,
            0.1,
            0.1,
            0.0,
            FEATURE_QUALITY_GOOD,
        ),

        # Exactly 0.001 RMS is not UNUSABLE,
        # but it is still below the LOW_SIGNAL_RMS
        # threshold and therefore DEGRADED.
        (
            10.0,
            0.001,
            0.1,
            0.0,
            FEATURE_QUALITY_DEGRADED,
        ),

        # Exactly 0.01 RMS is not considered
        # low signal.
        (
            10.0,
            0.01,
            0.1,
            0.0,
            FEATURE_QUALITY_GOOD,
        ),

        # Exact degraded silence threshold.
        (
            10.0,
            0.1,
            0.90,
            0.0,
            FEATURE_QUALITY_DEGRADED,
        ),

        # Exact unusable silence threshold.
        (
            10.0,
            0.1,
            0.995,
            0.0,
            FEATURE_QUALITY_UNUSABLE,
        ),

        # Exact clipping threshold.
        (
            10.0,
            0.1,
            0.1,
            0.05,
            FEATURE_QUALITY_DEGRADED,
        ),
    ],
)
def test_policy_boundaries(
    duration,
    rms,
    silence,
    clipping,
    expected,
):
    result = assess_audio_quality(
        duration_seconds=duration,
        measurements=measurements(
            rms=rms,
            silence=silence,
            clipping=clipping,
        ),
    )

    assert result.status == expected


def test_negative_duration_is_rejected():
    with pytest.raises(
        ValueError,
        match="negative",
    ):
        assess_audio_quality(
            duration_seconds=-1.0,
            measurements=measurements(),
        )