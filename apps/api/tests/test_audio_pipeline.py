from io import BytesIO

import av
import numpy as np
import pytest

from app.models.feature_constants import (
    FEATURE_QUALITY_GOOD,
    FEATURE_QUALITY_UNUSABLE,
)
from app.services.features.audio_pipeline import (
    AUDIO_PIPELINE_VERSION,
    process_audio,
)
from app.services.features.audio_preprocessing import (
    AUDIO_PREPROCESSING_VERSION,
)
from app.services.features.audio_quality import (
    AUDIO_QUALITY_VERSION,
)
from app.services.features.audio_quality_policy import (
    AUDIO_QUALITY_POLICY_VERSION,
)


def make_wav_bytes(
    *,
    sample_rate_hz: int = 48_000,
    duration_seconds: float = 3.0,
    amplitude: float = 0.25,
) -> bytes:
    sample_count = int(
        sample_rate_hz
        * duration_seconds
    )

    time = (
        np.arange(
            sample_count,
            dtype=np.float32,
        )
        / sample_rate_hz
    )

    signal = (
        amplitude
        * np.sin(
            2.0
            * np.pi
            * 440.0
            * time
        )
    ).astype(
        np.float32
    )

    samples = signal.reshape(
        1,
        -1,
    )

    output = BytesIO()

    with av.open(
        output,
        mode="w",
        format="wav",
    ) as container:
        stream = container.add_stream(
            "pcm_f32le",
            rate=sample_rate_hz,
        )

        stream.layout = "mono"

        frame = av.AudioFrame.from_ndarray(
            samples,
            format="fltp",
            layout="mono",
        )

        frame.sample_rate = (
            sample_rate_hz
        )

        for packet in stream.encode(
            frame
        ):
            container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    return output.getvalue()


def test_pipeline_versions_are_preserved():
    audio_bytes = make_wav_bytes()

    result = process_audio(
        audio_bytes
    )

    assert (
        result.pipeline_version
        == "audio-pipeline-v1"
    )

    assert (
        result.pipeline_version
        == AUDIO_PIPELINE_VERSION
    )

    assert (
        result.preprocessing_version
        == AUDIO_PREPROCESSING_VERSION
    )

    assert (
        result.quality_version
        == AUDIO_QUALITY_VERSION
    )

    assert (
        result.quality_policy_version
        == AUDIO_QUALITY_POLICY_VERSION
    )


def test_good_audio_complete_pipeline():
    audio_bytes = make_wav_bytes(
        sample_rate_hz=48_000,
        duration_seconds=3.0,
        amplitude=0.25,
    )

    result = process_audio(
        audio_bytes
    )

    assert result.sample_rate_hz == 16_000

    assert result.original_sample_rate_hz == (
        48_000
    )

    assert result.original_channels == 1

    assert result.waveform.dtype == np.float32
    assert result.waveform.ndim == 1

    assert result.sample_count == pytest.approx(
        48_000,
        abs=2,
    )

    assert result.duration_seconds == pytest.approx(
        3.0,
        abs=1e-3,
    )

    assert (
        result.quality_status
        == FEATURE_QUALITY_GOOD
    )

    assert result.quality_reasons == tuple()

    assert (
        result.measurements.peak_amplitude
        > 0.0
    )

    assert (
        result.measurements.rms_amplitude
        > 0.0
    )


def test_silent_audio_is_unusable():
    audio_bytes = make_wav_bytes(
        sample_rate_hz=16_000,
        duration_seconds=3.0,
        amplitude=0.0,
    )

    result = process_audio(
        audio_bytes
    )

    assert (
        result.quality_status
        == FEATURE_QUALITY_UNUSABLE
    )

    assert (
        "effectively_silent"
        in result.quality_reasons
    )

    assert (
        "almost_entirely_silent"
        in result.quality_reasons
    )

    assert (
        result.measurements.rms_amplitude
        == 0.0
    )

    assert (
        result.measurements.silence_ratio
        == 1.0
    )


def test_pipeline_is_deterministic():
    audio_bytes = make_wav_bytes(
        sample_rate_hz=48_000,
        duration_seconds=3.0,
        amplitude=0.25,
    )

    first = process_audio(
        audio_bytes
    )

    second = process_audio(
        audio_bytes
    )

    assert (
        first.sample_count
        == second.sample_count
    )

    assert (
        first.duration_seconds
        == second.duration_seconds
    )

    assert (
        first.measurements
        == second.measurements
    )

    assert (
        first.assessment
        == second.assessment
    )

    np.testing.assert_array_equal(
        first.waveform,
        second.waveform,
    )