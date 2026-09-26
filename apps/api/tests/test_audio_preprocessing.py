from io import BytesIO

import av
import numpy as np
import pytest

from app.services.features.audio_preprocessing import (
    AUDIO_PREPROCESSING_VERSION,
    CANONICAL_SAMPLE_RATE_HZ,
    AudioPreprocessingError,
    preprocess_audio,
)


def make_wav_bytes(
    *,
    sample_rate_hz: int,
    channels: int,
    duration_seconds: float = 1.0,
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
        0.25
        * np.sin(
            2.0
            * np.pi
            * 440.0
            * time
        )
    ).astype(
        np.float32
    )

    if channels == 1:
        samples = signal.reshape(
            1,
            -1,
        )
        layout = "mono"
    elif channels == 2:
        samples = np.stack(
            [
                signal,
                signal * 0.5,
            ],
            axis=0,
        )
        layout = "stereo"
    else:
        raise ValueError(
            "test helper supports only "
            "mono or stereo"
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

        stream.layout = layout

        frame = av.AudioFrame.from_ndarray(
            samples,
            format="fltp",
            layout=layout,
        )

        frame.sample_rate = sample_rate_hz

        for packet in stream.encode(
            frame
        ):
            container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    return output.getvalue()


def test_version_is_frozen():
    assert (
        AUDIO_PREPROCESSING_VERSION
        == "audio-preprocess-v1"
    )


def test_canonical_rate_is_16khz():
    assert (
        CANONICAL_SAMPLE_RATE_HZ
        == 16_000
    )


def test_mono_16khz_remains_canonical():
    audio_bytes = make_wav_bytes(
        sample_rate_hz=16_000,
        channels=1,
    )

    result = preprocess_audio(
        audio_bytes
    )

    assert result.sample_rate_hz == 16_000
    assert result.original_sample_rate_hz == 16_000
    assert result.original_channels == 1

    assert result.waveform.dtype == np.float32
    assert result.waveform.ndim == 1

    assert result.sample_count == 16_000

    assert result.duration_seconds == pytest.approx(
        1.0,
        abs=1e-3,
    )

    assert np.all(
        np.isfinite(result.waveform)
    )


def test_stereo_is_downmixed_to_mono():
    audio_bytes = make_wav_bytes(
        sample_rate_hz=16_000,
        channels=2,
    )

    result = preprocess_audio(
        audio_bytes
    )

    assert result.original_channels == 2
    assert result.waveform.ndim == 1
    assert result.sample_count == 16_000


def test_48khz_is_resampled_to_16khz():
    audio_bytes = make_wav_bytes(
        sample_rate_hz=48_000,
        channels=1,
    )

    result = preprocess_audio(
        audio_bytes
    )

    assert (
        result.original_sample_rate_hz
        == 48_000
    )

    assert result.sample_rate_hz == 16_000

    assert result.sample_count == pytest.approx(
        16_000,
        abs=2,
    )

    assert result.duration_seconds == pytest.approx(
        1.0,
        abs=1e-3,
    )


def test_empty_bytes_are_rejected():
    with pytest.raises(
        AudioPreprocessingError,
        match="empty",
    ):
        preprocess_audio(b"")


def test_corrupt_audio_is_rejected():
    with pytest.raises(
        AudioPreprocessingError,
        match="decoded",
    ):
        preprocess_audio(
            b"this is not an audio file"
        )
        
def test_preprocessing_is_deterministic():
    audio_bytes = make_wav_bytes(
        sample_rate_hz=48_000,
        channels=2,
    )

    first = preprocess_audio(
        audio_bytes
    )

    second = preprocess_audio(
        audio_bytes
    )

    assert (
        first.sample_rate_hz
        == second.sample_rate_hz
    )

    assert (
        first.sample_count
        == second.sample_count
    )

    assert (
        first.duration_seconds
        == second.duration_seconds
    )

    np.testing.assert_array_equal(
        first.waveform,
        second.waveform,
    )