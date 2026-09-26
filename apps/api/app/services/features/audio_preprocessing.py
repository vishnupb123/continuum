from dataclasses import dataclass
from io import BytesIO

import av
import numpy as np


AUDIO_PREPROCESSING_VERSION = (
    "audio-preprocess-v1"
)

CANONICAL_SAMPLE_RATE_HZ = 16_000


class AudioPreprocessingError(ValueError):
    pass


@dataclass(frozen=True)
class AudioPreprocessingResult:
    waveform: np.ndarray
    sample_rate_hz: int
    sample_count: int
    duration_seconds: float
    original_sample_rate_hz: int
    original_channels: int
    preprocessing_version: str = (
        AUDIO_PREPROCESSING_VERSION
    )

    def __post_init__(self) -> None:
        if self.waveform.dtype != np.float32:
            raise ValueError(
                "waveform must be float32"
            )

        if self.waveform.ndim != 1:
            raise ValueError(
                "waveform must be one-dimensional"
            )

        if (
            self.sample_count
            != self.waveform.shape[0]
        ):
            raise ValueError(
                "sample_count does not match "
                "waveform length"
            )

        if self.sample_rate_hz <= 0:
            raise ValueError(
                "sample_rate_hz must be positive"
            )

        if self.original_sample_rate_hz <= 0:
            raise ValueError(
                "original_sample_rate_hz "
                "must be positive"
            )

        if self.original_channels <= 0:
            raise ValueError(
                "original_channels must be positive"
            )

        if not np.all(
            np.isfinite(self.waveform)
        ):
            raise ValueError(
                "waveform contains non-finite "
                "values"
            )


def _frame_to_mono_float32(
    frame: av.AudioFrame,
) -> np.ndarray:
    samples = frame.to_ndarray()

    samples = np.asarray(
        samples,
        dtype=np.float32,
    )

    # After resampling to mono + fltp,
    # PyAV should produce one planar channel:
    #
    #     shape == (1, samples)
    #
    # Keep the checks explicit so an unexpected
    # decoder/layout change cannot silently alter
    # our canonical representation.
    if samples.ndim == 1:
        mono = samples

    elif (
        samples.ndim == 2
        and samples.shape[0] == 1
    ):
        mono = samples[0]

    else:
        raise AudioPreprocessingError(
            "resampled audio has an "
            "unexpected shape"
        )

    mono = np.asarray(
        mono,
        dtype=np.float32,
    )

    if mono.size == 0:
        raise AudioPreprocessingError(
            "resampled audio frame is empty"
        )

    if not np.all(
        np.isfinite(mono)
    ):
        raise AudioPreprocessingError(
            "resampled audio contains "
            "non-finite samples"
        )

    return mono


def _decode_audio(
    audio_bytes: bytes,
) -> tuple[
    np.ndarray,
    int,
    int,
]:
    if not isinstance(
        audio_bytes,
        bytes,
    ):
        raise TypeError(
            "audio_bytes must be bytes"
        )

    if not audio_bytes:
        raise AudioPreprocessingError(
            "audio input is empty"
        )

    try:
        container = av.open(
            BytesIO(audio_bytes),
            mode="r",
        )
    except Exception as exc:
        raise AudioPreprocessingError(
            "audio could not be decoded"
        ) from exc

    try:
        audio_streams = [
            stream
            for stream in container.streams
            if stream.type == "audio"
        ]

        if not audio_streams:
            raise AudioPreprocessingError(
                "input contains no audio stream"
            )

        stream = audio_streams[0]

        original_sample_rate_hz = (
            stream.codec_context.sample_rate
        )

        original_channels = (
            stream.codec_context.channels
        )

        if (
            original_sample_rate_hz is None
            or original_sample_rate_hz <= 0
        ):
            raise AudioPreprocessingError(
                "audio sample rate is invalid"
            )

        if (
            original_channels is None
            or original_channels <= 0
        ):
            raise AudioPreprocessingError(
                "audio channel count is invalid"
            )

        # PyAV/FFmpeg performs the deterministic
        # conversion into our canonical signal
        # representation:
        #
        #     float planar
        #     mono
        #     16 kHz
        #
        # No amplitude normalization, denoising,
        # silence trimming or VAD is performed.
        resampler = av.AudioResampler(
            format="fltp",
            layout="mono",
            rate=CANONICAL_SAMPLE_RATE_HZ,
        )

        frames: list[np.ndarray] = []

        for decoded_frame in container.decode(
            stream
        ):
            resampled_frames = resampler.resample(
                decoded_frame
            )

            for resampled_frame in resampled_frames:
                frames.append(
                    _frame_to_mono_float32(
                        resampled_frame
                    )
                )

        # Flush samples buffered internally by the
        # resampler. This is important for accurate
        # sample counts at the end of the stream.
        flushed_frames = resampler.resample(
            None
        )

        for flushed_frame in flushed_frames:
            frames.append(
                _frame_to_mono_float32(
                    flushed_frame
                )
            )

        if not frames:
            raise AudioPreprocessingError(
                "audio contains no decoded samples"
            )

        waveform = np.concatenate(
            frames
        ).astype(
            np.float32,
            copy=False,
        )

        if waveform.size == 0:
            raise AudioPreprocessingError(
                "audio contains no decoded samples"
            )

        if not np.all(
            np.isfinite(waveform)
        ):
            raise AudioPreprocessingError(
                "canonical waveform contains "
                "non-finite samples"
            )

        return (
            waveform,
            int(original_sample_rate_hz),
            int(original_channels),
        )

    except AudioPreprocessingError:
        raise

    except Exception as exc:
        raise AudioPreprocessingError(
            "audio could not be decoded"
        ) from exc

    finally:
        container.close()


def preprocess_audio(
    audio_bytes: bytes,
) -> AudioPreprocessingResult:
    (
        waveform,
        original_sample_rate_hz,
        original_channels,
    ) = _decode_audio(
        audio_bytes
    )

    sample_count = int(
        waveform.shape[0]
    )

    if sample_count == 0:
        raise AudioPreprocessingError(
            "canonical waveform is empty"
        )

    duration_seconds = (
        sample_count
        / CANONICAL_SAMPLE_RATE_HZ
    )

    return AudioPreprocessingResult(
        waveform=waveform,
        sample_rate_hz=(
            CANONICAL_SAMPLE_RATE_HZ
        ),
        sample_count=sample_count,
        duration_seconds=float(
            duration_seconds
        ),
        original_sample_rate_hz=(
            original_sample_rate_hz
        ),
        original_channels=(
            original_channels
        ),
    )