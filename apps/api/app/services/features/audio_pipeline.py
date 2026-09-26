from dataclasses import dataclass

import numpy as np

from app.services.features.audio_preprocessing import (
    AUDIO_PREPROCESSING_VERSION,
    AudioPreprocessingResult,
    preprocess_audio,
)
from app.services.features.audio_quality import (
    AUDIO_QUALITY_VERSION,
    AudioQualityMeasurements,
    measure_audio_quality,
)
from app.services.features.audio_quality_policy import (
    AUDIO_QUALITY_POLICY_VERSION,
    AudioQualityAssessment,
    assess_audio_quality,
)


AUDIO_PIPELINE_VERSION = "audio-pipeline-v1"


@dataclass(frozen=True)
class AudioPipelineResult:
    waveform: np.ndarray

    sample_rate_hz: int
    sample_count: int
    duration_seconds: float

    original_sample_rate_hz: int
    original_channels: int

    measurements: AudioQualityMeasurements
    assessment: AudioQualityAssessment

    pipeline_version: str = AUDIO_PIPELINE_VERSION
    preprocessing_version: str = (
        AUDIO_PREPROCESSING_VERSION
    )
    quality_version: str = (
        AUDIO_QUALITY_VERSION
    )
    quality_policy_version: str = (
        AUDIO_QUALITY_POLICY_VERSION
    )

    @property
    def quality_status(self) -> str:
        return self.assessment.status

    @property
    def quality_reasons(
        self,
    ) -> tuple[str, ...]:
        return self.assessment.reasons


def process_audio(
    audio_bytes: bytes,
) -> AudioPipelineResult:
    preprocessing: AudioPreprocessingResult = (
        preprocess_audio(
            audio_bytes
        )
    )

    measurements = measure_audio_quality(
        preprocessing.waveform
    )

    assessment = assess_audio_quality(
        duration_seconds=(
            preprocessing.duration_seconds
        ),
        measurements=measurements,
    )

    return AudioPipelineResult(
        waveform=preprocessing.waveform,
        sample_rate_hz=(
            preprocessing.sample_rate_hz
        ),
        sample_count=(
            preprocessing.sample_count
        ),
        duration_seconds=(
            preprocessing.duration_seconds
        ),
        original_sample_rate_hz=(
            preprocessing.original_sample_rate_hz
        ),
        original_channels=(
            preprocessing.original_channels
        ),
        measurements=measurements,
        assessment=assessment,
    )