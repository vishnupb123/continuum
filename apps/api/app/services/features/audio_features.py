from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models.audio_feature import AudioFeature
# from app.models.feature_constants import (
#     FEATURE_QUALITY_UNUSABLE,
# )
from app.models.journal import JournalEntry
from app.models.journal_feature_set import (
    JournalFeatureSet,
)
from app.services.features.audio_pipeline import (
    AudioPipelineResult,
    process_audio,
)
from app.services.features.feature_sets import (
    complete_feature_set_if_ready,
    mark_feature_set_processing,
)
from app.services.storage.factory import (
    get_audio_storage,
)


@dataclass(frozen=True)
class AudioFeatureExtractionResult:
    feature_set: JournalFeatureSet
    audio_feature: AudioFeature
    pipeline_result: AudioPipelineResult


def extract_audio_features(
    db: Session,
    *,
    journal: JournalEntry,
    feature_set: JournalFeatureSet,
) -> AudioFeatureExtractionResult:
    if feature_set.journal_id != journal.id:
        raise ValueError(
            "feature set does not belong "
            "to journal"
        )

    if journal.entry_type != "VOICE":
        raise ValueError(
            "audio features require "
            "a VOICE journal"
        )

    if journal.audio is None:
        raise ValueError(
            "voice journal has no audio"
        )

    storage = get_audio_storage()

    audio_bytes = storage.get(
        journal.audio.storage_key
    )

    mark_feature_set_processing(
        feature_set
    )

    pipeline_result = process_audio(
        audio_bytes
    )

    if feature_set.audio_feature is not None:
        raise ValueError(
            "feature set already has "
            "audio features"
        )

    audio_feature = AudioFeature(
        preprocessing_version=(
            pipeline_result.preprocessing_version
        ),
        encoder_name=None,
        encoder_version=None,
        embedding_dimension=None,
        duration_seconds=(
            pipeline_result.duration_seconds
        ),

        # We have NOT performed speech/VAD
        # detection. Silence ratio is not
        # equivalent to speech ratio.
        speech_ratio=None,

        sample_rate_hz=(
            pipeline_result.sample_rate_hz
        ),
        quality_status=(
            pipeline_result.quality_status
        ),
        feature_metadata={
            "audio_pipeline_version": (
                pipeline_result.pipeline_version
            ),
            "audio_quality_version": (
                pipeline_result.quality_version
            ),
            "audio_quality_policy_version": (
                pipeline_result.quality_policy_version
            ),
            "quality_reasons": list(
                pipeline_result.quality_reasons
            ),
            "sample_count": (
                pipeline_result.sample_count
            ),
            "original_sample_rate_hz": (
                pipeline_result.original_sample_rate_hz
            ),
            "original_channels": (
                pipeline_result.original_channels
            ),
            "peak_amplitude": (
                pipeline_result
                .measurements
                .peak_amplitude
            ),
            "rms_amplitude": (
                pipeline_result
                .measurements
                .rms_amplitude
            ),
            "silence_ratio": (
                pipeline_result
                .measurements
                .silence_ratio
            ),
            "clipping_ratio": (
                pipeline_result
                .measurements
                .clipping_ratio
            ),
        },
    )

    feature_set.audio_feature = (
        audio_feature
    )

    db.flush()

    # Important:
    #
    # For VOICE journals the generation is not
    # complete merely because audio preprocessing
    # succeeded. It must also contain the text
    # representation produced from the transcript.
    #
    # Therefore only complete the generation if
    # its TextFeature already exists.
    complete_feature_set_if_ready(
    feature_set,
    entry_type=journal.entry_type,
)

    db.flush()

    return AudioFeatureExtractionResult(
        feature_set=feature_set,
        audio_feature=audio_feature,
        pipeline_result=pipeline_result,
    )