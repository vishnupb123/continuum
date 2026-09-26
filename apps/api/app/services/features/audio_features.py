from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models.audio_feature import AudioFeature
from app.models.journal import JournalEntry
from app.models.journal_feature_set import JournalFeatureSet
from app.services.features.acoustic_features import (
    AcousticFeatureResult,
    extract_acoustic_features,
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
    acoustic_features: AcousticFeatureResult


def extract_audio_features(
    db: Session,
    *,
    journal: JournalEntry,
    feature_set: JournalFeatureSet,
) -> AudioFeatureExtractionResult:
    """
    Extract and persist deterministic audio features for a VOICE
    journal.

    Pipeline:

        private encoded audio
                |
                v
        M3.4 canonical audio pipeline
                |
                +--> engineering quality measurements
                |
                v
        canonical 16 kHz waveform
                |
                v
        M3.5 deterministic acoustic features
                |
                v
        AudioFeature persistence

    Learned neural audio embeddings are intentionally excluded.
    They belong to M3.6.
    """

    if feature_set.journal_id != journal.id:
        raise ValueError(
            "feature set does not belong to journal"
        )

    if journal.entry_type != "VOICE":
        raise ValueError(
            "audio features require a VOICE journal"
        )

    if journal.audio is None:
        raise ValueError(
            "voice journal has no audio"
        )

    if feature_set.audio_feature is not None:
        raise ValueError(
            "feature set already has audio features"
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

    acoustic_features = (
        extract_acoustic_features(
            pipeline_result.waveform,
            sample_rate_hz=(
                pipeline_result.sample_rate_hz
            ),
        )
    )

    audio_feature = AudioFeature(
        preprocessing_version=(
            pipeline_result.preprocessing_version
        ),

        # Reserved for M3.6 learned embeddings.
        encoder_name=None,
        encoder_version=None,
        embedding_dimension=None,

        duration_seconds=(
            pipeline_result.duration_seconds
        ),

        # Remains NULL until we implement an actual VAD.
        # M3.5 signal_activity_ratio is explicitly NOT a
        # speech ratio.
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
                pipeline_result.measurements.peak_amplitude
            ),
            "rms_amplitude": (
                pipeline_result.measurements.rms_amplitude
            ),
            "silence_ratio": (
                pipeline_result.measurements.silence_ratio
            ),
            "clipping_ratio": (
                pipeline_result.measurements.clipping_ratio
            ),

            # M3.5 deterministic acoustic representation.
            "acoustic_features": {
                "feature_version": (
                    acoustic_features.feature_version
                ),
                "frame_count": (
                    acoustic_features.frame_count
                ),
                "signal_activity_ratio": (
                    acoustic_features.signal_activity_ratio
                ),
                "signal_inactivity_ratio": (
                    acoustic_features.signal_inactivity_ratio
                ),
                "rms_mean": (
                    acoustic_features.rms_mean
                ),
                "rms_std": (
                    acoustic_features.rms_std
                ),
                "spectral_centroid_mean_hz": (
                    acoustic_features
                    .spectral_centroid_mean_hz
                ),
                "spectral_centroid_std_hz": (
                    acoustic_features
                    .spectral_centroid_std_hz
                ),
                "spectral_bandwidth_mean_hz": (
                    acoustic_features
                    .spectral_bandwidth_mean_hz
                ),
                "spectral_bandwidth_std_hz": (
                    acoustic_features
                    .spectral_bandwidth_std_hz
                ),
                "spectral_rolloff_mean_hz": (
                    acoustic_features
                    .spectral_rolloff_mean_hz
                ),
                "spectral_rolloff_std_hz": (
                    acoustic_features
                    .spectral_rolloff_std_hz
                ),
                "zero_crossing_rate_mean": (
                    acoustic_features
                    .zero_crossing_rate_mean
                ),
                "zero_crossing_rate_std": (
                    acoustic_features
                    .zero_crossing_rate_std
                ),
                "mfcc_mean": list(
                    acoustic_features.mfcc_mean
                ),
                "mfcc_std": list(
                    acoustic_features.mfcc_std
                ),
            },
        },
    )

    feature_set.audio_feature = (
        audio_feature
    )

    db.flush()

    complete_feature_set_if_ready(
        feature_set,
        entry_type=journal.entry_type,
    )

    db.flush()

    return AudioFeatureExtractionResult(
        feature_set=feature_set,
        audio_feature=audio_feature,
        pipeline_result=pipeline_result,
        acoustic_features=acoustic_features,
    )