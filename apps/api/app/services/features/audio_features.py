from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models.audio_feature import AudioFeature
from app.models.journal import JournalEntry
from app.models.journal_feature_set import JournalFeatureSet
from app.services.features.acoustic_features import (
    AcousticFeatureResult,
    extract_acoustic_features,
)
from app.services.features.audio_encoder import (
    AudioEncodingResult,
)
from app.services.features.audio_encoder_factory import (
    get_audio_encoder,
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
    audio_encoding: AudioEncodingResult


def extract_audio_features(
    db: Session,
    *,
    journal: JournalEntry,
    feature_set: JournalFeatureSet,
) -> AudioFeatureExtractionResult:
    """
    Extract and persist deterministic and learned audio
    representations for a VOICE journal.

    Pipeline:

        private encoded audio
                |
                v
        M3.4 canonical audio pipeline
                |
                +--> engineering quality measurements
                |
                v
        canonical mono 16 kHz float32 waveform
                |
                +--> M3.5 deterministic acoustic features
                |
                +--> M3.6 learned WavLM representation
                |
                v
        AudioFeature persistence

    The learned representation is a frozen journal-level embedding.

    No psychological, emotional, diagnostic, or clinical conclusions
    are produced at this layer.
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

    # ---------------------------------------------------------
    # M3.4
    # Encoded private audio -> canonical waveform + quality.
    # ---------------------------------------------------------

    pipeline_result = process_audio(
        audio_bytes
    )

    # ---------------------------------------------------------
    # M3.5
    # Deterministic, interpretable acoustic representation.
    # ---------------------------------------------------------

    acoustic_features = (
        extract_acoustic_features(
            pipeline_result.waveform,
            sample_rate_hz=(
                pipeline_result.sample_rate_hz
            ),
        )
    )

    # ---------------------------------------------------------
    # M3.6
    # Frozen learned audio representation.
    #
    # Both M3.5 and M3.6 consume the exact same canonical
    # waveform produced by M3.4.
    # ---------------------------------------------------------

    audio_encoder = get_audio_encoder()

    audio_encoding = audio_encoder.encode(
        pipeline_result.waveform,
        sample_rate_hz=(
            pipeline_result.sample_rate_hz
        ),
    )

    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    audio_feature = AudioFeature(
        preprocessing_version=(
            pipeline_result.preprocessing_version
        ),

        encoder_name=(
            audio_encoding.encoder_name
        ),
        encoder_version=(
            audio_encoding.encoder_version
        ),
        encoder_revision=(
            audio_encoding.encoder_revision
        ),
        embedding_dimension=(
            audio_encoding.embedding_dimension
        ),
        embedding=(
            audio_encoding.embedding.tolist()
        ),

        duration_seconds=(
            pipeline_result.duration_seconds
        ),

        # Remains NULL until an actual VAD is introduced.
        #
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

            # M3.6 learned representation metadata.
            #
            # The embedding itself belongs in the pgvector column,
            # not inside JSON.
            "audio_embedding": {
                "encoder_name": (
                    audio_encoding.encoder_name
                ),
                "encoder_version": (
                    audio_encoding.encoder_version
                ),
                "encoder_revision": (
                    audio_encoding.encoder_revision
                ),
                "embedding_dimension": (
                    audio_encoding.embedding_dimension
                ),
                "sample_rate_hz": (
                    audio_encoding.sample_rate_hz
                ),
                "pooling_strategy": (
                    audio_encoding.pooling_strategy
                ),
                "normalization": (
                    audio_encoding.normalization
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
        audio_encoding=audio_encoding,
    )