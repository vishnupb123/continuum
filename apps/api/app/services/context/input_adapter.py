import numpy as np
from sqlalchemy.orm import Session

from app.models.journal import JournalEntry
from app.models.journal_feature_set import JournalFeatureSet
from app.services.context.input_contract import (
    ContextModelInput,
    FeatureProvenance,
)
from app.services.context.input_validation import (
    ContextInputValidationError,
    validate_embedding,
    validate_feature_quality,
    validate_provenance,
)
from app.services.features.feature_sets import (
    get_current_completed_feature_set,
)


SUPPORTED_ENTRY_TYPES = frozenset(
    {
        "TEXT",
        "VOICE",
    }
)


class ContextInputUnavailableError(Exception):
    """
    Raised when the authoritative current M3 generation required
    by M4 is unavailable or incomplete.

    This does not mutate or downgrade M3.
    """


def _build_provenance(
    *,
    encoder_name: str | None,
    encoder_version: str | None,
    encoder_revision: str | None,
    modality: str,
) -> FeatureProvenance:
    if not encoder_name:
        raise ContextInputValidationError(
            f"{modality} encoder name is unavailable"
        )

    if not encoder_version:
        raise ContextInputValidationError(
            f"{modality} encoder version is unavailable"
        )

    provenance = FeatureProvenance(
        encoder_name=encoder_name,
        encoder_version=encoder_version,
        encoder_revision=encoder_revision,
    )

    validate_provenance(
        provenance=provenance,
        modality=modality,
    )

    return provenance


def _require_authoritative_feature_set(
    db: Session,
    *,
    journal: JournalEntry,
) -> JournalFeatureSet:
    feature_set = get_current_completed_feature_set(
        db,
        journal=journal,
    )

    if feature_set is None:
        raise ContextInputUnavailableError(
            "Current completed feature generation is unavailable"
        )

    return feature_set


def build_context_model_input(
    db: Session,
    *,
    journal: JournalEntry,
) -> ContextModelInput:
    """
    Build and validate one M4 observation from the authoritative
    current M3 feature generation.

    This boundary:
        - never calculates M3 features
        - never falls back to historical generations
        - never reads raw journal content for model evidence
        - never fabricates a missing modality
    """

    if journal.entry_type not in SUPPORTED_ENTRY_TYPES:
        raise ContextInputValidationError(
            "Journal entry type is unsupported by M4"
        )

    feature_set = _require_authoritative_feature_set(
        db,
        journal=journal,
    )

    text_feature = feature_set.text_feature

    if text_feature is None:
        raise ContextInputUnavailableError(
            "Current feature generation has no text feature"
        )

    if text_feature.embedding is None:
        raise ContextInputUnavailableError(
            "Current text feature has no learned embedding"
        )

    validate_feature_quality(
        quality=text_feature.quality_status,
        modality="Text",
    )

    text_embedding = validate_embedding(
        embedding=np.asarray(
            text_feature.embedding,
            dtype=np.float32,
        ),
        declared_dimension=(
            text_feature.embedding_dimension
        ),
        modality="Text",
    )

    text_provenance = _build_provenance(
        encoder_name=text_feature.encoder_name,
        encoder_version=text_feature.encoder_version,
        encoder_revision=text_feature.encoder_revision,
        modality="Text",
    )

    audio_embedding = None
    audio_quality = None
    audio_provenance = None

    if journal.entry_type == "VOICE":
        audio_feature = feature_set.audio_feature

        if audio_feature is None:
            raise ContextInputUnavailableError(
                "VOICE feature generation has no audio feature"
            )

        if audio_feature.embedding is None:
            raise ContextInputUnavailableError(
                "Current audio feature has no learned embedding"
            )

        validate_feature_quality(
            quality=audio_feature.quality_status,
            modality="Audio",
        )

        audio_embedding = validate_embedding(
            embedding=np.asarray(
                audio_feature.embedding,
                dtype=np.float32,
            ),
            declared_dimension=(
                audio_feature.embedding_dimension
            ),
            modality="Audio",
        )

        audio_provenance = _build_provenance(
            encoder_name=audio_feature.encoder_name,
            encoder_version=audio_feature.encoder_version,
            encoder_revision=audio_feature.encoder_revision,
            modality="Audio",
        )

        audio_quality = audio_feature.quality_status

    return ContextModelInput(
        journal_id=journal.id,
        feature_set_id=feature_set.id,
        entry_type=journal.entry_type,
        feature_pipeline_version=(
            feature_set.pipeline_version
        ),
        source_hash=feature_set.source_hash,
        text_embedding=text_embedding,
        text_quality=text_feature.quality_status,
        text_provenance=text_provenance,
        audio_embedding=audio_embedding,
        audio_quality=audio_quality,
        audio_provenance=audio_provenance,
    )