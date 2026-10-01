import numpy as np

from app.models.feature_constants import (
    FEATURE_QUALITY_DEGRADED,
    FEATURE_QUALITY_GOOD,
)
from app.services.context.input_contract import (
    EmbeddingVector,
    FeatureProvenance,
)


M3_EMBEDDING_DIMENSION = 768

ACCEPTED_FEATURE_QUALITIES = frozenset(
    {
        FEATURE_QUALITY_GOOD,
        FEATURE_QUALITY_DEGRADED,
    }
)


class ContextInputValidationError(ValueError):
    """
    Raised when persisted M3 evidence violates the structural
    contract required by M4.

    Error messages must describe the violated contract without
    exposing raw journal content or embedding values.
    """


def validate_feature_quality(
    *,
    quality: str,
    modality: str,
) -> None:
    if quality not in ACCEPTED_FEATURE_QUALITIES:
        raise ContextInputValidationError(
            f"{modality} feature quality is not usable by M4"
        )


def validate_embedding(
    *,
    embedding: EmbeddingVector,
    declared_dimension: int | None,
    modality: str,
) -> EmbeddingVector:
    """
    Validate and normalize one learned M3 embedding.

    M4 requires:
        - declared dimension = 768
        - actual shape = (768,)
        - all values finite
        - float32 model-boundary representation

    A defensive copy is returned.
    """

    if declared_dimension != M3_EMBEDDING_DIMENSION:
        raise ContextInputValidationError(
            f"{modality} embedding declared dimension "
            f"must be {M3_EMBEDDING_DIMENSION}"
        )

    normalized = np.asarray(
        embedding,
        dtype=np.float32,
    )

    if normalized.shape != (
        M3_EMBEDDING_DIMENSION,
    ):
        raise ContextInputValidationError(
            f"{modality} embedding shape "
            f"must be ({M3_EMBEDDING_DIMENSION},)"
        )

    if not np.all(np.isfinite(normalized)):
        raise ContextInputValidationError(
            f"{modality} embedding contains "
            "non-finite values"
        )

    return normalized.copy()


def validate_provenance(
    *,
    provenance: FeatureProvenance,
    modality: str,
) -> None:
    if not provenance.encoder_name.strip():
        raise ContextInputValidationError(
            f"{modality} encoder name is unavailable"
        )

    if not provenance.encoder_version.strip():
        raise ContextInputValidationError(
            f"{modality} encoder version is unavailable"
        )