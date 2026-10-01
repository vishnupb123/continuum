from dataclasses import dataclass
from uuid import UUID

import numpy as np
from numpy.typing import NDArray


EmbeddingVector = NDArray[np.float32]


@dataclass(frozen=True)
class FeatureProvenance:
    """
    Provenance for one M3 learned feature artifact.

    This describes which encoder produced the embedding consumed
    by M4. It is evidence lineage, not M4 model provenance.
    """

    encoder_name: str
    encoder_version: str
    encoder_revision: str | None


@dataclass(frozen=True)
class ContextModelInput:
    """
    Validated M4 input for exactly one current journal observation.

    This object contains no historical memory, baseline, trend,
    longitudinal state, or raw journal/audio content.

    TEXT:
        - text_embedding is required
        - audio_embedding is None

    VOICE:
        - text_embedding is required
        - audio_embedding is required
    """

    journal_id: UUID
    feature_set_id: UUID

    entry_type: str

    feature_pipeline_version: str
    source_hash: str

    text_embedding: EmbeddingVector
    text_quality: str
    text_provenance: FeatureProvenance

    audio_embedding: EmbeddingVector | None
    audio_quality: str | None
    audio_provenance: FeatureProvenance | None