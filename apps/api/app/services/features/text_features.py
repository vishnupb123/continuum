from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models.feature_constants import (
    FEATURE_QUALITY_UNUSABLE,
)
from app.models.journal import JournalEntry
from app.models.journal_feature_set import (
    JournalFeatureSet,
)
from app.models.text_feature import TextFeature
from app.services.features.feature_sets import (
    complete_feature_set_if_ready,
    mark_feature_set_processing,
)
from app.services.features.fingerprinting import (
    FINGERPRINT_VERSION,
)
from app.services.features.text_encoder.chunking import (
    TEXT_CHUNKING_VERSION,
)
from app.services.features.text_encoder.factory import (
    get_text_encoder,
)
from app.services.features.text_preprocessing import (
    preprocess_text,
)


@dataclass(frozen=True)
class TextFeatureExtractionResult:
    feature_set: JournalFeatureSet
    text_feature: TextFeature


def extract_text_features(
    db: Session,
    *,
    journal: JournalEntry,
    feature_set: JournalFeatureSet,
) -> TextFeatureExtractionResult:
    """
    Preprocess, encode, and persist the text representation
    for an already-resolved feature generation.

    This function intentionally does not commit.
    """

    if journal.id != feature_set.journal_id:
        raise ValueError(
            "Feature set does not belong to journal"
        )

    if journal.entry_type == "TEXT":
        source_type = "RAW_TEXT"
    elif journal.entry_type == "VOICE":
        source_type = "TRANSCRIPT"
    else:
        raise ValueError(
            "Unsupported journal entry type: "
            f"{journal.entry_type}"
        )

    if journal.raw_text is None:
        raise ValueError(
            "Journal has no text available "
            "for feature extraction"
        )

    mark_feature_set_processing(
        feature_set
    )

    preprocessed = preprocess_text(
        journal.raw_text
    )

    if (
        preprocessed.quality_status
        == FEATURE_QUALITY_UNUSABLE
    ):
        raise ValueError(
            "Journal text is unusable after "
            "preprocessing"
        )

    encoder = get_text_encoder()

    encoded = encoder.encode(
        preprocessed.text
    )

    text_feature = TextFeature(
        source_type=source_type,
        preprocessing_version=(
            preprocessed.preprocessing_version
        ),
        encoder_name=encoded.encoder_name,
        encoder_version=encoded.encoder_version,
        encoder_revision=encoded.encoder_revision,
        embedding_dimension=encoded.dimension,
        embedding=list(encoded.embedding),
        word_count=preprocessed.word_count,
        character_count=(
            preprocessed.character_count
        ),
        quality_status=(
            preprocessed.quality_status
        ),
        feature_metadata={
            "quality_reasons": list(
                preprocessed.quality_reasons
            ),
            "fingerprint_version": (
                FINGERPRINT_VERSION
            ),
            "chunking_version": (
                TEXT_CHUNKING_VERSION
            ),
            "embedding_normalized": (
                encoded.normalized
            ),
            "similarity_metric": "cosine",
        },
    )

    feature_set.text_feature = text_feature

    db.flush()

    complete_feature_set_if_ready(
      feature_set,
      entry_type=journal.entry_type,
    )

    db.flush()

    return TextFeatureExtractionResult(
        feature_set=feature_set,
        text_feature=text_feature,
    )