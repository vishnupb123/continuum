import math
from unittest.mock import patch

import pytest

from app.models.feature_constants import (
    FEATURE_STATUS_COMPLETED,
    FEATURE_STATUS_PENDING,
)
from app.models.journal import JournalEntry
from app.models.journal_feature_set import (
    JournalFeatureSet,
)
from app.models.user import User
from app.services.features.text_encoder.base import (
    TextEncodingResult,
)
from app.services.features.text_features import (
    extract_text_features,
)

from app.services.features.feature_sets import (
    get_current_completed_feature_set,
)
from app.services.features.fingerprinting import (
    calculate_journal_source_hash,
)


def make_journal(
    db_session,
    *,
    entry_type="TEXT",
    text="Today was a productive day.",
):
    user = User(
        email=(
            f"text-extraction-{entry_type.lower()}"
            "@example.com"
        ),
        password_hash="test-password-hash",
        display_name="Text Extraction User",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type=entry_type,
        raw_text=text,
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.commit()
    db_session.refresh(journal)

    return journal


def make_feature_set(
    db_session,
    journal,
):
    feature_set = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash="a" * 64,
        status=FEATURE_STATUS_PENDING,
    )

    db_session.add(feature_set)
    db_session.flush()

    return feature_set


def make_embedding():
    vector = [0.0] * 768
    vector[0] = 1.0
    return tuple(vector)


def make_encoding_result():
    return TextEncodingResult(
        embedding=make_embedding(),
        dimension=768,
        encoder_name=(
            "sentence-transformers/"
            "all-mpnet-base-v2"
        ),
        encoder_version="text-encoder-v1",
        encoder_revision=(
            "e8c3b32edf5434bc2275fc9bab85f82640a19130"
        ),
        normalized=True,
    )


@patch(
    "app.services.features.text_features."
    "get_text_encoder"
)
def test_text_journal_is_preprocessed_encoded_and_persisted(
    get_text_encoder_mock,
    db_session,
):
    journal = make_journal(
        db_session,
        text=(
            "  Today\twas productive.\n\n\n"
            "I finished my work.  "
        ),
    )

    feature_set = make_feature_set(
        db_session,
        journal,
    )

    encoder = get_text_encoder_mock.return_value
    encoder.encode.return_value = (
        make_encoding_result()
    )

    result = extract_text_features(
        db_session,
        journal=journal,
        feature_set=feature_set,
    )

    db_session.commit()
    db_session.expire_all()

    stored = db_session.get(
        JournalFeatureSet,
        feature_set.id,
    )

    assert stored is not None
    assert (
        stored.status
        == FEATURE_STATUS_COMPLETED
    )

    assert stored.text_feature is not None

    text_feature = stored.text_feature

    assert (
        text_feature.source_type
        == "RAW_TEXT"
    )

    assert (
        text_feature.preprocessing_version
        == "text-preprocess-v1"
    )

    assert (
        text_feature.encoder_version
        == "text-encoder-v1"
    )

    assert (
        text_feature.encoder_revision
        == (
            "e8c3b32edf5434bc2275fc9bab85f82640a19130"
        )
    )

    assert (
        text_feature.embedding_dimension
        == 768
    )

    assert len(text_feature.embedding) == 768

    assert text_feature.feature_metadata[
        "chunking_version"
    ] == "text-chunking-v1"

    assert text_feature.feature_metadata[
        "fingerprint_version"
    ] == "source-fingerprint-v1"

    assert text_feature.feature_metadata[
        "embedding_normalized"
    ] is True

    assert text_feature.feature_metadata[
        "similarity_metric"
    ] == "cosine"

    encoder.encode.assert_called_once_with(
        "Today was productive.\n\n"
        "I finished my work."
    )

    norm = math.sqrt(
        sum(
            float(value) ** 2
            for value in text_feature.embedding
        )
    )

    assert norm == pytest.approx(
        1.0,
        abs=1e-6,
    )

    assert result.feature_set.id == stored.id


@patch(
    "app.services.features.text_features."
    "get_text_encoder"
)
def test_text_generation_is_available_to_m4_handoff(
    get_text_encoder_mock,
    db_session,
):
    """
    M3.10 TEXT acceptance:

    A completed TEXT journal is converted into a canonical
    M3 generation, persisted, completed, and then retrieved
    through the authoritative M3 -> M4 handoff.
    """

    journal = make_journal(
        db_session,
        text=(
            "Today was productive and focused. "
            "I completed the work I had planned "
            "and felt positive about my progress."
        ),
    )

    # ---------------------------------------------------------
    # Canonical generation identity
    # ---------------------------------------------------------

    source_hash = (
        calculate_journal_source_hash(
            journal
        )
    )

    feature_set = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash=source_hash,
        status=FEATURE_STATUS_PENDING,
    )

    db_session.add(feature_set)
    db_session.flush()

    feature_set_id = feature_set.id

    # ---------------------------------------------------------
    # Deterministic encoder boundary
    # ---------------------------------------------------------

    encoder = (
        get_text_encoder_mock.return_value
    )

    encoder.encode.return_value = (
        make_encoding_result()
    )

    # ---------------------------------------------------------
    # Real text feature pipeline
    # ---------------------------------------------------------

    result = extract_text_features(
        db_session,
        journal=journal,
        feature_set=feature_set,
    )

    assert (
        result.feature_set.id
        == feature_set_id
    )

    assert (
        feature_set.status
        == FEATURE_STATUS_COMPLETED
    )

    assert (
        feature_set.text_feature
        is not None
    )

    assert (
        feature_set.text_feature.embedding
        is not None
    )

    assert (
        feature_set.text_feature.embedding_dimension
        == 768
    )

    # M3 must not mutate the completed M2 lifecycle.
    assert (
        journal.status
        == "COMPLETED"
    )

    # ---------------------------------------------------------
    # Persist the completed generation
    # ---------------------------------------------------------

    db_session.commit()
    db_session.expire_all()

    # ---------------------------------------------------------
    # Authoritative M3 -> M4 handoff
    # ---------------------------------------------------------

    current_feature_set = (
        get_current_completed_feature_set(
            db_session,
            journal=journal,
        )
    )

    assert (
        current_feature_set
        is not None
    )

    assert (
        current_feature_set.id
        == feature_set_id
    )

    assert (
        current_feature_set.status
        == FEATURE_STATUS_COMPLETED
    )

    assert (
        current_feature_set.source_hash
        == source_hash
    )

    assert (
        current_feature_set.pipeline_version
        == "m3-v1"
    )

    # ---------------------------------------------------------
    # M4-consumable text representation
    # ---------------------------------------------------------

    text_feature = (
        current_feature_set.text_feature
    )

    assert text_feature is not None

    assert (
        text_feature.source_type
        == "RAW_TEXT"
    )

    assert (
        text_feature.embedding
        is not None
    )

    assert (
        text_feature.embedding_dimension
        == 768
    )

    assert (
        text_feature.preprocessing_version
        == "text-preprocess-v1"
    )

    assert (
        text_feature.encoder_version
        == "text-encoder-v1"
    )

    assert (
        text_feature.feature_metadata[
            "fingerprint_version"
        ]
        == "source-fingerprint-v1"
    )

    assert (
        text_feature.feature_metadata[
            "embedding_normalized"
        ]
        is True
    )

    assert (
        text_feature.feature_metadata[
            "similarity_metric"
        ]
        == "cosine"
    )

@patch(
    "app.services.features.text_features."
    "get_text_encoder"
)
def test_voice_transcript_uses_transcript_source_type(
    get_text_encoder_mock,
    db_session,
):
    journal = make_journal(
        db_session,
        entry_type="VOICE",
        text="This came from transcription.",
    )

    feature_set = make_feature_set(
        db_session,
        journal,
    )

    encoder = get_text_encoder_mock.return_value
    encoder.encode.return_value = (
        make_encoding_result()
    )

    extract_text_features(
        db_session,
        journal=journal,
        feature_set=feature_set,
    )

    assert (
        feature_set.text_feature.source_type
        == "TRANSCRIPT"
    )


@patch(
    "app.services.features.text_features."
    "get_text_encoder"
)
def test_degraded_text_is_still_encoded(
    get_text_encoder_mock,
    db_session,
):
    journal = make_journal(
        db_session,
        text="Exhausted.",
    )

    feature_set = make_feature_set(
        db_session,
        journal,
    )

    encoder = get_text_encoder_mock.return_value
    encoder.encode.return_value = (
        make_encoding_result()
    )

    extract_text_features(
        db_session,
        journal=journal,
        feature_set=feature_set,
    )

    assert (
        feature_set.text_feature.quality_status
        == "DEGRADED"
    )

    assert (
        feature_set.text_feature.feature_metadata[
            "quality_reasons"
        ]
        == ["VERY_SHORT_TEXT"]
    )

    encoder.encode.assert_called_once_with(
        "Exhausted."
    )


@patch(
    "app.services.features.text_features."
    "get_text_encoder"
)
def test_unusable_text_is_not_encoded(
    get_text_encoder_mock,
    db_session,
):
    journal = make_journal(
        db_session,
        text="\t\n\r   ",
    )

    feature_set = make_feature_set(
        db_session,
        journal,
    )

    with pytest.raises(
        ValueError,
        match="unusable",
    ):
        extract_text_features(
            db_session,
            journal=journal,
            feature_set=feature_set,
        )

    get_text_encoder_mock.assert_not_called()


def test_feature_set_must_belong_to_journal(
    db_session,
):
    first = make_journal(
        db_session,
        text="First journal.",
    )

    # Avoid duplicate test-user email.
    user = User(
        email="second-journal@example.com",
        password_hash="test-password-hash",
        display_name="Second User",
    )

    db_session.add(user)
    db_session.flush()

    second = JournalEntry(
        user_id=user.id,
        entry_type="TEXT",
        raw_text="Second journal.",
        status="COMPLETED",
    )

    db_session.add(second)
    db_session.commit()

    feature_set = make_feature_set(
        db_session,
        first,
    )

    with pytest.raises(
        ValueError,
        match="does not belong",
    ):
        extract_text_features(
            db_session,
            journal=second,
            feature_set=feature_set,
        )