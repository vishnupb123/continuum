from unittest.mock import patch

import pytest

from app.models.feature_constants import (
    FEATURE_STATUS_COMPLETED,
    FEATURE_STATUS_FAILED,
)
from app.models.journal import JournalEntry
from app.models.journal_feature_set import (
    JournalFeatureSet,
)
from app.models.user import User
from app.tasks.feature_tasks import (
    extract_journal_text_features,
)
from app.services.features.text_encoder.base import (
    TextEncodingResult,
)


def make_completed_journal(
    db_session,
):
    user = User(
        email="feature-task@example.com",
        password_hash="test-password-hash",
        display_name="Feature Task User",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="TEXT",
        raw_text=(
            "This journal has enough text "
            "for feature extraction."
        ),
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.commit()
    db_session.refresh(journal)

    return journal


@patch(
    "app.tasks.feature_tasks.SessionLocal"
)
@patch(
    "app.tasks.feature_tasks."
    "extract_text_features"
)
def test_feature_failure_does_not_fail_journal(
    extract_mock,
    session_local_mock,
    db_session,
):
    # The Celery task normally creates its own
    # production DB session.
    #
    # During this test, force it to use the pytest
    # database session so the task can see the
    # journal created below.
    session_local_mock.return_value = (
        db_session
    )

    journal = make_completed_journal(
        db_session
    )

    journal_id = journal.id

    extract_mock.side_effect = RuntimeError(
        "forced encoder failure"
    )

    with pytest.raises(
        RuntimeError,
        match="forced encoder failure",
    ):
        extract_journal_text_features.run(
            str(journal_id)
        )

    # Session.close() is called by the task's finally
    # block. SQLAlchemy sessions can be reused after
    # close(), so expire/query again from the fixture
    # session.
    db_session.expire_all()

    journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    assert journal is not None

    # Critical invariant:
    # M3 failure must never corrupt successful
    # journal processing.
    assert journal.status == "COMPLETED"

    feature_sets = (
        db_session.query(
            JournalFeatureSet
        )
        .filter(
            JournalFeatureSet.journal_id
            == journal_id
        )
        .all()
    )

    # Exactly one generation should exist.
    assert len(feature_sets) == 1

    feature_set = feature_sets[0]

    assert (
        feature_set.status
        == FEATURE_STATUS_FAILED
    )

    # Failed extraction must not leave a partial
    # TextFeature behind.
    assert feature_set.text_feature is None

    assert (
        feature_set.error_message
        == (
            "Text feature extraction failed: "
            "RuntimeError"
        )
    )

    extract_mock.assert_called_once()
    
@patch(
    "app.tasks.feature_tasks.SessionLocal"
)
@patch(
    "app.services.features.text_features."
    "get_text_encoder"
)
def test_failed_generation_is_reused_and_completes_on_retry(
    get_text_encoder_mock,
    session_local_mock,
    db_session,
):
    session_local_mock.return_value = (
        db_session
    )

    journal = make_completed_journal(
        db_session
    )

    journal_id = journal.id

    # ---------------------------------------------
    # FIRST ATTEMPT — FORCE ENCODER FAILURE
    # ---------------------------------------------

    encoder = (
        get_text_encoder_mock.return_value
    )

    encoder.encode.side_effect = RuntimeError(
        "forced encoder failure"
    )

    with pytest.raises(
        RuntimeError,
        match="forced encoder failure",
    ):
        extract_journal_text_features.run(
            str(journal_id)
        )

    db_session.expire_all()

    failed_sets = (
        db_session.query(
            JournalFeatureSet
        )
        .filter(
            JournalFeatureSet.journal_id
            == journal_id
        )
        .all()
    )

    assert len(failed_sets) == 1

    failed_feature_set = failed_sets[0]

    assert (
        failed_feature_set.status
        == FEATURE_STATUS_FAILED
    )

    original_feature_set_id = (
        failed_feature_set.id
    )

    assert (
        failed_feature_set.text_feature
        is None
    )

    # ---------------------------------------------
    # SECOND ATTEMPT — ENCODER RECOVERS
    # ---------------------------------------------

    embedding = [0.0] * 768
    embedding[0] = 1.0

    encoder.encode.side_effect = None
    encoder.encode.return_value = (
        TextEncodingResult(
            embedding=tuple(embedding),
            dimension=768,
            encoder_name=(
                "sentence-transformers/"
                "all-mpnet-base-v2"
            ),
            encoder_version=(
                "text-encoder-v1"
            ),
            encoder_revision=(
                "e8c3b32edf5434bc2275fc9bab85f82640a19130"
            ),
            normalized=True,
        )
    )

    result = (
        extract_journal_text_features.run(
            str(journal_id)
        )
    )

    assert result["status"] == "completed"

    db_session.expire_all()

    recovered_sets = (
        db_session.query(
            JournalFeatureSet
        )
        .filter(
            JournalFeatureSet.journal_id
            == journal_id
        )
        .all()
    )

    # Critical idempotency invariant:
    # retry must reuse the failed generation.
    assert len(recovered_sets) == 1

    recovered_feature_set = (
        recovered_sets[0]
    )

    assert (
        recovered_feature_set.id
        == original_feature_set_id
    )

    assert (
        recovered_feature_set.status
        == FEATURE_STATUS_COMPLETED
    )

    assert (
        recovered_feature_set.error_message
        is None
    )

    assert (
        recovered_feature_set.completed_at
        is not None
    )

    assert (
        recovered_feature_set.text_feature
        is not None
    )

    text_feature = (
        recovered_feature_set.text_feature
    )

    assert (
        text_feature.embedding_dimension
        == 768
    )

    assert (
        len(text_feature.embedding)
        == 768
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

    # Journal remains independent from the M3
    # failure/recovery lifecycle.
    recovered_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    assert (
        recovered_journal.status
        == "COMPLETED"
    )