from app.models.feature_constants import (
    FEATURE_STATUS_COMPLETED,
    FEATURE_STATUS_FAILED,
    FEATURE_STATUS_PENDING,
    FEATURE_STATUS_PROCESSING,
)
from app.models.journal import JournalEntry
from app.models.journal_feature_set import (
    JournalFeatureSet,
)
from app.models.user import User
from app.services.features.feature_sets import (
    get_feature_set,
    mark_feature_set_completed,
    mark_feature_set_failed,
    mark_feature_set_processing,
    resolve_feature_set,
)


def make_journal(db_session):
    user = User(
        email="feature-service@example.com",
        password_hash="test-password-hash",
        display_name="Feature Service User",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="TEXT",
        raw_text="Feature service test.",
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.commit()
    db_session.refresh(journal)

    return journal


def test_resolve_creates_pending_feature_set(
    db_session,
):
    journal = make_journal(db_session)

    result = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash="a" * 64,
    )

    assert result.created is True
    assert result.should_process is True
    assert (
        result.feature_set.status
        == FEATURE_STATUS_PENDING
    )

    db_session.commit()

    stored = get_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash="a" * 64,
    )

    assert stored is not None


def test_completed_generation_is_reused(
    db_session,
):
    journal = make_journal(db_session)

    feature_set = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash="b" * 64,
        status=FEATURE_STATUS_COMPLETED,
    )

    db_session.add(feature_set)
    db_session.commit()

    result = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash="b" * 64,
    )

    assert result.created is False
    assert result.should_process is False
    assert result.feature_set.id == feature_set.id


def test_failed_generation_becomes_retryable(
    db_session,
):
    journal = make_journal(db_session)

    feature_set = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash="c" * 64,
        status=FEATURE_STATUS_FAILED,
        error_message="previous failure",
    )

    db_session.add(feature_set)
    db_session.commit()

    result = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash="c" * 64,
    )

    assert result.created is False
    assert result.should_process is True
    assert (
        result.feature_set.status
        == FEATURE_STATUS_PENDING
    )
    assert result.feature_set.error_message is None


def test_pending_generation_is_not_duplicated(
    db_session,
):
    journal = make_journal(db_session)

    first = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash="d" * 64,
    )

    db_session.commit()

    second = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash="d" * 64,
    )

    assert second.created is False
    assert second.should_process is False
    assert (
        second.feature_set.id
        == first.feature_set.id
    )


def test_processing_generation_is_not_duplicated(
    db_session,
):
    journal = make_journal(db_session)

    feature_set = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash="e" * 64,
        status=FEATURE_STATUS_PROCESSING,
    )

    db_session.add(feature_set)
    db_session.commit()

    result = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash="e" * 64,
    )

    assert result.created is False
    assert result.should_process is False


def test_lifecycle_helpers(
    db_session,
):
    journal = make_journal(db_session)

    result = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash="f" * 64,
    )

    feature_set = result.feature_set

    mark_feature_set_processing(
        feature_set
    )

    assert (
        feature_set.status
        == FEATURE_STATUS_PROCESSING
    )

    mark_feature_set_failed(
        feature_set,
        error_message="feature extraction failed",
    )

    assert (
        feature_set.status
        == FEATURE_STATUS_FAILED
    )
    assert (
        feature_set.error_message
        == "feature extraction failed"
    )
    assert feature_set.completed_at is None

    mark_feature_set_processing(
        feature_set
    )

    mark_feature_set_completed(
        feature_set
    )

    assert (
        feature_set.status
        == FEATURE_STATUS_COMPLETED
    )
    assert feature_set.error_message is None
    assert feature_set.completed_at is not None