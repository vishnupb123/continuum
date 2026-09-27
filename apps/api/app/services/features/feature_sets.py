from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.feature_constants import (
    FEATURE_STATUS_COMPLETED,
    FEATURE_STATUS_FAILED,
    FEATURE_STATUS_PENDING,
    FEATURE_STATUS_PROCESSING,
)
from app.models.journal_feature_set import (
    JournalFeatureSet,
)
from app.services.features.constants import (
    FEATURE_PIPELINE_VERSION,
)


@dataclass(frozen=True)
class FeatureSetResolution:
    feature_set: JournalFeatureSet
    created: bool
    should_process: bool


def get_feature_set(
    db: Session,
    *,
    journal_id: UUID,
    source_hash: str,
    pipeline_version: str = FEATURE_PIPELINE_VERSION,
) -> JournalFeatureSet | None:
    statement = select(
        JournalFeatureSet
    ).where(
        JournalFeatureSet.journal_id
        == journal_id,
        JournalFeatureSet.pipeline_version
        == pipeline_version,
        JournalFeatureSet.source_hash
        == source_hash,
    )

    return db.scalar(statement)


def _insert_feature_set_if_absent(
    db: Session,
    *,
    journal_id: UUID,
    source_hash: str,
    pipeline_version: str,
) -> UUID | None:
    """
    Atomically attempt to create a feature generation.

    PostgreSQL owns concurrency arbitration through the
    uq_journal_feature_generation unique constraint.

    Returns:
        UUID:
            This transaction inserted the generation.

        None:
            Another transaction already created the same
            generation identity.

    This deliberately uses ON CONFLICT DO NOTHING rather
    than relying on IntegrityError + Session.rollback().

    A full rollback here could discard unrelated work in
    the caller's transaction.
    """

    candidate_id = uuid4()

    statement = (
        insert(
            JournalFeatureSet
        )
        .values(
            id=candidate_id,
            journal_id=journal_id,
            pipeline_version=pipeline_version,
            source_hash=source_hash,
            status=FEATURE_STATUS_PENDING,
        )
        .on_conflict_do_nothing(
            constraint=(
                "uq_journal_feature_generation"
            )
        )
        .returning(
            JournalFeatureSet.id
        )
    )

    inserted_id = db.scalar(
        statement
    )

    return inserted_id


def _resolution_for_existing(
    db: Session,
    *,
    feature_set: JournalFeatureSet,
) -> FeatureSetResolution:
    """
    Apply lifecycle semantics to an already-existing
    generation.

    Generation identity is immutable. Lifecycle state
    remains mutable.
    """

    if (
        feature_set.status
        == FEATURE_STATUS_COMPLETED
    ):
        return FeatureSetResolution(
            feature_set=feature_set,
            created=False,
            should_process=False,
        )

    if (
        feature_set.status
        == FEATURE_STATUS_FAILED
    ):
        feature_set.status = (
            FEATURE_STATUS_PENDING
        )

        feature_set.error_message = None
        feature_set.completed_at = None

        db.flush()

        return FeatureSetResolution(
            feature_set=feature_set,
            created=False,
            should_process=True,
        )

    if feature_set.status in {
        FEATURE_STATUS_PENDING,
        FEATURE_STATUS_PROCESSING,
    }:
        return FeatureSetResolution(
            feature_set=feature_set,
            created=False,
            should_process=False,
        )

    raise ValueError(
        f"Unsupported feature set status: "
        f"{feature_set.status}"
    )


def resolve_feature_set(
    db: Session,
    *,
    journal_id: UUID,
    source_hash: str,
    pipeline_version: str = FEATURE_PIPELINE_VERSION,
) -> FeatureSetResolution:
    """
    Resolve one immutable feature-generation identity:

        (
            journal_id,
            pipeline_version,
            source_hash,
        )

    Concurrency contract:

        Multiple workers resolving the same identity
        converge on exactly one database row.

    Versioning contract:

        source change
            -> new generation

        pipeline change
            -> new generation

        same identity
            -> same generation
    """

    # -----------------------------------------------------
    # FAST PATH
    # -----------------------------------------------------
    #
    # Most calls are retries or duplicate deliveries.
    # Avoid an INSERT attempt when the generation already
    # exists.
    # -----------------------------------------------------

    existing = get_feature_set(
        db,
        journal_id=journal_id,
        source_hash=source_hash,
        pipeline_version=pipeline_version,
    )

    if existing is not None:
        return _resolution_for_existing(
            db,
            feature_set=existing,
        )

    # -----------------------------------------------------
    # ATOMIC CREATION
    # -----------------------------------------------------
    #
    # The initial SELECT is intentionally only an
    # optimization.
    #
    # Correctness comes from PostgreSQL's unique
    # constraint + ON CONFLICT DO NOTHING.
    # -----------------------------------------------------

    inserted_id = (
        _insert_feature_set_if_absent(
            db,
            journal_id=journal_id,
            source_hash=source_hash,
            pipeline_version=pipeline_version,
        )
    )

    if inserted_id is not None:
        feature_set = db.get(
            JournalFeatureSet,
            inserted_id,
        )

        if feature_set is None:
            raise RuntimeError(
                "Inserted feature generation "
                "could not be loaded"
            )

        return FeatureSetResolution(
            feature_set=feature_set,
            created=True,
            should_process=True,
        )

    # -----------------------------------------------------
    # CONCURRENT WINNER
    # -----------------------------------------------------
    #
    # ON CONFLICT DO NOTHING means another transaction
    # owns the row.
    #
    # PostgreSQL waits for the conflicting transaction
    # sufficiently to resolve the uniqueness decision,
    # after which this transaction can load the winner.
    # -----------------------------------------------------

    existing = get_feature_set(
        db,
        journal_id=journal_id,
        source_hash=source_hash,
        pipeline_version=pipeline_version,
    )

    if existing is None:
        raise RuntimeError(
            "Feature generation conflict occurred "
            "but the winning generation could not "
            "be loaded"
        )

    return _resolution_for_existing(
        db,
        feature_set=existing,
    )


def mark_feature_set_processing(
    feature_set: JournalFeatureSet,
) -> None:
    feature_set.status = (
        FEATURE_STATUS_PROCESSING
    )

    feature_set.error_message = None
    feature_set.completed_at = None


def mark_feature_set_completed(
    feature_set: JournalFeatureSet,
) -> None:
    feature_set.status = (
        FEATURE_STATUS_COMPLETED
    )

    feature_set.error_message = None

    feature_set.completed_at = datetime.now(
        timezone.utc
    )


def mark_feature_set_failed(
    feature_set: JournalFeatureSet,
    *,
    error_message: str,
) -> bool:
    """
    Mark an incomplete feature generation as FAILED.

    COMPLETED is terminal for an immutable generation.

    A stale or duplicate worker may report failure after
    another worker has already completed the generation.
    Such a late failure must never downgrade:

        COMPLETED -> FAILED

    Returns:
        True:
            The generation was marked FAILED.

        False:
            The generation was already COMPLETED and was
            therefore left unchanged.
    """

    if (
        feature_set.status
        == FEATURE_STATUS_COMPLETED
    ):
        return False

    feature_set.status = (
        FEATURE_STATUS_FAILED
    )

    feature_set.error_message = (
        error_message
    )

    feature_set.completed_at = None

    return True


def complete_feature_set_if_ready(
    feature_set: JournalFeatureSet,
    *,
    entry_type: str,
) -> bool:
    if entry_type == "TEXT":
        if (
            feature_set.text_feature
            is None
        ):
            return False

        mark_feature_set_completed(
            feature_set
        )

        return True

    if entry_type == "VOICE":
        if (
            feature_set.text_feature
            is None
        ):
            return False

        if (
            feature_set.audio_feature
            is None
        ):
            return False

        mark_feature_set_completed(
            feature_set
        )

        return True

    raise ValueError(
        f"Unsupported journal entry type: "
        f"{entry_type}"
    )
    
def get_current_completed_feature_set(
    db: Session,
    *,
    journal,
    pipeline_version: str = FEATURE_PIPELINE_VERSION,
) -> JournalFeatureSet | None:
    """
    Return the COMPLETED feature generation representing
    the journal's exact current persisted source.

    Historical COMPLETED generations are deliberately not
    used as fallbacks.
    """

    from app.services.features.fingerprinting import (
        calculate_journal_source_hash,
    )

    source_hash = (
        calculate_journal_source_hash(
            journal
        )
    )

    feature_set = get_feature_set(
        db,
        journal_id=journal.id,
        source_hash=source_hash,
        pipeline_version=pipeline_version,
    )

    if feature_set is None:
        return None

    if (
        feature_set.status
        != FEATURE_STATUS_COMPLETED
    ):
        return None

    return feature_set