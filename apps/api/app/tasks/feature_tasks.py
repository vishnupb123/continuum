from uuid import UUID

from sqlalchemy.exc import IntegrityError

from app.db.session import SessionLocal
from app.models.feature_constants import (
    FEATURE_STATUS_COMPLETED,
)
from app.models.journal import JournalEntry
from app.models.journal_feature_set import (
    JournalFeatureSet,
)
from app.services.features.audio_features import (
    extract_audio_features,
)
from app.services.features.feature_sets import (
    mark_feature_set_failed,
    resolve_feature_set,
)
from app.services.features.fingerprinting import (
    calculate_journal_source_hash,
)
from app.services.features.text_features import (
    extract_text_features,
)
from app.tasks.celery_app import celery_app


# ============================================================
# CHILD DISPATCH BOUNDARIES
# ============================================================


def _dispatch_text_feature_task(
    journal_id: str,
    feature_set_id: str,
) -> None:
    extract_journal_text_features.delay(
        journal_id,
        feature_set_id,
    )


def _dispatch_audio_feature_task(
    journal_id: str,
    feature_set_id: str,
) -> None:
    extract_journal_audio_features.delay(
        journal_id,
        feature_set_id,
    )


# ============================================================
# PARENT ORCHESTRATION
# ============================================================


@celery_app.task(
    name=(
        "app.tasks.feature_tasks."
        "generate_journal_features"
    ),
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def generate_journal_features(
    journal_id: str,
):
    """
    Resolve the durable M3 feature generation and dispatch
    modality-specific child tasks.

    This task is the single owner of feature-generation
    identity.

    Children receive the exact JournalFeatureSet UUID and
    must never independently resolve/create generations.
    """

    db = SessionLocal()

    try:
        journal_uuid = UUID(
            journal_id
        )

        journal = db.get(
            JournalEntry,
            journal_uuid,
        )

        if journal is None:
            return {
                "status": "missing",
            }

        if journal.status != "COMPLETED":
            return {
                "status": (
                    "journal_not_completed"
                ),
            }

        source_hash = (
            calculate_journal_source_hash(
                journal
            )
        )

        resolution = (
            resolve_feature_set(
                db,
                journal_id=journal.id,
                source_hash=source_hash,
            )
        )

        feature_set = (
            resolution.feature_set
        )

        # Completed generations are immutable.
        if (
            feature_set.status
            == FEATURE_STATUS_COMPLETED
        ):
            return {
                "status": (
                    "already_completed"
                ),
                "feature_set_id": str(
                    feature_set.id
                ),
                "source_hash": (
                    source_hash
                ),
            }

        # Persist generation identity before any
        # Celery handoff.
        #
        # If dispatch fails, the parent can later
        # rediscover this exact generation and
        # repair the handoff.
        db.commit()

        feature_set_id = str(
            feature_set.id
        )

        journal_entry_type = (
            journal.entry_type
        )

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()

    # --------------------------------------------------------
    # DOWNSTREAM HANDOFF
    # --------------------------------------------------------
    #
    # Intentionally outside the database transaction.
    #
    # Redis/Celery delivery failure must not erase the
    # durable feature-generation identity.
    # --------------------------------------------------------

    if journal_entry_type == "TEXT":
        _dispatch_text_feature_task(
            journal_id,
            feature_set_id,
        )

        return {
            "status": "queued",
            "feature_set_id": (
                feature_set_id
            ),
            "modalities": [
                "TEXT",
            ],
        }

    if journal_entry_type == "VOICE":
        _dispatch_text_feature_task(
            journal_id,
            feature_set_id,
        )

        _dispatch_audio_feature_task(
            journal_id,
            feature_set_id,
        )

        return {
            "status": "queued",
            "feature_set_id": (
                feature_set_id
            ),
            "modalities": [
                "TEXT",
                "AUDIO",
            ],
        }

    raise ValueError(
        "Unsupported journal entry type: "
        f"{journal_entry_type}"
    )


# ============================================================
# CHILD VALIDATION
# ============================================================


def _load_child_context(
    db,
    *,
    journal_id: str,
    feature_set_id: str,
) -> tuple[
    JournalEntry | None,
    JournalFeatureSet | None,
]:
    """
    Load and validate the exact journal/generation pair
    supplied by the parent orchestration task.

    A child must never operate on a feature generation
    belonging to another journal.
    """

    journal_uuid = UUID(
        journal_id
    )

    feature_set_uuid = UUID(
        feature_set_id
    )

    journal = db.get(
        JournalEntry,
        journal_uuid,
    )

    feature_set = db.get(
        JournalFeatureSet,
        feature_set_uuid,
    )

    if (
        journal is not None
        and feature_set is not None
        and feature_set.journal_id
        != journal.id
    ):
        raise ValueError(
            "Feature set does not belong "
            "to journal"
        )

    return (
        journal,
        feature_set,
    )


# ============================================================
# CONCURRENT CHILD CONVERGENCE
# ============================================================


def _text_feature_exists_after_rollback(
    db,
    *,
    journal_uuid: UUID,
    feature_set_uuid: UUID,
) -> bool:
    """
    Determine whether a concurrent text worker won the
    persistence race.

    Must only be called after the losing transaction has
    been rolled back.
    """

    feature_set = db.get(
        JournalFeatureSet,
        feature_set_uuid,
    )

    if feature_set is None:
        return False

    if (
        feature_set.journal_id
        != journal_uuid
    ):
        return False

    return (
        feature_set.text_feature
        is not None
    )


def _audio_feature_exists_after_rollback(
    db,
    *,
    journal_uuid: UUID,
    feature_set_uuid: UUID,
) -> bool:
    """
    Determine whether a concurrent audio worker won the
    persistence race.

    Must only be called after the losing transaction has
    been rolled back.
    """

    feature_set = db.get(
        JournalFeatureSet,
        feature_set_uuid,
    )

    if feature_set is None:
        return False

    if (
        feature_set.journal_id
        != journal_uuid
    ):
        return False

    return (
        feature_set.audio_feature
        is not None
    )


def _mark_child_failure(
    db,
    *,
    journal_uuid: UUID | None,
    feature_set_uuid: UUID | None,
    modality: str,
    exc: Exception,
) -> None:
    """
    Persist a genuine child extraction failure against the
    exact generation supplied to the task.

    Concurrent duplicate-persistence races are handled
    before this helper is called.
    """

    if (
        journal_uuid is None
        or feature_set_uuid is None
    ):
        return

    feature_set = db.get(
        JournalFeatureSet,
        feature_set_uuid,
    )

    if feature_set is None:
        return

    if (
        feature_set.journal_id
        != journal_uuid
    ):
        return

    mark_feature_set_failed(
        feature_set,
        error_message=(
            f"{modality} feature extraction "
            f"failed: {type(exc).__name__}"
        ),
    )

    db.commit()


# ============================================================
# TEXT CHILD
# ============================================================


@celery_app.task(
    name=(
        "app.tasks.feature_tasks."
        "extract_journal_text_features"
    ),
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def extract_journal_text_features(
    journal_id: str,
    feature_set_id: str,
):
    """
    Extract text features for one exact feature generation.

    Duplicate Celery delivery is expected and safe.

    Concurrent duplicate deliveries may both perform
    extraction, but only one TextFeature may be persisted.
    A worker losing that persistence race converges on the
    artifact persisted by the winning worker.
    """

    db = SessionLocal()

    journal_uuid: UUID | None = None
    feature_set_uuid: UUID | None = None

    try:
        journal_uuid = UUID(
            journal_id
        )

        feature_set_uuid = UUID(
            feature_set_id
        )

        (
            journal,
            feature_set,
        ) = _load_child_context(
            db,
            journal_id=journal_id,
            feature_set_id=feature_set_id,
        )

        if journal is None:
            return {
                "status": "missing_journal",
            }

        if feature_set is None:
            return {
                "status": (
                    "missing_feature_set"
                ),
            }

        if journal.status != "COMPLETED":
            return {
                "status": (
                    "journal_not_completed"
                ),
            }

        # -------------------------------------------------
        # IDEMPOTENT DUPLICATE DELIVERY
        # -------------------------------------------------

        if feature_set.text_feature is not None:
            return {
                "status": (
                    "already_extracted"
                ),
                "feature_set_id": str(
                    feature_set.id
                ),
            }

        # A completed generation must never be mutated.
        if (
            feature_set.status
            == FEATURE_STATUS_COMPLETED
        ):
            return {
                "status": (
                    "already_completed"
                ),
                "feature_set_id": str(
                    feature_set.id
                ),
            }

        result = extract_text_features(
            db,
            journal=journal,
            feature_set=feature_set,
        )

        db.commit()

        return {
            "status": "completed",
            "feature_set_id": str(
                result.feature_set.id
            ),
        }

    except IntegrityError as exc:
        # -------------------------------------------------
        # CONCURRENT DUPLICATE PERSISTENCE
        # -------------------------------------------------
        #
        # Two workers can both pass the initial
        # text_feature-is-None check.
        #
        # PostgreSQL's UNIQUE(feature_set_id) constraint
        # chooses the persistence winner.
        #
        # The losing transaction must rollback before it
        # can inspect the winner's committed artifact.
        # -------------------------------------------------

        db.rollback()

        if (
            journal_uuid is not None
            and feature_set_uuid is not None
            and _text_feature_exists_after_rollback(
                db,
                journal_uuid=journal_uuid,
                feature_set_uuid=feature_set_uuid,
            )
        ):
            return {
                "status": "already_extracted",
                "feature_set_id": str(
                    feature_set_uuid
                ),
            }

        # The IntegrityError was not explained by another
        # worker successfully persisting the same modality.
        # Treat it as a genuine extraction/persistence
        # failure.
        _mark_child_failure(
            db,
            journal_uuid=journal_uuid,
            feature_set_uuid=feature_set_uuid,
            modality="Text",
            exc=exc,
        )

        raise

    except Exception as exc:
        db.rollback()

        _mark_child_failure(
            db,
            journal_uuid=journal_uuid,
            feature_set_uuid=feature_set_uuid,
            modality="Text",
            exc=exc,
        )

        raise

    finally:
        db.close()


# ============================================================
# AUDIO CHILD
# ============================================================


@celery_app.task(
    name=(
        "app.tasks.feature_tasks."
        "extract_journal_audio_features"
    ),
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def extract_journal_audio_features(
    journal_id: str,
    feature_set_id: str,
):
    """
    Extract deterministic and learned audio features for
    one exact VOICE feature generation.

    Duplicate Celery delivery is expected and safe.

    Concurrent duplicate deliveries may both perform
    extraction, but only one AudioFeature may be persisted.
    A worker losing that persistence race converges on the
    artifact persisted by the winning worker.
    """

    db = SessionLocal()

    journal_uuid: UUID | None = None
    feature_set_uuid: UUID | None = None

    try:
        journal_uuid = UUID(
            journal_id
        )

        feature_set_uuid = UUID(
            feature_set_id
        )

        (
            journal,
            feature_set,
        ) = _load_child_context(
            db,
            journal_id=journal_id,
            feature_set_id=feature_set_id,
        )

        if journal is None:
            return {
                "status": "missing_journal",
            }

        if feature_set is None:
            return {
                "status": (
                    "missing_feature_set"
                ),
            }

        if journal.status != "COMPLETED":
            return {
                "status": (
                    "journal_not_completed"
                ),
            }

        if journal.entry_type != "VOICE":
            return {
                "status": "not_voice",
            }

        # -------------------------------------------------
        # IDEMPOTENT DUPLICATE DELIVERY
        # -------------------------------------------------

        if feature_set.audio_feature is not None:
            return {
                "status": (
                    "already_extracted"
                ),
                "feature_set_id": str(
                    feature_set.id
                ),
            }

        # A completed generation must never be mutated.
        if (
            feature_set.status
            == FEATURE_STATUS_COMPLETED
        ):
            return {
                "status": (
                    "already_completed"
                ),
                "feature_set_id": str(
                    feature_set.id
                ),
            }

        result = extract_audio_features(
            db,
            journal=journal,
            feature_set=feature_set,
        )

        db.commit()

        return {
            "status": "completed",
            "feature_set_id": str(
                result.feature_set.id
            ),
        }

    except IntegrityError as exc:
        # Same convergence contract as the text child.
        db.rollback()

        if (
            journal_uuid is not None
            and feature_set_uuid is not None
            and _audio_feature_exists_after_rollback(
                db,
                journal_uuid=journal_uuid,
                feature_set_uuid=feature_set_uuid,
            )
        ):
            return {
                "status": "already_extracted",
                "feature_set_id": str(
                    feature_set_uuid
                ),
            }

        _mark_child_failure(
            db,
            journal_uuid=journal_uuid,
            feature_set_uuid=feature_set_uuid,
            modality="Audio",
            exc=exc,
        )

        raise

    except Exception as exc:
        db.rollback()

        _mark_child_failure(
            db,
            journal_uuid=journal_uuid,
            feature_set_uuid=feature_set_uuid,
            modality="Audio",
            exc=exc,
        )

        raise

    finally:
        db.close()