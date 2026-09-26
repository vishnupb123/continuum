from uuid import UUID

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
    fingerprint_text_journal,
    fingerprint_voice_journal,
)
from app.services.features.text_features import (
    extract_text_features,
)
from app.services.storage.factory import (
    get_audio_storage,
)
from app.tasks.celery_app import celery_app


def _calculate_source_hash(
    journal: JournalEntry,
) -> str:
    """
    Calculate the feature-generation fingerprint
    from the exact persisted journal source.

    TEXT:
        exact persisted raw text

    VOICE:
        exact persisted transcript + stored audio bytes
    """

    if journal.entry_type == "TEXT":
        if journal.raw_text is None:
            raise ValueError(
                "TEXT journal has no source text"
            )

        return fingerprint_text_journal(
            journal.raw_text
        ).value

    if journal.entry_type == "VOICE":
        if (
            journal.raw_text is None
            or not journal.raw_text.strip()
        ):
            raise ValueError(
                "VOICE journal has no transcript"
            )

        if journal.audio is None:
            raise ValueError(
                "VOICE journal has no audio"
            )

        storage = get_audio_storage()

        audio_bytes = storage.get(
            journal.audio.storage_key
        )

        return fingerprint_voice_journal(
            journal.raw_text,
            audio_bytes,
        ).value

    raise ValueError(
        "Unsupported journal entry type: "
        f"{journal.entry_type}"
    )


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
            _calculate_source_hash(
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

    except Exception as exc:
        db.rollback()

        # Persist failure only when the exact generation
        # supplied to this child can be safely reacquired.
        if (
            journal_uuid is not None
            and feature_set_uuid is not None
        ):
            feature_set = db.get(
                JournalFeatureSet,
                feature_set_uuid,
            )

            if (
                feature_set is not None
                and feature_set.journal_id
                == journal_uuid
            ):
                mark_feature_set_failed(
                    feature_set,
                    error_message=(
                        "Text feature extraction "
                        "failed: "
                        f"{type(exc).__name__}"
                    ),
                )

                db.commit()

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

    except Exception as exc:
        db.rollback()

        if (
            journal_uuid is not None
            and feature_set_uuid is not None
        ):
            feature_set = db.get(
                JournalFeatureSet,
                feature_set_uuid,
            )

            if (
                feature_set is not None
                and feature_set.journal_id
                == journal_uuid
            ):
                mark_feature_set_failed(
                    feature_set,
                    error_message=(
                        "Audio feature extraction "
                        "failed: "
                        f"{type(exc).__name__}"
                    ),
                )

                db.commit()

        raise

    finally:
        db.close()