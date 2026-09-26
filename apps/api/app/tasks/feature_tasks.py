from uuid import UUID

from app.db.session import SessionLocal
from app.models.journal import JournalEntry
from app.services.features.feature_sets import (
    get_feature_set,
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
):
    db = SessionLocal()

    source_hash: str | None = None
    journal_uuid: UUID | None = None

    try:
        journal_uuid = UUID(journal_id)

        journal = db.get(
            JournalEntry,
            journal_uuid,
        )

        if journal is None:
            return {
                "status": "missing",
            }

        # Feature extraction must never run against
        # an unfinished journal source.
        if journal.status != "COMPLETED":
            return {
                "status": "journal_not_completed",
            }

        source_hash = _calculate_source_hash(
            journal
        )

        resolution = resolve_feature_set(
            db,
            journal_id=journal.id,
            source_hash=source_hash,
        )

        # -------------------------------------------------
        # IDEMPOTENCY
        # -------------------------------------------------

        if not resolution.should_process:
            return {
                "status": "already_resolved",
                "feature_set_status": (
                    resolution.feature_set.status
                ),
                "feature_set_id": str(
                    resolution.feature_set.id
                ),
            }

        feature_set = resolution.feature_set

        # -------------------------------------------------
        # DURABLE GENERATION BOUNDARY
        # -------------------------------------------------
        #
        # A newly-created or retried generation must exist
        # durably before expensive feature extraction.
        #
        # If MPNet fails after this commit, rollback will
        # not erase the feature generation. We can then
        # reacquire it and persist FAILED.
        #
        db.commit()

        # Refresh/reacquire after transaction boundary.
        feature_set = get_feature_set(
            db,
            journal_id=journal.id,
            source_hash=source_hash,
        )

        if feature_set is None:
            raise RuntimeError(
                "Feature set disappeared after "
                "resolution commit"
            )

        result = extract_text_features(
            db,
            journal=journal,
            feature_set=feature_set,
        )

        # extract_text_features() marks COMPLETED but
        # intentionally does not commit.
        db.commit()

        return {
            "status": "completed",
            "feature_set_id": str(
                result.feature_set.id
            ),
            "source_hash": source_hash,
        }

    except Exception as exc:
        db.rollback()

        # -------------------------------------------------
        # FAILURE PERSISTENCE
        # -------------------------------------------------
        #
        # Only mark FAILED if a feature generation had
        # already been resolved. Failures before that
        # point (invalid UUID, missing source/audio, etc.)
        # have no generation to mark.
        #
        if (
            journal_uuid is not None
            and source_hash is not None
        ):
            feature_set = get_feature_set(
                db,
                journal_id=journal_uuid,
                source_hash=source_hash,
            )

            if feature_set is not None:
                mark_feature_set_failed(
                    feature_set,
                    error_message=(
                        "Text feature extraction failed: "
                        f"{type(exc).__name__}"
                    ),
                )

                db.commit()

        raise

    finally:
        db.close()