from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.feature_constants import (
    FEATURE_STATUS_COMPLETED,
    FEATURE_STATUS_FAILED,
    FEATURE_STATUS_PENDING,
    FEATURE_STATUS_PROCESSING,
)
from app.models.journal_feature_set import JournalFeatureSet
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


def resolve_feature_set(
    db: Session,
    *,
    journal_id: UUID,
    source_hash: str,
    pipeline_version: str = FEATURE_PIPELINE_VERSION,
) -> FeatureSetResolution:
    existing = get_feature_set(
        db,
        journal_id=journal_id,
        source_hash=source_hash,
        pipeline_version=pipeline_version,
    )

    if existing is None:
        feature_set = JournalFeatureSet(
            journal_id=journal_id,
            pipeline_version=pipeline_version,
            source_hash=source_hash,
            status=FEATURE_STATUS_PENDING,
        )

        db.add(feature_set)
        db.flush()

        return FeatureSetResolution(
            feature_set=feature_set,
            created=True,
            should_process=True,
        )

    if existing.status == FEATURE_STATUS_COMPLETED:
        return FeatureSetResolution(
            feature_set=existing,
            created=False,
            should_process=False,
        )

    if existing.status == FEATURE_STATUS_FAILED:
        existing.status = FEATURE_STATUS_PENDING
        existing.error_message = None
        existing.completed_at = None

        db.flush()

        return FeatureSetResolution(
            feature_set=existing,
            created=False,
            should_process=True,
        )

    if existing.status in {
        FEATURE_STATUS_PENDING,
        FEATURE_STATUS_PROCESSING,
    }:
        return FeatureSetResolution(
            feature_set=existing,
            created=False,
            should_process=False,
        )

    raise ValueError(
        f"Unsupported feature set status: "
        f"{existing.status}"
    )


def mark_feature_set_processing(
    feature_set: JournalFeatureSet,
) -> None:
    feature_set.status = FEATURE_STATUS_PROCESSING
    feature_set.error_message = None
    feature_set.completed_at = None


def mark_feature_set_completed(
    feature_set: JournalFeatureSet,
) -> None:
    feature_set.status = FEATURE_STATUS_COMPLETED
    feature_set.error_message = None
    feature_set.completed_at = datetime.now(
        timezone.utc
    )


def mark_feature_set_failed(
    feature_set: JournalFeatureSet,
    *,
    error_message: str,
) -> None:
    feature_set.status = FEATURE_STATUS_FAILED
    feature_set.error_message = error_message
    feature_set.completed_at = None
    
def complete_feature_set_if_ready(
    feature_set: JournalFeatureSet,
    *,
    entry_type: str,
) -> bool:
    if entry_type == "TEXT":
        if feature_set.text_feature is None:
            return False

        mark_feature_set_completed(
            feature_set
        )
        return True

    if entry_type == "VOICE":
        if feature_set.text_feature is None:
            return False

        if feature_set.audio_feature is None:
            return False

        mark_feature_set_completed(
            feature_set
        )
        return True

    raise ValueError(
        f"Unsupported journal entry type: "
        f"{entry_type}"
    )