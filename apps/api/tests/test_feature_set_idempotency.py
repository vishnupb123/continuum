from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.models.journal import JournalEntry
from app.models.journal_feature_set import (
    JournalFeatureSet,
)
from app.models.user import User
from app.services.features.feature_sets import (
    get_feature_set,
    resolve_feature_set,
)
from unittest.mock import patch

import app.services.features.feature_sets as feature_set_service
# Reuse the real PostgreSQL test engine/session factory.
from tests.conftest import TestingSessionLocal

from app.services.features.fingerprinting import (
    calculate_journal_source_hash,
)

from app.services.features.feature_sets import (
    get_current_completed_feature_set,
)


def make_journal(
    db_session,
    *,
    email: str = "m38@example.com",
):
    user = User(
        email=email,
        password_hash="test-password-hash",
        display_name="M3.8 User",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="TEXT",
        raw_text=(
            "M3.8 generation identity and "
            "idempotency test journal."
        ),
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.commit()
    db_session.refresh(journal)

    return journal


# ============================================================
# SAME IDENTITY -> SAME GENERATION
# ============================================================


def test_same_identity_reuses_same_generation(
    db_session,
):
    journal = make_journal(
        db_session,
    )

    journal_id = journal.id

    first = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash="same-source-hash",
        pipeline_version="pipeline-v1",
    )

    db_session.commit()

    first_id = first.feature_set.id

    second = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash="same-source-hash",
        pipeline_version="pipeline-v1",
    )

    assert (
        second.feature_set.id
        == first_id
    )

    assert second.created is False


# ============================================================
# SOURCE CHANGE -> NEW GENERATION
# ============================================================


def test_source_change_creates_new_generation(
    db_session,
):
    journal = make_journal(
        db_session,
    )

    journal_id = journal.id

    first = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash="source-version-a",
        pipeline_version="pipeline-v1",
    )

    db_session.commit()

    first_id = first.feature_set.id

    second = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash="source-version-b",
        pipeline_version="pipeline-v1",
    )

    db_session.commit()

    second_id = second.feature_set.id

    assert first_id != second_id

    count = db_session.scalar(
        select(
            func.count(
                JournalFeatureSet.id
            )
        ).where(
            JournalFeatureSet.journal_id
            == journal_id
        )
    )

    assert count == 2


# ============================================================
# PIPELINE CHANGE -> NEW GENERATION
# ============================================================


def test_pipeline_change_creates_new_generation(
    db_session,
):
    journal = make_journal(
        db_session,
    )

    journal_id = journal.id

    first = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash="same-content",
        pipeline_version="pipeline-v1",
    )

    db_session.commit()

    first_id = first.feature_set.id

    second = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash="same-content",
        pipeline_version="pipeline-v2",
    )

    db_session.commit()

    second_id = second.feature_set.id

    assert first_id != second_id

    count = db_session.scalar(
        select(
            func.count(
                JournalFeatureSet.id
            )
        ).where(
            JournalFeatureSet.journal_id
            == journal_id
        )
    )

    assert count == 2


# ============================================================
# OLD GENERATIONS ARE PRESERVED
# ============================================================


def test_new_generation_does_not_overwrite_old_generation(
    db_session,
):
    journal = make_journal(
        db_session,
    )

    journal_id = journal.id

    first = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash="historical-source",
        pipeline_version="pipeline-v1",
    )

    db_session.commit()

    first_id = first.feature_set.id

    second = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash="current-source",
        pipeline_version="pipeline-v2",
    )

    db_session.commit()

    second_id = second.feature_set.id

    historical = db_session.get(
        JournalFeatureSet,
        first_id,
    )

    current = db_session.get(
        JournalFeatureSet,
        second_id,
    )

    assert historical is not None
    assert current is not None

    assert (
        historical.source_hash
        == "historical-source"
    )

    assert (
        historical.pipeline_version
        == "pipeline-v1"
    )

    assert (
        current.source_hash
        == "current-source"
    )

    assert (
        current.pipeline_version
        == "pipeline-v2"
    )


# ============================================================
# CONCURRENT SAME-IDENTITY RESOLUTION
# ============================================================


def test_concurrent_same_identity_converges_on_one_generation(
    db_session,
):
    """
    Force the real SELECT -> INSERT race deterministically.

    Both transactions must:

        1. independently SELECT the generation
        2. both observe that it does not exist
        3. only then proceed toward INSERT

    M3.8 contract:

        - no IntegrityError escapes
        - both callers receive the same generation
        - exactly one database row exists

    The pre-M3.8 resolver is expected to fail this test.
    """

    journal = make_journal(
        db_session,
        email="m38-concurrency@example.com",
    )

    journal_id = journal.id

    source_hash = (
        "concurrent-source-hash"
    )

    pipeline_version = (
        "pipeline-concurrency-v1"
    )

    # --------------------------------------------------------
    # DETERMINISTIC RACE BARRIER
    # --------------------------------------------------------
    #
    # The previous test synchronized workers BEFORE
    # resolve_feature_set().
    #
    # That still allowed:
    #
    #   A: SELECT -> INSERT -> COMMIT
    #   B: SELECT -> sees A
    #
    # Here we synchronize them AFTER each worker has
    # completed get_feature_set() and observed None.
    #
    # Therefore both workers enter the creation path.
    # --------------------------------------------------------

    select_barrier = Barrier(2)

    original_get_feature_set = (
        feature_set_service.get_feature_set
    )

    def synchronized_get_feature_set(
        db,
        *,
        journal_id,
        source_hash,
        pipeline_version,
    ):
        result = original_get_feature_set(
            db,
            journal_id=journal_id,
            source_hash=source_hash,
            pipeline_version=pipeline_version,
        )

        # We only synchronize the initial "not found"
        # result. Future lookups performed by the M3.8
        # recovery implementation must not deadlock.
        if result is None:
            select_barrier.wait(
                timeout=5
            )

        return result

    def worker():
        session = TestingSessionLocal()

        try:
            result = resolve_feature_set(
                session,
                journal_id=journal_id,
                source_hash=source_hash,
                pipeline_version=pipeline_version,
            )

            generation_id = (
                result.feature_set.id
            )

            session.commit()

            return generation_id

        finally:
            session.close()

    with patch.object(
        feature_set_service,
        "get_feature_set",
        side_effect=synchronized_get_feature_set,
    ):
        with ThreadPoolExecutor(
            max_workers=2
        ) as executor:
            futures = [
                executor.submit(worker)
                for _ in range(2)
            ]

            generation_ids = [
                future.result(
                    timeout=10
                )
                for future in futures
            ]

    # --------------------------------------------------------
    # BOTH CALLERS CONVERGE
    # --------------------------------------------------------

    assert (
        len(set(generation_ids))
        == 1
    )

    db_session.expire_all()

    # --------------------------------------------------------
    # DATABASE CONTAINS EXACTLY ONE GENERATION
    # --------------------------------------------------------

    count = db_session.scalar(
        select(
            func.count(
                JournalFeatureSet.id
            )
        ).where(
            JournalFeatureSet.journal_id
            == journal_id,
            JournalFeatureSet.source_hash
            == source_hash,
            JournalFeatureSet.pipeline_version
            == pipeline_version,
        )
    )

    assert count == 1

# ============================================================
# LOOKUP IDENTITY IS EXACT
# ============================================================


def test_get_feature_set_uses_full_generation_identity(
    db_session,
):
    journal = make_journal(
        db_session,
    )

    journal_id = journal.id

    first = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash="shared-source",
        pipeline_version="pipeline-v1",
    )

    second = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash="shared-source",
        pipeline_version="pipeline-v2",
    )

    db_session.commit()

    found_v1 = get_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash="shared-source",
        pipeline_version="pipeline-v1",
    )

    found_v2 = get_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash="shared-source",
        pipeline_version="pipeline-v2",
    )

    assert found_v1 is not None
    assert found_v2 is not None

    assert (
        found_v1.id
        == first.feature_set.id
    )

    assert (
        found_v2.id
        == second.feature_set.id
    )

    assert (
        found_v1.id
        != found_v2.id
    )
    
def test_failed_generation_is_recovered_in_place(
    db_session,
):
    journal = make_journal(
        db_session,
        email="m38-failed@example.com",
    )

    journal_id = journal.id

    first = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash="failed-source",
        pipeline_version="pipeline-v1",
    )

    feature_set_id = first.feature_set.id

    first.feature_set.status = "FAILED"
    first.feature_set.error_message = (
        "Simulated extraction failure"
    )

    db_session.commit()

    recovered = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash="failed-source",
        pipeline_version="pipeline-v1",
    )

    db_session.commit()

    assert recovered.feature_set.id == feature_set_id
    assert recovered.created is False
    assert recovered.should_process is True

    assert recovered.feature_set.status == "PENDING"
    assert recovered.feature_set.error_message is None
    assert recovered.feature_set.completed_at is None

    count = db_session.scalar(
        select(
            func.count(
                JournalFeatureSet.id
            )
        ).where(
            JournalFeatureSet.journal_id
            == journal_id,
            JournalFeatureSet.source_hash
            == "failed-source",
            JournalFeatureSet.pipeline_version
            == "pipeline-v1",
        )
    )

    assert count == 1


def test_completed_generation_is_terminal_for_same_identity(
    db_session,
):
    journal = make_journal(
        db_session,
        email="m38-completed@example.com",
    )

    journal_id = journal.id

    first = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash="completed-source",
        pipeline_version="pipeline-v1",
    )

    feature_set_id = first.feature_set.id

    first.feature_set.status = "COMPLETED"

    db_session.commit()

    resolved = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash="completed-source",
        pipeline_version="pipeline-v1",
    )

    assert resolved.feature_set.id == feature_set_id
    assert resolved.created is False
    assert resolved.should_process is False
    assert resolved.feature_set.status == "COMPLETED"


@pytest.mark.parametrize(
    "status",
    [
        "PENDING",
        "PROCESSING",
    ],
)
def test_active_generation_is_reused_without_new_generation(
    db_session,
    status,
):
    journal = make_journal(
        db_session,
        email=f"m38-{status.lower()}@example.com",
    )

    journal_id = journal.id

    first = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash=f"{status.lower()}-source",
        pipeline_version="pipeline-v1",
    )

    feature_set_id = first.feature_set.id

    first.feature_set.status = status

    db_session.commit()

    resolved = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash=f"{status.lower()}-source",
        pipeline_version="pipeline-v1",
    )

    assert resolved.feature_set.id == feature_set_id
    assert resolved.created is False
    assert resolved.should_process is False
    assert resolved.feature_set.status == status


def test_completed_historical_generation_survives_new_source_generation(
    db_session,
):
    journal = make_journal(
        db_session,
        email="m38-history@example.com",
    )

    journal_id = journal.id

    historical = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash="source-before-edit",
        pipeline_version="pipeline-v1",
    )

    historical_id = historical.feature_set.id

    historical.feature_set.status = "COMPLETED"

    db_session.commit()

    current = resolve_feature_set(
        db_session,
        journal_id=journal_id,
        source_hash="source-after-edit",
        pipeline_version="pipeline-v1",
    )

    current_id = current.feature_set.id

    db_session.commit()

    assert current_id != historical_id

    preserved = db_session.get(
        JournalFeatureSet,
        historical_id,
    )

    assert preserved is not None
    assert preserved.status == "COMPLETED"
    assert preserved.source_hash == "source-before-edit"

    assert current.feature_set.source_hash == "source-after-edit"
    assert current.feature_set.status == "PENDING"

    count = db_session.scalar(
        select(
            func.count(
                JournalFeatureSet.id
            )
        ).where(
            JournalFeatureSet.journal_id
            == journal_id
        )
    )

    assert count == 2
def test_current_completed_generation_is_selectable(
    db_session,
    ):
    journal = make_journal(
        db_session,
        email="m38-current-completed@example.com",
    )

    source_hash = (
        calculate_journal_source_hash(
            journal
        )
    )

    resolution = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash=source_hash,
    )

    feature_set_id = (
        resolution.feature_set.id
    )

    resolution.feature_set.status = (
        "COMPLETED"
    )

    db_session.commit()

    selected = (
        get_current_completed_feature_set(
            db_session,
            journal=journal,
        )
    )

    assert selected is not None
    assert selected.id == feature_set_id


@pytest.mark.parametrize(
    "status",
    [
        "PENDING",
        "PROCESSING",
        "FAILED",
    ],
)
def test_current_noncompleted_generation_is_not_selectable(
    db_session,
    status,
):
    journal = make_journal(
        db_session,
        email=(
            f"m38-current-"
            f"{status.lower()}@example.com"
        ),
    )

    source_hash = (
        calculate_journal_source_hash(
            journal
        )
    )

    resolution = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash=source_hash,
    )

    resolution.feature_set.status = status

    db_session.commit()

    selected = (
        get_current_completed_feature_set(
            db_session,
            journal=journal,
        )
    )

    assert selected is None


def test_historical_completed_generation_is_not_current_after_edit(
    db_session,
):
    journal = make_journal(
        db_session,
        email="m38-stale-generation@example.com",
    )

    old_source_hash = (
        calculate_journal_source_hash(
            journal
        )
    )

    old_generation = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash=old_source_hash,
    )

    old_generation_id = (
        old_generation.feature_set.id
    )

    old_generation.feature_set.status = (
        "COMPLETED"
    )

    db_session.commit()

    # The persisted journal source changes.
    journal.raw_text = (
        "This journal has now been edited, "
        "so its feature identity must change."
    )

    db_session.commit()
    db_session.refresh(journal)

    new_source_hash = (
        calculate_journal_source_hash(
            journal
        )
    )

    assert (
        new_source_hash
        != old_source_hash
    )

    selected = (
        get_current_completed_feature_set(
            db_session,
            journal=journal,
        )
    )

    # The historical generation is COMPLETED,
    # but it represents stale source content.
    assert selected is None

    historical = db_session.get(
        JournalFeatureSet,
        old_generation_id,
    )

    assert historical is not None
    assert historical.status == "COMPLETED"


def test_historical_completed_generation_is_not_used_when_current_failed(
    db_session,
):
    journal = make_journal(
        db_session,
        email="m38-current-failed@example.com",
    )

    old_source_hash = (
        calculate_journal_source_hash(
            journal
        )
    )

    historical = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash=old_source_hash,
    )

    historical.feature_set.status = (
        "COMPLETED"
    )

    historical_id = (
        historical.feature_set.id
    )

    db_session.commit()

    # Change the journal source.
    journal.raw_text = (
        "A new journal version whose feature "
        "generation later fails."
    )

    db_session.commit()
    db_session.refresh(journal)

    current_source_hash = (
        calculate_journal_source_hash(
            journal
        )
    )

    current = resolve_feature_set(
        db_session,
        journal_id=journal.id,
        source_hash=current_source_hash,
    )

    current.feature_set.status = "FAILED"
    current.feature_set.error_message = (
        "Simulated current-generation failure"
    )

    current_id = current.feature_set.id

    db_session.commit()

    assert current_id != historical_id

    selected = (
        get_current_completed_feature_set(
            db_session,
            journal=journal,
        )
    )

    # Never fall back to historical features.
    assert selected is None


def test_no_generation_returns_none(
    db_session,
):
    journal = make_journal(
        db_session,
        email="m38-no-generation@example.com",
    )

    selected = (
        get_current_completed_feature_set(
            db_session,
            journal=journal,
        )
    )

    assert selected is None