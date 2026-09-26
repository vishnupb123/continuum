from unittest.mock import patch

from sqlalchemy import func, select

from app.models import (
    JournalEntry,
    StateObservation,
)
from app.models.user import User
from app.tasks.journal_tasks import (
    process_journal,
)


def make_queued_text_journal(
    db_session,
):
    user = User(
        email="m37-handoff@example.com",
        password_hash="test-password-hash",
        display_name="M3.7 Handoff User",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="TEXT",
        raw_text=(
            "Today was productive, although "
            "I felt slightly tired by evening."
        ),
        status="QUEUED",
    )

    db_session.add(journal)
    db_session.commit()
    db_session.refresh(journal)

    return journal


# ============================================================
# NORMAL M2 -> M3 HANDOFF
# ============================================================


@patch(
    "app.tasks.journal_tasks.SessionLocal"
)
@patch(
    "app.tasks.journal_tasks."
    "generate_journal_features.delay"
)
def test_process_journal_queues_parent_feature_orchestrator(
    feature_delay_mock,
    session_local_mock,
    db_session,
):
    """
    A successfully processed journal must:

        1. persist its StateObservation
        2. become COMPLETED
        3. commit that state
        4. queue the M3 parent orchestrator exactly once

    process_journal must NOT know anything about
    text/audio child feature tasks.
    """

    session_local_mock.return_value = (
        db_session
    )

    journal = make_queued_text_journal(
        db_session
    )

    # Capture scalar identity before process_journal()
    # closes the patched SQLAlchemy session.
    journal_id = journal.id

    result = process_journal.run(
        str(journal_id)
    )

    assert (
        result["status"]
        == "completed"
    )

    assert (
        result["features"]
        == "queued"
    )

    feature_delay_mock.assert_called_once_with(
        str(journal_id)
    )

    db_session.expire_all()

    persisted_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    assert persisted_journal is not None

    assert (
        persisted_journal.status
        == "COMPLETED"
    )

    assert (
        persisted_journal.error_message
        is None
    )

    observation = db_session.scalar(
        select(
            StateObservation
        ).where(
            StateObservation.journal_id
            == journal_id
        )
    )

    assert observation is not None


# ============================================================
# LOST M3 DISPATCH RECOVERY
# ============================================================


@patch(
    "app.tasks.journal_tasks.SessionLocal"
)
@patch(
    "app.tasks.journal_tasks."
    "generate_journal_features.delay"
)
def test_completed_journal_retries_feature_handoff_without_duplicate_state(
    feature_delay_mock,
    session_local_mock,
    db_session,
):
    """
    If M2 analysis was committed successfully but the
    M3 Celery handoff was lost, rerunning process_journal
    must repair only the downstream handoff.

    It must NOT:
        - rerun MockContextModel
        - create another StateObservation
        - change the journal away from COMPLETED
    """

    session_local_mock.return_value = (
        db_session
    )

    journal = make_queued_text_journal(
        db_session
    )

    journal_id = journal.id

    # --------------------------------------------------------
    # FIRST DELIVERY
    # --------------------------------------------------------

    first_result = process_journal.run(
        str(journal_id)
    )

    assert (
        first_result["status"]
        == "completed"
    )

    feature_delay_mock.assert_called_once_with(
        str(journal_id)
    )

    db_session.expire_all()

    first_observation = db_session.scalar(
        select(
            StateObservation
        ).where(
            StateObservation.journal_id
            == journal_id
        )
    )

    assert first_observation is not None

    first_observation_id = (
        first_observation.id
    )

    # Simulate another delivery of process_journal.
    #
    # This represents recovery from a previously lost
    # downstream M3 dispatch.
    feature_delay_mock.reset_mock()

    # --------------------------------------------------------
    # SECOND DELIVERY / RECOVERY
    # --------------------------------------------------------

    second_result = process_journal.run(
        str(journal_id)
    )

    assert (
        second_result["status"]
        == "already_completed"
    )

    assert (
        second_result["features"]
        == "queued"
    )

    feature_delay_mock.assert_called_once_with(
        str(journal_id)
    )

    db_session.expire_all()

    persisted_journal = db_session.get(
        JournalEntry,
        journal_id,
    )

    assert persisted_journal is not None

    assert (
        persisted_journal.status
        == "COMPLETED"
    )

    # --------------------------------------------------------
    # NO DUPLICATE STATE OBSERVATION
    # --------------------------------------------------------

    observation_count = db_session.scalar(
        select(
            func.count(
                StateObservation.id
            )
        ).where(
            StateObservation.journal_id
            == journal_id
        )
    )

    assert observation_count == 1

    recovered_observation = db_session.scalar(
        select(
            StateObservation
        ).where(
            StateObservation.journal_id
            == journal_id
        )
    )

    assert recovered_observation is not None

    assert (
        recovered_observation.id
        == first_observation_id
    )