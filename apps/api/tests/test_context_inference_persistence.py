import hashlib

import numpy as np
import pytest
from sqlalchemy.exc import IntegrityError

from app.models.context_inference import ContextInference
from app.models.journal import JournalEntry
from app.models.journal_feature_set import JournalFeatureSet
from app.models.user import User


ARCHITECTURE_VERSION = "context-mode-v2-gated"
MODEL_REVISION = "test-r1"
MODEL_ARTIFACT_HASH = hashlib.sha256(
    b"context-mode-test-artifact-r1"
).hexdigest()


def make_feature_set(
    db_session,
    *,
    email="context-inference@example.com",
):
    user = User(
        email=email,
        password_hash="test-password-hash",
        display_name="Context Inference Test User",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="TEXT",
        raw_text=(
            "Context-MoDE persistence contract test."
        ),
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.flush()

    source_hash = hashlib.sha256(
        journal.raw_text.encode("utf-8")
    ).hexdigest()

    feature_set = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash=source_hash,
        status="COMPLETED",
    )

    db_session.add(feature_set)
    db_session.commit()
    db_session.refresh(feature_set)

    return feature_set


def make_representation():
    return [
        float(index) / 256.0
        for index in range(256)
    ]


def make_completed_inference(
    feature_set,
    *,
    model_revision=MODEL_REVISION,
):
    return ContextInference(
        feature_set_id=feature_set.id,
        architecture_version=ARCHITECTURE_VERSION,
        model_revision=model_revision,
        model_artifact_hash=MODEL_ARTIFACT_HASH,
        status="COMPLETED",
        representation_dimension=256,
        representation=make_representation(),
        energy=0.70,
        stress=0.30,
        positive_mood=0.80,
        social_connection=0.60,
        confidence_score=0.85,
        confidence_calibrated=False,
        inference_metadata={
            "test": True,
            "entry_type": "TEXT",
        },
    )


def test_context_inference_persists(db_session):
    feature_set = make_feature_set(
        db_session,
        email="context-persist@example.com",
    )

    inference = make_completed_inference(
        feature_set
    )

    db_session.add(inference)
    db_session.commit()

    inference_id = inference.id

    db_session.expire_all()

    stored = db_session.get(
        ContextInference,
        inference_id,
    )

    assert stored is not None

    assert (
        stored.feature_set_id
        == feature_set.id
    )

    assert (
        stored.architecture_version
        == ARCHITECTURE_VERSION
    )

    assert (
        stored.model_revision
        == MODEL_REVISION
    )

    assert (
        stored.model_artifact_hash
        == MODEL_ARTIFACT_HASH
    )

    assert stored.status == "COMPLETED"

    assert (
        stored.representation_dimension
        == 256
    )

    assert stored.representation is not None
    assert len(stored.representation) == 256

    np.testing.assert_allclose(
        np.asarray(
            stored.representation,
            dtype=np.float32,
        ),
        np.asarray(
            make_representation(),
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )

    assert stored.energy == pytest.approx(0.70)
    assert stored.stress == pytest.approx(0.30)

    assert (
        stored.positive_mood
        == pytest.approx(0.80)
    )

    assert (
        stored.social_connection
        == pytest.approx(0.60)
    )

    assert (
        stored.confidence_score
        == pytest.approx(0.85)
    )

    assert stored.confidence_calibrated is False

    assert stored.inference_metadata == {
        "test": True,
        "entry_type": "TEXT",
    }


def test_feature_set_relationship_works(
    db_session,
):
    feature_set = make_feature_set(
        db_session,
        email="context-relationship@example.com",
    )

    inference = make_completed_inference(
        feature_set
    )

    feature_set.context_inferences.append(
        inference
    )

    db_session.commit()

    db_session.expire_all()

    stored_feature_set = db_session.get(
        JournalFeatureSet,
        feature_set.id,
    )

    assert stored_feature_set is not None

    assert (
        len(
            stored_feature_set.context_inferences
        )
        == 1
    )

    assert (
        stored_feature_set
        .context_inferences[0]
        .id
        == inference.id
    )

    assert (
        stored_feature_set
        .context_inferences[0]
        .feature_set
        .id
        == feature_set.id
    )


def test_pending_inference_allows_null_outputs(
    db_session,
):
    feature_set = make_feature_set(
        db_session,
        email="context-pending@example.com",
    )

    inference = ContextInference(
        feature_set_id=feature_set.id,
        architecture_version=ARCHITECTURE_VERSION,
        model_revision=MODEL_REVISION,
        model_artifact_hash=MODEL_ARTIFACT_HASH,
        status="PENDING",
        confidence_calibrated=False,
        inference_metadata={},
    )

    db_session.add(inference)
    db_session.commit()

    db_session.refresh(inference)

    assert inference.status == "PENDING"
    assert inference.representation is None

    assert (
        inference.representation_dimension
        is None
    )

    assert inference.energy is None
    assert inference.stress is None
    assert inference.positive_mood is None
    assert inference.social_connection is None
    assert inference.confidence_score is None


def test_same_inference_generation_cannot_be_duplicated(
    db_session,
):
    feature_set = make_feature_set(
        db_session,
        email="context-duplicate@example.com",
    )

    first = ContextInference(
        feature_set_id=feature_set.id,
        architecture_version=ARCHITECTURE_VERSION,
        model_revision=MODEL_REVISION,
        model_artifact_hash=MODEL_ARTIFACT_HASH,
        status="PENDING",
        confidence_calibrated=False,
        inference_metadata={},
    )

    db_session.add(first)
    db_session.commit()

    duplicate = ContextInference(
        feature_set_id=feature_set.id,
        architecture_version=ARCHITECTURE_VERSION,
        model_revision=MODEL_REVISION,
        model_artifact_hash=MODEL_ARTIFACT_HASH,
        status="PENDING",
        confidence_calibrated=False,
        inference_metadata={},
    )

    db_session.add(duplicate)

    with pytest.raises(IntegrityError):
        db_session.commit()

    db_session.rollback()


def test_same_feature_set_allows_new_model_revision(
    db_session,
):
    feature_set = make_feature_set(
        db_session,
        email="context-new-revision@example.com",
    )

    r1 = ContextInference(
        feature_set_id=feature_set.id,
        architecture_version=ARCHITECTURE_VERSION,
        model_revision="test-r1",
        model_artifact_hash=hashlib.sha256(
            b"artifact-r1"
        ).hexdigest(),
        status="PENDING",
        confidence_calibrated=False,
        inference_metadata={},
    )

    r2 = ContextInference(
        feature_set_id=feature_set.id,
        architecture_version=ARCHITECTURE_VERSION,
        model_revision="test-r2",
        model_artifact_hash=hashlib.sha256(
            b"artifact-r2"
        ).hexdigest(),
        status="PENDING",
        confidence_calibrated=False,
        inference_metadata={},
    )

    db_session.add_all([r1, r2])
    db_session.commit()

    assert r1.id != r2.id

    assert (
        len(feature_set.context_inferences)
        == 2
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("energy", -0.01),
        ("energy", 1.01),
        ("stress", -0.01),
        ("stress", 1.01),
        ("positive_mood", -0.01),
        ("positive_mood", 1.01),
        ("social_connection", -0.01),
        ("social_connection", 1.01),
        ("confidence_score", -0.01),
        ("confidence_score", 1.01),
    ],
)
def test_state_and_confidence_ranges_are_enforced(
    db_session,
    field_name,
    invalid_value,
):
    feature_set = make_feature_set(
        db_session,
        email=(
            "context-range-"
            f"{field_name}-"
            f"{str(invalid_value).replace('.', '-')}"
            "@example.com"
        ),
    )

    inference = make_completed_inference(
        feature_set
    )

    setattr(
        inference,
        field_name,
        invalid_value,
    )

    db_session.add(inference)

    with pytest.raises(IntegrityError):
        db_session.commit()

    db_session.rollback()


@pytest.mark.parametrize(
    "boundary_value",
    [0.0, 1.0],
)
def test_state_boundaries_are_valid(
    db_session,
    boundary_value,
):
    feature_set = make_feature_set(
        db_session,
        email=(
            "context-boundary-"
            f"{boundary_value}"
            "@example.com"
        ),
    )

    inference = make_completed_inference(
        feature_set
    )

    inference.energy = boundary_value
    inference.stress = boundary_value
    inference.positive_mood = boundary_value
    inference.social_connection = boundary_value
    inference.confidence_score = boundary_value

    db_session.add(inference)
    db_session.commit()

    assert inference.energy == pytest.approx(
        boundary_value
    )

    assert (
        inference.confidence_score
        == pytest.approx(boundary_value)
    )


def test_deleting_feature_set_cascades_context_inference(
    db_session,
):
    feature_set = make_feature_set(
        db_session,
        email="context-cascade@example.com",
    )

    inference = make_completed_inference(
        feature_set
    )

    db_session.add(inference)
    db_session.commit()

    inference_id = inference.id
    feature_set_id = feature_set.id

    db_session.delete(feature_set)
    db_session.commit()

    assert (
        db_session.get(
            JournalFeatureSet,
            feature_set_id,
        )
        is None
    )

    assert (
        db_session.get(
            ContextInference,
            inference_id,
        )
        is None
    )