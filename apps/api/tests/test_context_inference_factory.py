import uuid

import pytest

from app.models.context_constants import (
    CONTEXT_ARCHITECTURE_VERSION,
    CONTEXT_STATUS_PENDING,
)
from app.models.journal_feature_set import JournalFeatureSet
from app.services.context.inference_factory import (
    ContextInferenceFactoryError,
    build_pending_context_inference,
)
from app.services.context.model_release import (
    ContextModelRelease,
)
from app.services.context.model.state_contract import (
    StateCapability,
)


def make_feature_set():
    return JournalFeatureSet(
        id=uuid.uuid4(),
    )


def make_release(
    *,
    capability=StateCapability.RESEARCH,
    calibrated=False,
):
    return ContextModelRelease(
        architecture_version=CONTEXT_ARCHITECTURE_VERSION,
        model_revision="test-r1",
        model_artifact_hash="abc123",
        state_capability=capability,
        confidence_calibrated=calibrated,
    )


def test_builds_pending_inference_from_release():
    feature_set = make_feature_set()
    release = make_release()

    inference = build_pending_context_inference(
        feature_set=feature_set,
        release=release,
    )

    assert inference.feature_set_id == feature_set.id

    assert (
        inference.architecture_version
        == release.architecture_version
    )

    assert (
        inference.model_revision
        == release.model_revision
    )

    assert (
        inference.model_artifact_hash
        == release.model_artifact_hash
    )

    assert inference.state_capability == "RESEARCH"
    assert inference.status == CONTEXT_STATUS_PENDING


def test_pending_inference_contains_no_outputs():
    inference = build_pending_context_inference(
        feature_set=make_feature_set(),
        release=make_release(),
    )

    assert inference.error_message is None

    assert inference.representation_dimension is None
    assert inference.representation is None

    assert inference.energy is None
    assert inference.stress is None
    assert inference.positive_mood is None
    assert inference.social_connection is None

    assert inference.confidence_score is None
    assert inference.confidence_calibrated is False

    assert inference.completed_at is None


def test_validated_capability_is_preserved():
    inference = build_pending_context_inference(
        feature_set=make_feature_set(),
        release=make_release(
            capability=StateCapability.VALIDATED,
            calibrated=True,
        ),
    )

    assert inference.state_capability == "VALIDATED"

    # Pending rows have published no confidence result yet.
    assert inference.confidence_score is None
    assert inference.confidence_calibrated is False


def test_release_identity_is_preserved_exactly():
    release = ContextModelRelease(
        architecture_version=CONTEXT_ARCHITECTURE_VERSION,
        model_revision="2026-10-r7",
        model_artifact_hash="deadbeef1234",
        state_capability=StateCapability.RESEARCH,
        confidence_calibrated=False,
    )

    inference = build_pending_context_inference(
        feature_set=make_feature_set(),
        release=release,
    )

    assert (
        inference.architecture_version
        == CONTEXT_ARCHITECTURE_VERSION
    )
    assert inference.model_revision == "2026-10-r7"
    assert inference.model_artifact_hash == "deadbeef1234"


def test_unpersisted_feature_set_is_rejected():
    feature_set = JournalFeatureSet()

    with pytest.raises(
        ContextInferenceFactoryError,
        match="must be persisted",
    ):
        build_pending_context_inference(
            feature_set=feature_set,
            release=make_release(),
        )


def test_invalid_feature_set_type_is_rejected():
    with pytest.raises(
        ContextInferenceFactoryError,
        match="JournalFeatureSet",
    ):
        build_pending_context_inference(
            feature_set=object(),
            release=make_release(),
        )


def test_invalid_release_type_is_rejected():
    with pytest.raises(
        ContextInferenceFactoryError,
        match="ContextModelRelease",
    ):
        build_pending_context_inference(
            feature_set=make_feature_set(),
            release=object(),
        )