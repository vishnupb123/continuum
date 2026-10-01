from app.models.context_inference import ContextInference
from app.models.journal_feature_set import JournalFeatureSet
from app.models.context_constants import (
    CONTEXT_STATUS_PENDING,
)
from app.services.context.model_release import (
    ContextModelRelease,
)


class ContextInferenceFactoryError(ValueError):
    """Raised when a ContextInference cannot be safely constructed."""


def build_pending_context_inference(
    *,
    feature_set: JournalFeatureSet,
    release: ContextModelRelease,
) -> ContextInference:
    if not isinstance(feature_set, JournalFeatureSet):
        raise ContextInferenceFactoryError(
            "feature_set must be a JournalFeatureSet"
        )

    if not isinstance(release, ContextModelRelease):
        raise ContextInferenceFactoryError(
            "release must be a ContextModelRelease"
        )

    if feature_set.id is None:
        raise ContextInferenceFactoryError(
            "feature_set must be persisted before context inference"
        )

    return ContextInference(
        feature_set_id=feature_set.id,
        architecture_version=release.architecture_version,
        model_revision=release.model_revision,
        model_artifact_hash=release.model_artifact_hash,
        state_capability=release.state_capability.value,
        status=CONTEXT_STATUS_PENDING,
        error_message=None,
        representation_dimension=None,
        representation=None,
        energy=None,
        stress=None,
        positive_mood=None,
        social_connection=None,
        confidence_score=None,
        confidence_calibrated=False,
        inference_metadata={},
        completed_at=None,
    )