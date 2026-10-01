import pytest
import torch

from app.models.context_constants import (
    CONTEXT_ARCHITECTURE_VERSION,
    CONTEXT_STATUS_COMPLETED,
    CONTEXT_STATUS_FAILED,
    CONTEXT_STATUS_PENDING,
    CONTEXT_STATUS_PROCESSING,
)
from app.models.context_inference import ContextInference
from app.services.context.inference_lifecycle import (
    ContextInferenceLifecycleError,
    complete_context_inference,
    fail_context_inference,
    mark_context_inference_processing,
    reset_failed_context_inference_for_retry,
)
from app.services.context.model.output_contract import (
    ContextModelOutput,
)
from app.services.context.model.state_contract import (
    StateCapability,
)


def make_inference(
    *,
    status=CONTEXT_STATUS_PENDING,
    capability="RESEARCH",
):
    return ContextInference(
        architecture_version=CONTEXT_ARCHITECTURE_VERSION,
        model_revision="test-r1",
        model_artifact_hash="abc123",
        state_capability=capability,
        status=status,
        confidence_calibrated=False,
    )


def make_output(
    *,
    capability=StateCapability.RESEARCH,
    calibrated=False,
):
    return ContextModelOutput(
        representation=torch.randn(256),
        state=torch.tensor(
            [0.2, 0.7, 0.6, 0.4],
            dtype=torch.float32,
        ),
        confidence_score=torch.tensor(
            0.8,
            dtype=torch.float32,
        ),
        confidence_calibrated=calibrated,
        state_capability=capability,
    )


def test_pending_can_enter_processing():
    inference = make_inference()

    mark_context_inference_processing(inference)

    assert inference.status == CONTEXT_STATUS_PROCESSING
    assert inference.error_message is None
    assert inference.completed_at is None


@pytest.mark.parametrize(
    "status",
    [
        CONTEXT_STATUS_PROCESSING,
        CONTEXT_STATUS_FAILED,
        CONTEXT_STATUS_COMPLETED,
    ],
)
def test_only_pending_can_enter_processing(status):
    inference = make_inference(status=status)

    with pytest.raises(
        ContextInferenceLifecycleError,
        match="Only PENDING",
    ):
        mark_context_inference_processing(inference)


def test_research_completion_persists_representation_only():
    inference = make_inference(
        status=CONTEXT_STATUS_PROCESSING,
        capability="RESEARCH",
    )

    output = make_output(
        capability=StateCapability.RESEARCH,
    )

    complete_context_inference(
        inference,
        output,
    )

    assert inference.status == CONTEXT_STATUS_COMPLETED
    assert inference.representation_dimension == 256
    assert len(inference.representation) == 256

    assert inference.energy is None
    assert inference.stress is None
    assert inference.positive_mood is None
    assert inference.social_connection is None

    assert inference.confidence_score is None
    assert inference.confidence_calibrated is False

    assert inference.error_message is None
    assert inference.completed_at is not None


def test_validated_completion_persists_state_and_confidence():
    inference = make_inference(
        status=CONTEXT_STATUS_PROCESSING,
        capability="VALIDATED",
    )

    output = make_output(
        capability=StateCapability.VALIDATED,
        calibrated=True,
    )

    complete_context_inference(
        inference,
        output,
    )

    assert inference.status == CONTEXT_STATUS_COMPLETED
    assert inference.representation_dimension == 256
    assert len(inference.representation) == 256

    assert inference.energy == pytest.approx(0.2)
    assert inference.stress == pytest.approx(0.7)
    assert inference.positive_mood == pytest.approx(0.6)
    assert inference.social_connection == pytest.approx(0.4)

    assert inference.confidence_score == pytest.approx(0.8)
    assert inference.confidence_calibrated is True
    assert inference.completed_at is not None


def test_capability_mismatch_is_rejected():
    inference = make_inference(
        status=CONTEXT_STATUS_PROCESSING,
        capability="RESEARCH",
    )

    output = make_output(
        capability=StateCapability.VALIDATED,
    )

    with pytest.raises(
        ContextInferenceLifecycleError,
        match="capability does not match",
    ):
        complete_context_inference(
            inference,
            output,
        )


def test_only_processing_can_complete():
    inference = make_inference(
        status=CONTEXT_STATUS_PENDING,
    )

    with pytest.raises(
        ContextInferenceLifecycleError,
        match="Only PROCESSING",
    ):
        complete_context_inference(
            inference,
            make_output(),
        )


def test_completed_is_terminal_against_failure():
    inference = make_inference(
        status=CONTEXT_STATUS_COMPLETED,
    )

    with pytest.raises(
        ContextInferenceLifecycleError,
        match="terminal",
    ):
        fail_context_inference(
            inference,
            error_message="safe failure",
        )


@pytest.mark.parametrize(
    "status",
    [
        CONTEXT_STATUS_PENDING,
        CONTEXT_STATUS_PROCESSING,
    ],
)
def test_pending_or_processing_can_fail(status):
    inference = make_inference(status=status)

    fail_context_inference(
        inference,
        error_message="safe failure",
    )

    assert inference.status == CONTEXT_STATUS_FAILED
    assert inference.error_message == "safe failure"
    assert inference.completed_at is None


def test_failure_clears_partial_outputs():
    inference = make_inference(
        status=CONTEXT_STATUS_PROCESSING,
    )

    inference.representation_dimension = 256
    inference.representation = [0.1] * 256

    inference.energy = 0.2
    inference.stress = 0.7
    inference.positive_mood = 0.6
    inference.social_connection = 0.4

    inference.confidence_score = 0.8
    inference.confidence_calibrated = True

    fail_context_inference(
        inference,
        error_message="safe failure",
    )

    assert inference.representation_dimension is None
    assert inference.representation is None

    assert inference.energy is None
    assert inference.stress is None
    assert inference.positive_mood is None
    assert inference.social_connection is None

    assert inference.confidence_score is None
    assert inference.confidence_calibrated is False


def test_blank_failure_message_is_rejected():
    inference = make_inference(
        status=CONTEXT_STATUS_PROCESSING,
    )

    with pytest.raises(
        ContextInferenceLifecycleError,
        match="non-blank",
    ):
        fail_context_inference(
            inference,
            error_message="   ",
        )


def test_failed_can_reset_to_pending():
    inference = make_inference(
        status=CONTEXT_STATUS_FAILED,
    )

    inference.error_message = "safe failure"

    reset_failed_context_inference_for_retry(inference)

    assert inference.status == CONTEXT_STATUS_PENDING
    assert inference.error_message is None
    assert inference.completed_at is None


def test_retry_clears_stale_outputs():
    inference = make_inference(
        status=CONTEXT_STATUS_FAILED,
    )

    inference.error_message = "safe failure"
    inference.representation_dimension = 256
    inference.representation = [0.1] * 256
    inference.energy = 0.2
    inference.confidence_score = 0.8
    inference.confidence_calibrated = True

    reset_failed_context_inference_for_retry(inference)

    assert inference.representation_dimension is None
    assert inference.representation is None
    assert inference.energy is None
    assert inference.confidence_score is None
    assert inference.confidence_calibrated is False


@pytest.mark.parametrize(
    "status",
    [
        CONTEXT_STATUS_PENDING,
        CONTEXT_STATUS_PROCESSING,
        CONTEXT_STATUS_COMPLETED,
    ],
)
def test_only_failed_can_be_retried(status):
    inference = make_inference(status=status)

    with pytest.raises(
        ContextInferenceLifecycleError,
        match="Only FAILED",
    ):
        reset_failed_context_inference_for_retry(inference)


def test_unsupported_persisted_capability_is_rejected():
    inference = make_inference(
        status=CONTEXT_STATUS_PROCESSING,
        capability="UNKNOWN",
    )

    with pytest.raises(
        ContextInferenceLifecycleError,
        match="unsupported state capability",
    ):
        complete_context_inference(
            inference,
            make_output(),
        )