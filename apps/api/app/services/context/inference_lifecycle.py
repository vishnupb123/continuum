from datetime import datetime, timezone

from app.models.context_constants import (
    CONTEXT_REPRESENTATION_DIMENSION,
    CONTEXT_STATUS_COMPLETED,
    CONTEXT_STATUS_FAILED,
    CONTEXT_STATUS_PENDING,
    CONTEXT_STATUS_PROCESSING,
)
from app.models.context_inference import ContextInference
from app.services.context.model.output_contract import (
    ContextModelOutput,
    publish_context_output,
    validate_context_model_output,
)
from app.services.context.model.state_contract import (
    StateCapability,
)


class ContextInferenceLifecycleError(RuntimeError):
    """Raised when a ContextInference lifecycle transition is invalid."""


def mark_context_inference_processing(
    inference: ContextInference,
) -> None:
    if inference.status != CONTEXT_STATUS_PENDING:
        raise ContextInferenceLifecycleError(
            "Only PENDING context inference may enter PROCESSING"
        )

    _clear_outputs(inference)

    inference.status = CONTEXT_STATUS_PROCESSING
    inference.error_message = None
    inference.completed_at = None


def complete_context_inference(
    inference: ContextInference,
    output: ContextModelOutput,
) -> None:
    if inference.status != CONTEXT_STATUS_PROCESSING:
        raise ContextInferenceLifecycleError(
            "Only PROCESSING context inference may complete"
        )

    capability = _parse_capability(
        inference.state_capability
    )

    if output.state_capability is not capability:
        raise ContextInferenceLifecycleError(
            "Model output capability does not match "
            "context inference capability"
        )

    validate_context_model_output(output)

    representation = (
        output.representation.detach().cpu()
    )

    if representation.ndim != 1:
        raise ContextInferenceLifecycleError(
            "Persistence requires one context observation"
        )

    inference.representation_dimension = (
        CONTEXT_REPRESENTATION_DIMENSION
    )

    inference.representation = (
        representation.tolist()
    )

    if capability is StateCapability.RESEARCH:
        inference.energy = None
        inference.stress = None
        inference.positive_mood = None
        inference.social_connection = None
        inference.confidence_score = None
        inference.confidence_calibrated = False

    else:
        published = publish_context_output(
            output
        )

        inference.energy = published.state.energy
        inference.stress = published.state.stress
        inference.positive_mood = (
            published.state.positive_mood
        )
        inference.social_connection = (
            published.state.social_connection
        )
        inference.confidence_score = (
            published.confidence_score
        )
        inference.confidence_calibrated = (
            published.confidence_calibrated
        )

    inference.status = CONTEXT_STATUS_COMPLETED
    inference.error_message = None
    inference.completed_at = datetime.now(
        timezone.utc
    )


def fail_context_inference(
    inference: ContextInference,
    *,
    error_message: str,
) -> None:
    if inference.status == CONTEXT_STATUS_COMPLETED:
        raise ContextInferenceLifecycleError(
            "COMPLETED context inference is terminal"
        )

    if inference.status not in (
        CONTEXT_STATUS_PENDING,
        CONTEXT_STATUS_PROCESSING,
    ):
        raise ContextInferenceLifecycleError(
            "Only PENDING or PROCESSING context inference may fail"
        )

    if (
        not isinstance(error_message, str)
        or not error_message.strip()
    ):
        raise ContextInferenceLifecycleError(
            "Failure error message must be non-blank"
        )

    _clear_outputs(inference)

    inference.status = CONTEXT_STATUS_FAILED
    inference.error_message = error_message.strip()
    inference.completed_at = None


def reset_failed_context_inference_for_retry(
    inference: ContextInference,
) -> None:
    if inference.status != CONTEXT_STATUS_FAILED:
        raise ContextInferenceLifecycleError(
            "Only FAILED context inference may be retried"
        )

    _clear_outputs(inference)

    inference.status = CONTEXT_STATUS_PENDING
    inference.error_message = None
    inference.completed_at = None


def _parse_capability(
    value: str,
) -> StateCapability:
    try:
        return StateCapability(value)
    except ValueError as exc:
        raise ContextInferenceLifecycleError(
            "Context inference has unsupported state capability"
        ) from exc


def _clear_outputs(
    inference: ContextInference,
) -> None:
    inference.representation_dimension = None
    inference.representation = None

    inference.energy = None
    inference.stress = None
    inference.positive_mood = None
    inference.social_connection = None

    inference.confidence_score = None
    inference.confidence_calibrated = False