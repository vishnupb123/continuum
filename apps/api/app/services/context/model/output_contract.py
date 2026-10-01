from dataclasses import dataclass

import torch

from app.services.context.model.state_contract import (
    STATE_DIMENSION,
    StateCapability,
    StateEstimate,
    require_state_publication_capability,
    validate_state_tensor,
)


class ContextOutputContractError(ValueError):
    """Raised when a Context-MoDE output violates its contract."""


@dataclass(frozen=True)
class ContextModelOutput:
    """
    Raw output of one Context-MoDE observation.

    This object may exist in research mode.

    Its state values are model outputs, not diagnoses,
    longitudinal changes, baseline deltas, or validated
    psychological measurements.
    """

    representation: torch.Tensor
    state: torch.Tensor
    confidence_score: torch.Tensor
    confidence_calibrated: bool
    state_capability: StateCapability


@dataclass(frozen=True)
class PublishableContextOutput:
    """
    Named output permitted to cross the publication boundary.

    Construction requires VALIDATED state capability.
    """

    representation: torch.Tensor
    state: StateEstimate
    confidence_score: float
    confidence_calibrated: bool


def validate_context_model_output(
    output: ContextModelOutput,
) -> None:
    _validate_representation(
        output.representation
    )

    validate_state_tensor(
        output.state
    )

    _validate_confidence_score(
        output.confidence_score
    )

    if (
        output.representation.ndim
        != output.state.ndim
    ):
        raise ContextOutputContractError(
            "Representation and state must use "
            "matching observation dimensions"
        )

    if output.representation.ndim == 2:
        if (
            output.representation.shape[0]
            != output.state.shape[0]
        ):
            raise ContextOutputContractError(
                "Representation and state batch sizes "
                "must match"
            )

        if output.confidence_score.ndim != 1:
            raise ContextOutputContractError(
                "Batched confidence must have shape (batch,)"
            )

        if (
            output.confidence_score.shape[0]
            != output.representation.shape[0]
        ):
            raise ContextOutputContractError(
                "Confidence batch size must match "
                "representation batch size"
            )

    else:
        if output.confidence_score.ndim != 0:
            raise ContextOutputContractError(
                "Single-observation confidence must be scalar"
            )

    if not isinstance(
        output.confidence_calibrated,
        bool,
    ):
        raise ContextOutputContractError(
            "confidence_calibrated must be boolean"
        )

    if not isinstance(
        output.state_capability,
        StateCapability,
    ):
        raise ContextOutputContractError(
            "state_capability must be a StateCapability"
        )


def publish_context_output(
    output: ContextModelOutput,
) -> PublishableContextOutput:
    """
    Convert one validated model output into a publishable form.

    Research state outputs cannot cross this boundary.
    """

    validate_context_model_output(
        output
    )

    if output.representation.ndim != 1:
        raise ContextOutputContractError(
            "Publication requires one observation"
        )

    require_state_publication_capability(
        output.state_capability
    )

    state_values = output.state.detach().cpu().tolist()

    state = StateEstimate(
        energy=float(state_values[0]),
        stress=float(state_values[1]),
        positive_mood=float(state_values[2]),
        social_connection=float(state_values[3]),
    )

    return PublishableContextOutput(
        representation=(
            output.representation
            .detach()
            .clone()
        ),
        state=state,
        confidence_score=float(
            output.confidence_score
            .detach()
            .cpu()
            .item()
        ),
        confidence_calibrated=(
            output.confidence_calibrated
        ),
    )


def _validate_representation(
    representation: torch.Tensor,
) -> None:
    if representation.ndim not in (1, 2):
        raise ContextOutputContractError(
            "Context representation must have shape "
            "(256,) or (batch, 256)"
        )

    if representation.shape[-1] != 256:
        raise ContextOutputContractError(
            "Context representation final dimension "
            "must be 256"
        )

    if not torch.is_floating_point(
        representation
    ):
        raise ContextOutputContractError(
            "Context representation must use "
            "a floating-point dtype"
        )

    if not torch.isfinite(
        representation
    ).all():
        raise ContextOutputContractError(
            "Context representation contains "
            "non-finite values"
        )


def _validate_confidence_score(
    confidence: torch.Tensor,
) -> None:
    if confidence.ndim not in (0, 1):
        raise ContextOutputContractError(
            "Confidence must be scalar or shape (batch,)"
        )

    if not torch.is_floating_point(
        confidence
    ):
        raise ContextOutputContractError(
            "Confidence must use a floating-point dtype"
        )

    if not torch.isfinite(
        confidence
    ).all():
        raise ContextOutputContractError(
            "Confidence contains non-finite values"
        )

    if (
        torch.any(confidence < 0.0)
        or torch.any(confidence > 1.0)
    ):
        raise ContextOutputContractError(
            "Confidence must be within [0, 1]"
        )