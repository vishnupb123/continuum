from dataclasses import dataclass
from enum import Enum

import torch


STATE_DIMENSION = 4

STATE_NAMES = (
    "energy",
    "stress",
    "positive_mood",
    "social_connection",
)


class StateCapability(str, Enum):
    """
    Defines what a released model artifact is permitted to do.

    RESEARCH:
        State computation may be used for training, evaluation,
        experiments, and tests, but inferred states must not be
        published as production user-state estimates.

    VALIDATED:
        The model release has passed the external validation
        process required for production state publication.

    This enum represents release capability, not model quality.
    """

    RESEARCH = "RESEARCH"
    VALIDATED = "VALIDATED"


class StateContractError(ValueError):
    """Raised when state output violates the M4 state contract."""


class StatePublicationError(RuntimeError):
    """
    Raised when a model release attempts to publish inferred
    state without the required production capability.
    """


@dataclass(frozen=True)
class StateEstimate:
    """
    One current-observation state estimate.

    Values are bounded model outputs in [0, 1].

    They are not:
        - diagnoses
        - probabilities of psychiatric conditions
        - longitudinal trends
        - baseline deltas
    """

    energy: float
    stress: float
    positive_mood: float
    social_connection: float


def validate_state_tensor(
    state: torch.Tensor,
) -> None:
    """
    Validate the raw four-dimensional output of a state head.

    Accepted shapes:
        (4,)
        (batch, 4)
    """

    if state.ndim not in (1, 2):
        raise StateContractError(
            "State output must have shape "
            f"({STATE_DIMENSION},) or "
            f"(batch, {STATE_DIMENSION})"
        )

    if state.shape[-1] != STATE_DIMENSION:
        raise StateContractError(
            "State output final dimension "
            f"must be {STATE_DIMENSION}"
        )

    if not torch.is_floating_point(state):
        raise StateContractError(
            "State output must use a floating-point dtype"
        )

    if not torch.isfinite(state).all():
        raise StateContractError(
            "State output contains non-finite values"
        )

    if torch.any(state < 0.0) or torch.any(state > 1.0):
        raise StateContractError(
            "State output values must be within [0, 1]"
        )


def require_state_publication_capability(
    capability: StateCapability,
) -> None:
    """
    Enforce the production publication boundary.

    Research models may compute state outputs internally, but
    those outputs must not be published as user-state estimates.
    """

    if capability is not StateCapability.VALIDATED:
        raise StatePublicationError(
            "Model release is not permitted to publish "
            "inferred user state"
        )


def state_tensor_to_estimate(
    state: torch.Tensor,
    *,
    capability: StateCapability,
) -> StateEstimate:
    """
    Convert one raw state tensor into a publishable state estimate.

    This function deliberately combines validation with the
    release-capability check so callers cannot accidentally use
    this conversion as a publication bypass.
    """

    validate_state_tensor(state)

    if state.ndim != 1:
        raise StateContractError(
            "A publishable StateEstimate requires "
            "exactly one observation"
        )

    require_state_publication_capability(
        capability
    )

    values = state.detach().cpu().tolist()

    return StateEstimate(
        energy=float(values[0]),
        stress=float(values[1]),
        positive_mood=float(values[2]),
        social_connection=float(values[3]),
    )