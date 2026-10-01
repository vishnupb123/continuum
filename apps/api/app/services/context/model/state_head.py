import torch
from torch import nn

from app.services.context.model.projections import (
    CONTEXT_PROJECTION_DIMENSION,
)
from app.services.context.model.state_contract import (
    STATE_DIMENSION,
    validate_state_tensor,
)


STATE_HIDDEN_DIMENSION = 128


class StateHead(nn.Module):
    """
    Maps the current-observation representation Rc to four
    bounded state outputs.

    Output order:
        energy
        stress
        positive_mood
        social_connection

    This module only computes raw model outputs.

    Its existence does NOT imply that a model release is
    validated or permitted to publish inferred user state.
    Publication capability is enforced separately by the
    state contract.
    """

    def __init__(
        self,
        *,
        input_dimension: int = CONTEXT_PROJECTION_DIMENSION,
        hidden_dimension: int = STATE_HIDDEN_DIMENSION,
    ) -> None:
        super().__init__()

        if input_dimension <= 0:
            raise ValueError(
                "State head input dimension must be positive"
            )

        if hidden_dimension <= 0:
            raise ValueError(
                "State head hidden dimension must be positive"
            )

        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension

        self.network = nn.Sequential(
            nn.Linear(
                input_dimension,
                hidden_dimension,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dimension,
                STATE_DIMENSION,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        representation: torch.Tensor,
    ) -> torch.Tensor:
        if representation.ndim not in (1, 2):
            raise ValueError(
                "State head input must have shape "
                f"({self.input_dimension},) or "
                f"(batch, {self.input_dimension})"
            )

        if (
            representation.shape[-1]
            != self.input_dimension
        ):
            raise ValueError(
                "State head input final dimension "
                f"must be {self.input_dimension}"
            )

        if not torch.is_floating_point(
            representation
        ):
            raise ValueError(
                "State head input must use "
                "a floating-point dtype"
            )

        if not torch.isfinite(
            representation
        ).all():
            raise ValueError(
                "State head input contains non-finite values"
            )

        state = self.network(
            representation
        )

        validate_state_tensor(
            state
        )

        return state