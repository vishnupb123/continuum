import torch
from torch import nn

from app.services.context.model.projections import (
    CONTEXT_PROJECTION_DIMENSION,
)


class ObservationBlock(nn.Module):
    """
    Residual transformation in the shared M4 representation space.

    Input:
        (..., 256)

    Output:
        (..., 256)

    Architecture:
        Linear(256 -> 256)
        GELU
        Dropout
        Linear(256 -> 256)
        Residual addition
        LayerNorm(256)
    """

    def __init__(
        self,
        *,
        dimension: int = CONTEXT_PROJECTION_DIMENSION,
        dropout_probability: float = 0.1,
    ) -> None:
        super().__init__()

        if dimension <= 0:
            raise ValueError(
                "Observation dimension must be positive"
            )

        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError(
                "Dropout probability must be in [0, 1)"
            )

        self.dimension = dimension

        self.transform = nn.Sequential(
            nn.Linear(
                dimension,
                dimension,
            ),
            nn.GELU(),
            nn.Dropout(
                p=dropout_probability,
            ),
            nn.Linear(
                dimension,
                dimension,
            ),
        )

        self.normalization = nn.LayerNorm(
            dimension
        )

    def forward(
        self,
        representation: torch.Tensor,
    ) -> torch.Tensor:
        if representation.ndim not in (1, 2):
            raise ValueError(
                "Observation representation must have shape "
                f"({self.dimension},) or "
                f"(batch, {self.dimension})"
            )

        if representation.shape[-1] != self.dimension:
            raise ValueError(
                "Observation representation final dimension "
                f"must be {self.dimension}"
            )

        if not torch.is_floating_point(
            representation
        ):
            raise ValueError(
                "Observation representation must use "
                "a floating-point dtype"
            )

        if not torch.isfinite(
            representation
        ).all():
            raise ValueError(
                "Observation representation contains "
                "non-finite values"
            )

        transformed = self.transform(
            representation
        )

        output = self.normalization(
            representation + transformed
        )

        if not torch.isfinite(output).all():
            raise RuntimeError(
                "Observation block produced "
                "non-finite values"
            )

        return output