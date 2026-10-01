import torch
from torch import nn

from app.services.context.model.confidence_contract import (
    CONFIDENCE_EVIDENCE_DIMENSION,
)


CONFIDENCE_HIDDEN_DIMENSION = 128


class ConfidenceHead(nn.Module):
    """
    Maps confidence evidence to one bounded reliability score.

    The output is an uncalibrated model score in [0, 1].

    It is NOT:
        - probability that the inferred state is correct
        - probability of stress, depression, or another condition
        - automatically calibrated confidence

    Calibration status is tracked separately from this module.
    """

    def __init__(
        self,
        *,
        input_dimension: int = CONFIDENCE_EVIDENCE_DIMENSION,
        hidden_dimension: int = CONFIDENCE_HIDDEN_DIMENSION,
    ) -> None:
        super().__init__()

        if input_dimension <= 0:
            raise ValueError(
                "Confidence head input dimension must be positive"
            )

        if hidden_dimension <= 0:
            raise ValueError(
                "Confidence head hidden dimension must be positive"
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
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        evidence: torch.Tensor,
    ) -> torch.Tensor:
        if evidence.ndim not in (1, 2):
            raise ValueError(
                "Confidence evidence must have shape "
                f"({self.input_dimension},) or "
                f"(batch, {self.input_dimension})"
            )

        if (
            evidence.shape[-1]
            != self.input_dimension
        ):
            raise ValueError(
                "Confidence evidence final dimension "
                f"must be {self.input_dimension}"
            )

        if not torch.is_floating_point(
            evidence
        ):
            raise ValueError(
                "Confidence evidence must use "
                "a floating-point dtype"
            )

        if not torch.isfinite(
            evidence
        ).all():
            raise ValueError(
                "Confidence evidence contains non-finite values"
            )

        confidence = self.network(
            evidence
        ).squeeze(-1)

        if not torch.isfinite(
            confidence
        ).all():
            raise RuntimeError(
                "Confidence head produced non-finite values"
            )

        if (
            torch.any(confidence < 0.0)
            or torch.any(confidence > 1.0)
        ):
            raise RuntimeError(
                "Confidence score must be within [0, 1]"
            )

        return confidence