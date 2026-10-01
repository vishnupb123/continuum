import torch
from torch import nn

from app.services.context.model.fusion_contract import (
    FUSION_GATE_INPUT_DIMENSION,
    build_fusion_gate_input,
)
from app.services.context.model.projections import (
    CONTEXT_PROJECTION_DIMENSION,
)


class GatedMultimodalFusion(nn.Module):
    """
    Learned feature-wise fusion of projected text and audio evidence.

    Inputs:
        text_representation:  (..., 256)
        audio_representation: (..., 256)

    Gate:
        g = sigmoid(W * gate_input + b)

    Fusion:
        F = g * T + (1 - g) * A

    The gate is a latent learned routing mechanism.
    It is not a calibrated confidence score.
    """

    def __init__(self) -> None:
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(
                FUSION_GATE_INPUT_DIMENSION,
                CONTEXT_PROJECTION_DIMENSION,
            ),
            nn.Sigmoid(),
        )

    def compute_gate(
        self,
        text_representation: torch.Tensor,
        audio_representation: torch.Tensor,
        *,
        text_quality: str,
        audio_quality: str,
    ) -> torch.Tensor:
        gate_input = build_fusion_gate_input(
            text_representation,
            audio_representation,
            text_quality=text_quality,
            audio_quality=audio_quality,
        )

        gate = self.gate(
            gate_input
        )

        if (
            gate.shape[-1]
            != CONTEXT_PROJECTION_DIMENSION
        ):
            raise RuntimeError(
                "Fusion gate has an invalid final dimension"
            )

        if not torch.isfinite(gate).all():
            raise RuntimeError(
                "Fusion gate produced non-finite values"
            )

        if (
            torch.any(gate < 0.0)
            or torch.any(gate > 1.0)
        ):
            raise RuntimeError(
                "Fusion gate produced values outside [0, 1]"
            )

        return gate

    def forward(
        self,
        text_representation: torch.Tensor,
        audio_representation: torch.Tensor,
        *,
        text_quality: str,
        audio_quality: str,
    ) -> torch.Tensor:
        gate = self.compute_gate(
            text_representation,
            audio_representation,
            text_quality=text_quality,
            audio_quality=audio_quality,
        )

        fused = (
            gate * text_representation
            + (1.0 - gate) * audio_representation
        )

        if not torch.isfinite(fused).all():
            raise RuntimeError(
                "Multimodal fusion produced non-finite values"
            )

        return fused