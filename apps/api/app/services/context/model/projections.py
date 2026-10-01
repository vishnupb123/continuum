import torch
from torch import nn


M3_EMBEDDING_DIMENSION = 768
CONTEXT_PROJECTION_DIMENSION = 256


class ModalityProjection(nn.Module):
    """
    Learned projection from one M3 embedding space into the
    shared M4 representation space.

    Architecture:
        Linear(768 -> 256)
        GELU
        LayerNorm(256)

    This module does not perform multimodal fusion.
    """

    def __init__(
        self,
        *,
        input_dimension: int = M3_EMBEDDING_DIMENSION,
        output_dimension: int = CONTEXT_PROJECTION_DIMENSION,
    ) -> None:
        super().__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.projection = nn.Sequential(
            nn.Linear(
                input_dimension,
                output_dimension,
            ),
            nn.GELU(),
            nn.LayerNorm(
                output_dimension,
            ),
        )

    def forward(
        self,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        if embedding.ndim not in (1, 2):
            raise ValueError(
                "Modality embedding must have shape "
                f"({self.input_dimension},) or "
                f"(batch, {self.input_dimension})"
            )

        if embedding.shape[-1] != self.input_dimension:
            raise ValueError(
                "Modality embedding final dimension "
                f"must be {self.input_dimension}"
            )

        if not torch.is_floating_point(embedding):
            raise ValueError(
                "Modality embedding must use a floating-point dtype"
            )

        if not torch.isfinite(embedding).all():
            raise ValueError(
                "Modality embedding contains non-finite values"
            )

        output = self.projection(
            embedding
        )

        if not torch.isfinite(output).all():
            raise RuntimeError(
                "Projection produced non-finite values"
            )

        return output


class ContextModalityProjections(nn.Module):
    """
    Independent learned projections for semantic text evidence
    and acoustic audio evidence.

    Text and audio intentionally do not share parameters even
    though both M3 embeddings are 768-dimensional.
    """

    def __init__(self) -> None:
        super().__init__()

        self.text = ModalityProjection()
        self.audio = ModalityProjection()

    def project_text(
        self,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        return self.text(embedding)

    def project_audio(
        self,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        return self.audio(embedding)