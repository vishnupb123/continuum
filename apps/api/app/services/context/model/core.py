import torch
from torch import nn

from app.services.context.model.observation import (
    ObservationBlock,
)
from app.services.context.model.projections import (
    CONTEXT_PROJECTION_DIMENSION,
    ContextModalityProjections,
)


class ContextObservationCore(nn.Module):
    """
    Current-observation representation core for Context-MoDE.

    Responsibilities:
        - project M3 modality embeddings into the shared
          256-dimensional M4 space
        - transform a shared representation into Rc

    This module does NOT own:
        - multimodal fusion
        - state prediction
        - confidence prediction
        - longitudinal memory
        - baseline/trend reasoning
    """

    def __init__(
        self,
        *,
        dropout_probability: float = 0.1,
    ) -> None:
        super().__init__()

        self.projections = ContextModalityProjections()

        self.observation = ObservationBlock(
            dimension=CONTEXT_PROJECTION_DIMENSION,
            dropout_probability=dropout_probability,
        )

    def project_text(
        self,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        return self.projections.project_text(
            embedding
        )

    def project_audio(
        self,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        return self.projections.project_audio(
            embedding
        )

    def encode_shared_representation(
        self,
        representation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert one already-projected/shared representation
        into the current-observation representation Rc.

        In M4.4 the TEXT path can feed its projected text
        representation directly here.

        M4.5 will feed the fused VOICE representation here.
        """

        return self.observation(
            representation
        )

    def encode_text(
        self,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Complete TEXT-only current-observation path:

            Et[768]
              -> text projection
              -> T[256]
              -> observation block
              -> Rc[256]
        """

        projected = self.project_text(
            embedding
        )

        return self.encode_shared_representation(
            projected
        )