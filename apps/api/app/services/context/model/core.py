import torch
from torch import nn

from app.services.context.model.observation import (
    ObservationBlock,
)
from app.services.context.model.projections import (
    CONTEXT_PROJECTION_DIMENSION,
    ContextModalityProjections,
)
from app.services.context.model.fusion import (
    GatedMultimodalFusion,
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
        self.fusion = GatedMultimodalFusion()

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

    def encode_voice(
    self,
    text_embedding: torch.Tensor,
    audio_embedding: torch.Tensor,
    *,
    text_quality: str,
    audio_quality: str,
    ) -> torch.Tensor:
        """
        Complete VOICE current-observation path:

            Et[768] -> text projection  -> T[256]
                                          |
                                          | gated multimodal fusion
                                          |
            Ea[768] -> audio projection -> A[256]
                                          |
                                          v
                                        F[256]
                                          |
                                  observation block
                                          |
                                          v
                                       Rc[256]

        Quality labels influence the learned fusion gate.
        They do not directly modify the M3 embeddings.

        This represents one current observation only.
        No longitudinal memory or trend reasoning occurs here.
        """

        projected_text = self.project_text(
            text_embedding
        )

        projected_audio = self.project_audio(
            audio_embedding
        )

        fused = self.fusion(
            projected_text,
            projected_audio,
            text_quality=text_quality,
            audio_quality=audio_quality,
        )

        return self.encode_shared_representation(
            fused
        )
