import torch
from torch import nn

from app.services.context.model.confidence_contract import (
    build_text_confidence_evidence,
    build_voice_confidence_evidence,
)
from app.services.context.model.confidence_head import (
    ConfidenceHead,
)
from app.services.context.model.core import (
    ContextObservationCore,
)
from app.services.context.model.output_contract import (
    ContextModelOutput,
    validate_context_model_output,
)
from app.services.context.model.state_contract import (
    StateCapability,
)
from app.services.context.model.state_head import (
    StateHead,
)


class ContextModel(nn.Module):
    """
    Complete M4 current-observation model.

    This module combines:

        M3 embeddings
            -> M4 representation core
            -> state head
            -> confidence evidence
            -> confidence head

    It performs current-observation inference only.

    It does not perform:
        - longitudinal reasoning
        - baseline comparison
        - trend detection
        - memory retrieval
        - diagnosis

    State publication remains controlled separately through
    StateCapability and the output publication boundary.
    """

    def __init__(
        self,
        *,
        state_capability: StateCapability = (
            StateCapability.RESEARCH
        ),
        confidence_calibrated: bool = False,
    ) -> None:
        super().__init__()

        if not isinstance(
            state_capability,
            StateCapability,
        ):
            raise ValueError(
                "state_capability must be a StateCapability"
            )

        if not isinstance(
            confidence_calibrated,
            bool,
        ):
            raise ValueError(
                "confidence_calibrated must be boolean"
            )

        self.state_capability = state_capability
        self.confidence_calibrated = (
            confidence_calibrated
        )

        self.core = ContextObservationCore()
        self.state_head = StateHead()
        self.confidence_head = ConfidenceHead()

    def forward_text(
        self,
        text_embedding: torch.Tensor,
        *,
        text_quality: str,
    ) -> ContextModelOutput:
        representation = self.core.encode_text(
            text_embedding
        )

        state = self.state_head(
            representation
        )

        confidence_evidence = (
            build_text_confidence_evidence(
                representation,
                text_quality=text_quality,
            )
        )

        confidence_score = self.confidence_head(
            confidence_evidence
        )

        output = ContextModelOutput(
            representation=representation,
            state=state,
            confidence_score=confidence_score,
            confidence_calibrated=(
                self.confidence_calibrated
            ),
            state_capability=(
                self.state_capability
            ),
        )

        validate_context_model_output(
            output
        )

        return output

    def forward_voice(
        self,
        text_embedding: torch.Tensor,
        audio_embedding: torch.Tensor,
        *,
        text_quality: str,
        audio_quality: str,
    ) -> ContextModelOutput:
        projected_text = (
            self.core.project_text(
                text_embedding
            )
        )

        projected_audio = (
            self.core.project_audio(
                audio_embedding
            )
        )

        fused = self.core.fusion(
            projected_text,
            projected_audio,
            text_quality=text_quality,
            audio_quality=audio_quality,
        )

        representation = (
            self.core.encode_shared_representation(
                fused
            )
        )

        state = self.state_head(
            representation
        )

        confidence_evidence = (
            build_voice_confidence_evidence(
                representation,
                projected_text=projected_text,
                projected_audio=projected_audio,
                text_quality=text_quality,
                audio_quality=audio_quality,
            )
        )

        confidence_score = self.confidence_head(
            confidence_evidence
        )

        output = ContextModelOutput(
            representation=representation,
            state=state,
            confidence_score=confidence_score,
            confidence_calibrated=(
                self.confidence_calibrated
            ),
            state_capability=(
                self.state_capability
            ),
        )

        validate_context_model_output(
            output
        )

        return output