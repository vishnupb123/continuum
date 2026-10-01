import torch
import torch.nn.functional as F

from app.services.context.model.projections import (
    CONTEXT_PROJECTION_DIMENSION,
)
from app.services.context.model.quality_evidence import (
    QualityEvidenceError,
    encode_quality_evidence,
)


CONFIDENCE_AUXILIARY_DIMENSION = 5

CONFIDENCE_EVIDENCE_DIMENSION = (
    CONTEXT_PROJECTION_DIMENSION
    + CONFIDENCE_AUXILIARY_DIMENSION
)


class ConfidenceContractError(ValueError):
    """
    Raised when confidence evidence violates the M4 contract.
    """


def _validate_representation(
    representation: torch.Tensor,
    *,
    name: str,
) -> None:
    if representation.ndim not in (1, 2):
        raise ConfidenceContractError(
            f"{name} must have shape "
            f"({CONTEXT_PROJECTION_DIMENSION},) or "
            f"(batch, {CONTEXT_PROJECTION_DIMENSION})"
        )

    if (
        representation.shape[-1]
        != CONTEXT_PROJECTION_DIMENSION
    ):
        raise ConfidenceContractError(
            f"{name} final dimension must be "
            f"{CONTEXT_PROJECTION_DIMENSION}"
        )

    if not torch.is_floating_point(
        representation
    ):
        raise ConfidenceContractError(
            f"{name} must use a floating-point dtype"
        )

    if not torch.isfinite(
        representation
    ).all():
        raise ConfidenceContractError(
            f"{name} contains non-finite values"
        )


def _encode_quality(
    quality: str,
) -> float:
    try:
        return encode_quality_evidence(
            quality
        )
    except QualityEvidenceError as exc:
        raise ConfidenceContractError(
            "Confidence quality must be GOOD or DEGRADED"
        ) from exc


def compute_cross_modal_agreement(
    text_representation: torch.Tensor,
    audio_representation: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine agreement between projected text and audio.

    Output:
        scalar for a single observation
        (batch,) for batched observations

    Agreement remains in its natural [-1, 1] range.

    It is evidence supplied to the confidence model, not a
    confidence score itself.
    """

    _validate_representation(
        text_representation,
        name="Text representation",
    )

    _validate_representation(
        audio_representation,
        name="Audio representation",
    )

    if (
        text_representation.shape
        != audio_representation.shape
    ):
        raise ConfidenceContractError(
            "Text and audio representations must "
            "have identical shapes"
        )

    if (
        text_representation.dtype
        != audio_representation.dtype
    ):
        raise ConfidenceContractError(
            "Text and audio representations must "
            "use the same dtype"
        )

    if (
        text_representation.device
        != audio_representation.device
    ):
        raise ConfidenceContractError(
            "Text and audio representations must "
            "use the same device"
        )

    agreement = F.cosine_similarity(
        text_representation,
        audio_representation,
        dim=-1,
        eps=1e-8,
    )

    if not torch.isfinite(
        agreement
    ).all():
        raise ConfidenceContractError(
            "Cross-modal agreement is non-finite"
        )

    agreement_tolerance = 1e-6

    if (
       torch.any(
        agreement < (-1.0 - agreement_tolerance)
       )
    or torch.any(
        agreement > (1.0 + agreement_tolerance)
       )
   ):
     raise ConfidenceContractError(
        "Cross-modal agreement must be within "
        "[-1, 1] within numerical tolerance"
    )

    return agreement


def build_text_confidence_evidence(
    representation: torch.Tensor,
    *,
    text_quality: str,
) -> torch.Tensor:
    """
    Build confidence evidence for a TEXT observation.

    Auxiliary fields:
        text_quality
        audio_quality=0
        is_voice=0
        agreement=0
        agreement_available=0

    The zeros represent modality absence, not poor audio
    quality or zero measured agreement.
    """

    _validate_representation(
        representation,
        name="Context representation",
    )

    text_quality_value = _encode_quality(
        text_quality
    )

    auxiliary = representation.new_tensor(
        [
            text_quality_value,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    if representation.ndim == 2:
        auxiliary = (
            auxiliary
            .unsqueeze(0)
            .expand(
                representation.shape[0],
                -1,
            )
        )

    evidence = torch.cat(
        (
            representation,
            auxiliary,
        ),
        dim=-1,
    )

    _validate_confidence_evidence(
        evidence
    )

    return evidence


def build_voice_confidence_evidence(
    representation: torch.Tensor,
    *,
    projected_text: torch.Tensor,
    projected_audio: torch.Tensor,
    text_quality: str,
    audio_quality: str,
) -> torch.Tensor:
    """
    Build confidence evidence for a VOICE observation.

    Auxiliary fields:
        text_quality
        audio_quality
        is_voice=1
        cosine(T, A)
        agreement_available=1
    """

    _validate_representation(
        representation,
        name="Context representation",
    )

    _validate_representation(
        projected_text,
        name="Text representation",
    )

    _validate_representation(
        projected_audio,
        name="Audio representation",
    )

    if (
        representation.shape
        != projected_text.shape
        or representation.shape
        != projected_audio.shape
    ):
        raise ConfidenceContractError(
            "VOICE confidence representations must "
            "have identical shapes"
        )

    if (
        representation.dtype
        != projected_text.dtype
        or representation.dtype
        != projected_audio.dtype
    ):
        raise ConfidenceContractError(
            "VOICE confidence representations must "
            "use the same dtype"
        )

    if (
        representation.device
        != projected_text.device
        or representation.device
        != projected_audio.device
    ):
        raise ConfidenceContractError(
            "VOICE confidence representations must "
            "use the same device"
        )

    text_quality_value = _encode_quality(
        text_quality
    )

    audio_quality_value = _encode_quality(
        audio_quality
    )

    agreement = compute_cross_modal_agreement(
        projected_text,
        projected_audio,
    )

    if representation.ndim == 1:
        auxiliary = torch.stack(
            (
                representation.new_tensor(
                    text_quality_value
                ),
                representation.new_tensor(
                    audio_quality_value
                ),
                representation.new_tensor(1.0),
                agreement,
                representation.new_tensor(1.0),
            )
        )

    else:
        batch_size = representation.shape[0]

        text_quality_feature = (
            representation.new_full(
                (batch_size,),
                text_quality_value,
            )
        )

        audio_quality_feature = (
            representation.new_full(
                (batch_size,),
                audio_quality_value,
            )
        )

        is_voice = representation.new_ones(
            batch_size
        )

        agreement_available = (
            representation.new_ones(
                batch_size
            )
        )

        auxiliary = torch.stack(
            (
                text_quality_feature,
                audio_quality_feature,
                is_voice,
                agreement,
                agreement_available,
            ),
            dim=-1,
        )

    evidence = torch.cat(
        (
            representation,
            auxiliary,
        ),
        dim=-1,
    )

    _validate_confidence_evidence(
        evidence
    )

    return evidence


def _validate_confidence_evidence(
    evidence: torch.Tensor,
) -> None:
    if (
        evidence.shape[-1]
        != CONFIDENCE_EVIDENCE_DIMENSION
    ):
        raise RuntimeError(
            "Confidence evidence has an invalid "
            "final dimension"
        )

    if not torch.isfinite(
        evidence
    ).all():
        raise RuntimeError(
            "Confidence evidence contains "
            "non-finite values"
        )