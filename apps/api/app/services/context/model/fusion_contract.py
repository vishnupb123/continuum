import torch

from app.models.feature_constants import (
    FEATURE_QUALITY_DEGRADED,
    FEATURE_QUALITY_GOOD,
)
from app.services.context.model.projections import (
    CONTEXT_PROJECTION_DIMENSION,
)


FUSION_INTERACTION_DIMENSION = (
    CONTEXT_PROJECTION_DIMENSION * 4
)

FUSION_QUALITY_DIMENSION = 2

FUSION_GATE_INPUT_DIMENSION = (
    FUSION_INTERACTION_DIMENSION
    + FUSION_QUALITY_DIMENSION
)


QUALITY_FEATURE_VALUES = {
    FEATURE_QUALITY_GOOD: 1.0,
    FEATURE_QUALITY_DEGRADED: 0.5,
}


class FusionContractError(ValueError):
    """Raised when multimodal fusion inputs violate the M4 contract."""


def encode_quality(
    quality: str,
) -> float:
    """
    Convert an accepted M3 quality label into an ordinal
    M4 fusion feature.

    These values are model features, not calibrated
    probabilities or reliability percentages.
    """

    try:
        return QUALITY_FEATURE_VALUES[
            quality
        ]
    except KeyError as exc:
        raise FusionContractError(
            "Fusion quality must be GOOD or DEGRADED"
        ) from exc


def _validate_projected_modality(
    representation: torch.Tensor,
    *,
    modality: str,
) -> None:
    if representation.ndim not in (1, 2):
        raise FusionContractError(
            f"{modality} representation must have shape "
            f"({CONTEXT_PROJECTION_DIMENSION},) or "
            f"(batch, {CONTEXT_PROJECTION_DIMENSION})"
        )

    if (
        representation.shape[-1]
        != CONTEXT_PROJECTION_DIMENSION
    ):
        raise FusionContractError(
            f"{modality} representation final dimension "
            f"must be {CONTEXT_PROJECTION_DIMENSION}"
        )

    if not torch.is_floating_point(
        representation
    ):
        raise FusionContractError(
            f"{modality} representation must use "
            "a floating-point dtype"
        )

    if not torch.isfinite(
        representation
    ).all():
        raise FusionContractError(
            f"{modality} representation contains "
            "non-finite values"
        )


def build_fusion_interaction(
    text_representation: torch.Tensor,
    audio_representation: torch.Tensor,
) -> torch.Tensor:
    """
    Construct the deterministic cross-modal interaction tensor:

        [T, A, |T-A|, T*A]

    Output dimension:
        4 * 256 = 1024
    """

    _validate_projected_modality(
        text_representation,
        modality="Text",
    )

    _validate_projected_modality(
        audio_representation,
        modality="Audio",
    )

    if (
        text_representation.shape
        != audio_representation.shape
    ):
        raise FusionContractError(
            "Text and audio representations must "
            "have identical shapes"
        )

    if (
        text_representation.dtype
        != audio_representation.dtype
    ):
        raise FusionContractError(
            "Text and audio representations must "
            "use the same dtype"
        )

    difference = torch.abs(
        text_representation
        - audio_representation
    )

    interaction = (
        text_representation
        * audio_representation
    )

    output = torch.cat(
        (
            text_representation,
            audio_representation,
            difference,
            interaction,
        ),
        dim=-1,
    )

    if not torch.isfinite(output).all():
        raise FusionContractError(
            "Fusion interaction contains "
            "non-finite values"
        )

    return output


def build_quality_features(
    *,
    text_quality: str,
    audio_quality: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    Build [text_quality, audio_quality] on the same
    dtype/device as the fusion representation.
    """

    text_value = encode_quality(
        text_quality
    )

    audio_value = encode_quality(
        audio_quality
    )

    values = reference.new_tensor(
        [
            text_value,
            audio_value,
        ]
    )

    if reference.ndim == 1:
        return values

    return values.unsqueeze(0).expand(
        reference.shape[0],
        -1,
    )


def build_fusion_gate_input(
    text_representation: torch.Tensor,
    audio_representation: torch.Tensor,
    *,
    text_quality: str,
    audio_quality: str,
) -> torch.Tensor:
    """
    Construct the complete deterministic gate input:

        [T, A, |T-A|, T*A, q_text, q_audio]

    Final dimension:
        1024 + 2 = 1026
    """

    interaction = build_fusion_interaction(
        text_representation,
        audio_representation,
    )

    quality = build_quality_features(
        text_quality=text_quality,
        audio_quality=audio_quality,
        reference=text_representation,
    )

    output = torch.cat(
        (
            interaction,
            quality,
        ),
        dim=-1,
    )

    if (
        output.shape[-1]
        != FUSION_GATE_INPUT_DIMENSION
    ):
        raise RuntimeError(
            "Fusion gate input has an invalid "
            "final dimension"
        )

    if not torch.isfinite(output).all():
        raise RuntimeError(
            "Fusion gate input contains "
            "non-finite values"
        )

    return output