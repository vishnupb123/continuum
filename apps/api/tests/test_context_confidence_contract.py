import pytest
import torch

from app.services.context.model.confidence_contract import (
    CONFIDENCE_EVIDENCE_DIMENSION,
    ConfidenceContractError,
    build_text_confidence_evidence,
    build_voice_confidence_evidence,
    compute_cross_modal_agreement,
)


def test_confidence_evidence_dimension_is_frozen():
    assert CONFIDENCE_EVIDENCE_DIMENSION == 261


def test_identical_modalities_have_positive_agreement():
    text = torch.randn(256)

    agreement = compute_cross_modal_agreement(
        text,
        text.clone(),
    )

    assert agreement.item() == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_opposite_modalities_have_negative_agreement():
    text = torch.randn(256)

    agreement = compute_cross_modal_agreement(
        text,
        -text,
    )

    assert agreement.item() == pytest.approx(
        -1.0,
        abs=1e-6,
    )


def test_orthogonal_modalities_have_zero_agreement():
    text = torch.zeros(256)
    audio = torch.zeros(256)

    text[0] = 1.0
    audio[1] = 1.0

    agreement = compute_cross_modal_agreement(
        text,
        audio,
    )

    assert agreement.item() == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_batched_agreement_is_supported():
    text = torch.randn((4, 256))
    audio = torch.randn((4, 256))

    agreement = compute_cross_modal_agreement(
        text,
        audio,
    )

    assert agreement.shape == (4,)
    assert torch.isfinite(agreement).all()

    assert torch.all(agreement >= -1.0)
    assert torch.all(agreement <= 1.0)


def test_text_evidence_has_261_dimensions():
    representation = torch.randn(256)

    evidence = build_text_confidence_evidence(
        representation,
        text_quality="GOOD",
    )

    assert evidence.shape == (261,)
    assert torch.isfinite(evidence).all()


def test_text_auxiliary_contract():
    representation = torch.randn(256)

    evidence = build_text_confidence_evidence(
        representation,
        text_quality="DEGRADED",
    )

    auxiliary = evidence[-5:]

    expected = torch.tensor(
        [
            0.5,  # text quality
            0.0,  # audio absent
            0.0,  # not VOICE
            0.0,  # no agreement measurement
            0.0,  # agreement unavailable
        ],
        dtype=evidence.dtype,
    )

    torch.testing.assert_close(
        auxiliary,
        expected,
    )


def test_batched_text_evidence_is_supported():
    representation = torch.randn(
        (3, 256)
    )

    evidence = build_text_confidence_evidence(
        representation,
        text_quality="GOOD",
    )

    assert evidence.shape == (3, 261)


def test_voice_evidence_has_261_dimensions():
    representation = torch.randn(256)
    text = torch.randn(256)
    audio = torch.randn(256)

    evidence = build_voice_confidence_evidence(
        representation,
        projected_text=text,
        projected_audio=audio,
        text_quality="GOOD",
        audio_quality="DEGRADED",
    )

    assert evidence.shape == (261,)
    assert torch.isfinite(evidence).all()


def test_voice_auxiliary_contract():
    representation = torch.randn(256)

    text = torch.zeros(256)
    audio = torch.zeros(256)

    text[0] = 1.0
    audio[0] = 1.0

    evidence = build_voice_confidence_evidence(
        representation,
        projected_text=text,
        projected_audio=audio,
        text_quality="GOOD",
        audio_quality="DEGRADED",
    )

    auxiliary = evidence[-5:]

    assert auxiliary[0].item() == pytest.approx(1.0)
    assert auxiliary[1].item() == pytest.approx(0.5)
    assert auxiliary[2].item() == pytest.approx(1.0)
    assert auxiliary[3].item() == pytest.approx(
        1.0,
        abs=1e-6,
    )
    assert auxiliary[4].item() == pytest.approx(1.0)


def test_batched_voice_evidence_is_supported():
    representation = torch.randn((5, 256))
    text = torch.randn((5, 256))
    audio = torch.randn((5, 256))

    evidence = build_voice_confidence_evidence(
        representation,
        projected_text=text,
        projected_audio=audio,
        text_quality="GOOD",
        audio_quality="GOOD",
    )

    assert evidence.shape == (5, 261)


@pytest.mark.parametrize(
    "quality",
    [
        "UNUSABLE",
        "UNKNOWN",
        "",
    ],
)
def test_invalid_text_quality_is_rejected(
    quality,
):
    representation = torch.randn(256)

    with pytest.raises(
        ConfidenceContractError,
        match="GOOD or DEGRADED",
    ):
        build_text_confidence_evidence(
            representation,
            text_quality=quality,
        )


@pytest.mark.parametrize(
    "quality",
    [
        "UNUSABLE",
        "UNKNOWN",
        "",
    ],
)
def test_invalid_voice_audio_quality_is_rejected(
    quality,
):
    representation = torch.randn(256)
    text = torch.randn(256)
    audio = torch.randn(256)

    with pytest.raises(
        ConfidenceContractError,
        match="GOOD or DEGRADED",
    ):
        build_voice_confidence_evidence(
            representation,
            projected_text=text,
            projected_audio=audio,
            text_quality="GOOD",
            audio_quality=quality,
        )


def test_mismatched_modal_shapes_are_rejected():
    text = torch.randn((2, 256))
    audio = torch.randn((3, 256))

    with pytest.raises(
        ConfidenceContractError,
        match="identical shapes",
    ):
        compute_cross_modal_agreement(
            text,
            audio,
        )


def test_voice_context_and_modality_shapes_must_match():
    representation = torch.randn((2, 256))
    text = torch.randn((3, 256))
    audio = torch.randn((3, 256))

    with pytest.raises(
        ConfidenceContractError,
        match="identical shapes",
    ):
        build_voice_confidence_evidence(
            representation,
            projected_text=text,
            projected_audio=audio,
            text_quality="GOOD",
            audio_quality="GOOD",
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_context_representation_is_rejected(
    invalid_value,
):
    representation = torch.zeros(256)
    representation[10] = invalid_value

    with pytest.raises(
        ConfidenceContractError,
        match="non-finite",
    ):
        build_text_confidence_evidence(
            representation,
            text_quality="GOOD",
        )


def test_integer_context_representation_is_rejected():
    representation = torch.ones(
        256,
        dtype=torch.int64,
    )

    with pytest.raises(
        ConfidenceContractError,
        match="floating-point",
    ):
        build_text_confidence_evidence(
            representation,
            text_quality="GOOD",
        )