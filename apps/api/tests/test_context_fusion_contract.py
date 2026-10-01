import pytest
import torch

from app.services.context.model.fusion_contract import (
    FUSION_GATE_INPUT_DIMENSION,
    FUSION_INTERACTION_DIMENSION,
    FusionContractError,
    build_fusion_gate_input,
    build_fusion_interaction,
    build_quality_features,
    encode_quality,
)


def test_quality_encoding_contract():
    assert encode_quality("GOOD") == 1.0
    assert encode_quality("DEGRADED") == 0.5


@pytest.mark.parametrize(
    "quality",
    [
        "UNUSABLE",
        "UNKNOWN",
        "",
    ],
)
def test_invalid_quality_is_rejected(
    quality,
):
    with pytest.raises(
        FusionContractError,
        match="GOOD or DEGRADED",
    ):
        encode_quality(quality)


def test_single_interaction_has_1024_dimensions():
    text = torch.randn(256)
    audio = torch.randn(256)

    output = build_fusion_interaction(
        text,
        audio,
    )

    assert output.shape == (
        FUSION_INTERACTION_DIMENSION,
    )

    assert (
        FUSION_INTERACTION_DIMENSION
        == 1024
    )


def test_interaction_contains_expected_components():
    text = torch.tensor(
        [1.0] * 256
    )

    audio = torch.tensor(
        [0.25] * 256
    )

    output = build_fusion_interaction(
        text,
        audio,
    )

    text_part = output[0:256]
    audio_part = output[256:512]
    difference_part = output[512:768]
    interaction_part = output[768:1024]

    torch.testing.assert_close(
        text_part,
        text,
    )

    torch.testing.assert_close(
        audio_part,
        audio,
    )

    torch.testing.assert_close(
        difference_part,
        torch.full(
            (256,),
            0.75,
        ),
    )

    torch.testing.assert_close(
        interaction_part,
        torch.full(
            (256,),
            0.25,
        ),
    )


def test_batched_interaction_is_supported():
    text = torch.randn(
        (4, 256)
    )

    audio = torch.randn(
        (4, 256)
    )

    output = build_fusion_interaction(
        text,
        audio,
    )

    assert output.shape == (
        4,
        1024,
    )


def test_quality_features_single_observation():
    reference = torch.randn(
        256,
        dtype=torch.float32,
    )

    quality = build_quality_features(
        text_quality="GOOD",
        audio_quality="DEGRADED",
        reference=reference,
    )

    assert quality.shape == (2,)

    torch.testing.assert_close(
        quality,
        torch.tensor(
            [1.0, 0.5],
            dtype=torch.float32,
        ),
    )


def test_quality_features_are_batched():
    reference = torch.randn(
        (3, 256),
        dtype=torch.float32,
    )

    quality = build_quality_features(
        text_quality="DEGRADED",
        audio_quality="GOOD",
        reference=reference,
    )

    assert quality.shape == (3, 2)

    expected = torch.tensor(
        [
            [0.5, 1.0],
            [0.5, 1.0],
            [0.5, 1.0],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        quality,
        expected,
    )


def test_complete_gate_input_has_1026_dimensions():
    text = torch.randn(256)
    audio = torch.randn(256)

    output = build_fusion_gate_input(
        text,
        audio,
        text_quality="GOOD",
        audio_quality="DEGRADED",
    )

    assert output.shape == (
        FUSION_GATE_INPUT_DIMENSION,
    )

    assert (
        FUSION_GATE_INPUT_DIMENSION
        == 1026
    )


def test_batched_gate_input_has_1026_dimensions():
    text = torch.randn(
        (5, 256)
    )

    audio = torch.randn(
        (5, 256)
    )

    output = build_fusion_gate_input(
        text,
        audio,
        text_quality="GOOD",
        audio_quality="GOOD",
    )

    assert output.shape == (
        5,
        1026,
    )


def test_mismatched_shapes_are_rejected():
    text = torch.randn(
        (2, 256)
    )

    audio = torch.randn(
        (3, 256)
    )

    with pytest.raises(
        FusionContractError,
        match="identical shapes",
    ):
        build_fusion_interaction(
            text,
            audio,
        )


def test_mismatched_dtype_is_rejected():
    text = torch.randn(
        256,
        dtype=torch.float32,
    )

    audio = torch.randn(
        256,
        dtype=torch.float64,
    )

    with pytest.raises(
        FusionContractError,
        match="same dtype",
    ):
        build_fusion_interaction(
            text,
            audio,
        )


@pytest.mark.parametrize(
    "dimension",
    [
        255,
        257,
        768,
    ],
)
def test_wrong_projection_dimension_is_rejected(
    dimension,
):
    text = torch.randn(
        dimension
    )

    audio = torch.randn(
        dimension
    )

    with pytest.raises(
        FusionContractError,
        match="final dimension",
    ):
        build_fusion_interaction(
            text,
            audio,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_representation_is_rejected(
    invalid_value,
):
    text = torch.zeros(256)
    audio = torch.zeros(256)

    audio[10] = invalid_value

    with pytest.raises(
        FusionContractError,
        match="non-finite",
    ):
        build_fusion_interaction(
            text,
            audio,
        )


def test_integer_representation_is_rejected():
    text = torch.ones(
        256,
        dtype=torch.int64,
    )

    audio = torch.ones(
        256,
        dtype=torch.int64,
    )

    with pytest.raises(
        FusionContractError,
        match="floating-point",
    ):
        build_fusion_interaction(
            text,
            audio,
        )