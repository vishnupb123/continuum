import pytest
import torch

from app.services.context.model.confidence_head import (
    CONFIDENCE_HIDDEN_DIMENSION,
    ConfidenceHead,
)


def test_confidence_head_architecture_dimensions():
    head = ConfidenceHead()

    assert head.input_dimension == 261
    assert head.hidden_dimension == 128
    assert CONFIDENCE_HIDDEN_DIMENSION == 128

    assert head.network[0].in_features == 261
    assert head.network[0].out_features == 128

    assert head.network[2].in_features == 128
    assert head.network[2].out_features == 1


def test_single_evidence_produces_scalar_confidence():
    head = ConfidenceHead()

    evidence = torch.randn(
        261,
        dtype=torch.float32,
    )

    confidence = head(evidence)

    assert confidence.shape == ()
    assert confidence.dtype == torch.float32
    assert torch.isfinite(confidence)


def test_batched_evidence_produces_one_score_per_observation():
    head = ConfidenceHead()

    evidence = torch.randn(
        (5, 261),
        dtype=torch.float32,
    )

    confidence = head(evidence)

    assert confidence.shape == (5,)
    assert torch.isfinite(confidence).all()


def test_confidence_is_bounded():
    head = ConfidenceHead()

    evidence = torch.randn(
        (16, 261),
        dtype=torch.float32,
    )

    confidence = head(evidence)

    assert torch.all(confidence >= 0.0)
    assert torch.all(confidence <= 1.0)


def test_zero_logits_produce_half_score():
    head = ConfidenceHead()

    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()

    evidence = torch.randn(
        261,
        dtype=torch.float32,
    )

    confidence = head(evidence)

    assert confidence.item() == pytest.approx(
        0.5
    )


def test_confidence_head_supports_gradient_flow():
    head = ConfidenceHead()

    evidence = torch.randn(
        (2, 261),
        dtype=torch.float32,
        requires_grad=True,
    )

    confidence = head(evidence)

    loss = confidence.square().mean()
    loss.backward()

    assert evidence.grad is not None
    assert torch.isfinite(
        evidence.grad
    ).all()

    gradients = [
        parameter.grad
        for parameter in head.parameters()
        if parameter.requires_grad
    ]

    assert gradients

    assert all(
        gradient is not None
        for gradient in gradients
    )


def test_eval_confidence_head_is_deterministic():
    torch.manual_seed(42)

    head = ConfidenceHead()
    head.eval()

    evidence = torch.randn(
        261,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        first = head(evidence)
        second = head(evidence)

    torch.testing.assert_close(
        first,
        second,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "dimension",
    [
        1,
        128,
        256,
        260,
        262,
    ],
)
def test_wrong_input_dimension_is_rejected(
    dimension,
):
    head = ConfidenceHead()

    evidence = torch.randn(
        dimension,
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="final dimension",
    ):
        head(evidence)


def test_more_than_two_dimensions_is_rejected():
    head = ConfidenceHead()

    evidence = torch.zeros(
        (2, 3, 261),
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="must have shape",
    ):
        head(evidence)


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_evidence_is_rejected(
    invalid_value,
):
    head = ConfidenceHead()

    evidence = torch.zeros(
        261,
        dtype=torch.float32,
    )

    evidence[20] = invalid_value

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        head(evidence)


def test_integer_evidence_is_rejected():
    head = ConfidenceHead()

    evidence = torch.ones(
        261,
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match="floating-point",
    ):
        head(evidence)


@pytest.mark.parametrize(
    ("input_dimension", "hidden_dimension"),
    [
        (0, 128),
        (-1, 128),
        (261, 0),
        (261, -1),
    ],
)
def test_invalid_architecture_dimension_is_rejected(
    input_dimension,
    hidden_dimension,
):
    with pytest.raises(
        ValueError,
        match="dimension must be positive",
    ):
        ConfidenceHead(
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
        )