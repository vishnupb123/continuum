import pytest
import torch

from app.services.context.model.state_head import (
    STATE_HIDDEN_DIMENSION,
    StateHead,
)


def test_state_head_architecture_dimensions():
    head = StateHead()

    assert head.input_dimension == 256
    assert head.hidden_dimension == 128
    assert STATE_HIDDEN_DIMENSION == 128

    assert head.network[0].in_features == 256
    assert head.network[0].out_features == 128

    assert head.network[2].in_features == 128
    assert head.network[2].out_features == 4


def test_single_representation_produces_four_states():
    head = StateHead()

    representation = torch.randn(
        256,
        dtype=torch.float32,
    )

    state = head(
        representation
    )

    assert state.shape == (4,)
    assert state.dtype == torch.float32
    assert torch.isfinite(state).all()


def test_batched_representation_produces_four_states():
    head = StateHead()

    representation = torch.randn(
        (5, 256),
        dtype=torch.float32,
    )

    state = head(
        representation
    )

    assert state.shape == (5, 4)
    assert torch.isfinite(state).all()


def test_state_outputs_are_bounded():
    head = StateHead()

    representation = torch.randn(
        (16, 256),
        dtype=torch.float32,
    )

    state = head(
        representation
    )

    assert torch.all(state >= 0.0)
    assert torch.all(state <= 1.0)


def test_zero_logits_produce_half_state():
    head = StateHead()

    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()

    representation = torch.randn(
        256,
        dtype=torch.float32,
    )

    state = head(
        representation
    )

    expected = torch.full(
        (4,),
        0.5,
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        state,
        expected,
    )


def test_state_head_supports_gradient_flow():
    head = StateHead()

    representation = torch.randn(
        (2, 256),
        dtype=torch.float32,
        requires_grad=True,
    )

    state = head(
        representation
    )

    loss = state.square().mean()
    loss.backward()

    assert representation.grad is not None
    assert torch.isfinite(
        representation.grad
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


def test_eval_state_head_is_deterministic():
    torch.manual_seed(42)

    head = StateHead()
    head.eval()

    representation = torch.randn(
        256,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        first = head(representation)
        second = head(representation)

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
        255,
        257,
        768,
    ],
)
def test_wrong_input_dimension_is_rejected(
    dimension,
):
    head = StateHead()

    representation = torch.randn(
        dimension,
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="final dimension",
    ):
        head(representation)


def test_more_than_two_dimensions_is_rejected():
    head = StateHead()

    representation = torch.zeros(
        (2, 3, 256),
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="must have shape",
    ):
        head(representation)


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_input_is_rejected(
    invalid_value,
):
    head = StateHead()

    representation = torch.zeros(
        256,
        dtype=torch.float32,
    )

    representation[20] = invalid_value

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        head(representation)


def test_integer_input_is_rejected():
    head = StateHead()

    representation = torch.ones(
        256,
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match="floating-point",
    ):
        head(representation)


@pytest.mark.parametrize(
    ("input_dimension", "hidden_dimension"),
    [
        (0, 128),
        (-1, 128),
        (256, 0),
        (256, -1),
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
        StateHead(
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
        )