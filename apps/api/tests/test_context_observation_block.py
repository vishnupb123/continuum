import pytest
import torch

from app.services.context.model.observation import (
    ObservationBlock,
)


def test_single_representation_preserves_dimension():
    block = ObservationBlock()

    representation = torch.randn(
        256,
        dtype=torch.float32,
    )

    output = block(representation)

    assert output.shape == (256,)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_batched_representations_preserve_dimension():
    block = ObservationBlock()

    representations = torch.randn(
        (4, 256),
        dtype=torch.float32,
    )

    output = block(representations)

    assert output.shape == (4, 256)
    assert torch.isfinite(output).all()


def test_observation_block_supports_gradient_flow():
    block = ObservationBlock()

    representation = torch.randn(
        (2, 256),
        dtype=torch.float32,
        requires_grad=True,
    )

    output = block(representation)

    loss = output.square().mean()
    loss.backward()

    assert representation.grad is not None
    assert torch.isfinite(
        representation.grad
    ).all()

    gradients = [
        parameter.grad
        for parameter in block.parameters()
        if parameter.requires_grad
    ]

    assert gradients

    assert all(
        gradient is not None
        for gradient in gradients
    )


def test_eval_mode_is_deterministic():
    torch.manual_seed(42)

    block = ObservationBlock(
        dropout_probability=0.1,
    )

    block.eval()

    representation = torch.randn(
        256,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        first = block(representation)
        second = block(representation)

    torch.testing.assert_close(
        first,
        second,
        rtol=0.0,
        atol=0.0,
    )


def test_training_mode_uses_dropout():
    torch.manual_seed(42)

    block = ObservationBlock(
        dropout_probability=0.5,
    )

    block.train()

    representation = torch.randn(
        (8, 256),
        dtype=torch.float32,
    )

    first = block(representation)
    second = block(representation)

    assert not torch.equal(
        first,
        second,
    )


@pytest.mark.parametrize(
    "dimension",
    [
        1,
        128,
        255,
        257,
    ],
)
def test_wrong_final_dimension_is_rejected(
    dimension,
):
    block = ObservationBlock()

    representation = torch.zeros(
        dimension,
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="final dimension",
    ):
        block(representation)


def test_more_than_two_dimensions_is_rejected():
    block = ObservationBlock()

    representation = torch.zeros(
        (2, 3, 256),
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="must have shape",
    ):
        block(representation)


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
    block = ObservationBlock()

    representation = torch.zeros(
        256,
        dtype=torch.float32,
    )

    representation[100] = invalid_value

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        block(representation)


def test_integer_input_is_rejected():
    block = ObservationBlock()

    representation = torch.ones(
        256,
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match="floating-point",
    ):
        block(representation)


@pytest.mark.parametrize(
    "probability",
    [
        -0.1,
        1.0,
        1.5,
    ],
)
def test_invalid_dropout_probability_is_rejected(
    probability,
):
    with pytest.raises(
        ValueError,
        match="Dropout probability",
    ):
        ObservationBlock(
            dropout_probability=probability,
        )


def test_residual_path_is_structurally_active():
    block = ObservationBlock(
        dropout_probability=0.0,
    )

    block.eval()

    # Remove the learned transform contribution.
    with torch.no_grad():
        for parameter in (
            block.transform.parameters()
        ):
            parameter.zero_()

    representation = torch.randn(
        256,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        output = block(representation)

        expected = block.normalization(
            representation
        )

    torch.testing.assert_close(
        output,
        expected,
    )
    