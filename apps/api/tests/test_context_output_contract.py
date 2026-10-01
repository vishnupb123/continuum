import pytest
import torch

from app.services.context.model.output_contract import (
    ContextModelOutput,
    ContextOutputContractError,
    publish_context_output,
    validate_context_model_output,
)
from app.services.context.model.state_contract import (
    StateCapability,
    StatePublicationError,
)


def make_output(
    *,
    capability=StateCapability.RESEARCH,
    calibrated=False,
):
    return ContextModelOutput(
        representation=torch.randn(256),
        state=torch.tensor(
            [0.2, 0.7, 0.6, 0.4],
            dtype=torch.float32,
        ),
        confidence_score=torch.tensor(
            0.8,
            dtype=torch.float32,
        ),
        confidence_calibrated=calibrated,
        state_capability=capability,
    )


def test_valid_research_output_is_structurally_valid():
    output = make_output()

    validate_context_model_output(
        output
    )


def test_research_output_cannot_be_published():
    output = make_output(
        capability=StateCapability.RESEARCH,
    )

    with pytest.raises(
        StatePublicationError,
    ):
        publish_context_output(
            output
        )


def test_validated_output_can_be_published():
    output = make_output(
        capability=StateCapability.VALIDATED,
    )

    published = publish_context_output(
        output
    )

    assert published.state.energy == pytest.approx(0.2)
    assert published.state.stress == pytest.approx(0.7)
    assert (
        published.state.positive_mood
        == pytest.approx(0.6)
    )
    assert (
        published.state.social_connection
        == pytest.approx(0.4)
    )

    assert published.confidence_score == pytest.approx(
        0.8
    )


def test_uncalibrated_status_is_preserved():
    output = make_output(
        capability=StateCapability.VALIDATED,
        calibrated=False,
    )

    published = publish_context_output(
        output
    )

    assert published.confidence_calibrated is False


def test_calibrated_status_is_preserved():
    output = make_output(
        capability=StateCapability.VALIDATED,
        calibrated=True,
    )

    published = publish_context_output(
        output
    )

    assert published.confidence_calibrated is True


def test_invalid_representation_dimension_is_rejected():
    output = make_output()

    output = ContextModelOutput(
        representation=torch.randn(255),
        state=output.state,
        confidence_score=output.confidence_score,
        confidence_calibrated=False,
        state_capability=StateCapability.RESEARCH,
    )

    with pytest.raises(
        ContextOutputContractError,
        match="256",
    ):
        validate_context_model_output(
            output
        )


def test_out_of_range_state_is_rejected():
    output = make_output()

    output = ContextModelOutput(
        representation=output.representation,
        state=torch.tensor(
            [0.2, 1.1, 0.6, 0.4],
            dtype=torch.float32,
        ),
        confidence_score=output.confidence_score,
        confidence_calibrated=False,
        state_capability=StateCapability.RESEARCH,
    )

    with pytest.raises(
        ValueError,
        match=r"\[0, 1\]",
    ):
        validate_context_model_output(
            output
        )


@pytest.mark.parametrize(
    "confidence",
    [
        -0.01,
        1.01,
    ],
)
def test_out_of_range_confidence_is_rejected(
    confidence,
):
    output = make_output()

    output = ContextModelOutput(
        representation=output.representation,
        state=output.state,
        confidence_score=torch.tensor(
            confidence,
            dtype=torch.float32,
        ),
        confidence_calibrated=False,
        state_capability=StateCapability.RESEARCH,
    )

    with pytest.raises(
        ContextOutputContractError,
        match=r"\[0, 1\]",
    ):
        validate_context_model_output(
            output
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_confidence_is_rejected(
    invalid_value,
):
    output = make_output()

    output = ContextModelOutput(
        representation=output.representation,
        state=output.state,
        confidence_score=torch.tensor(
            invalid_value,
            dtype=torch.float32,
        ),
        confidence_calibrated=False,
        state_capability=StateCapability.RESEARCH,
    )

    with pytest.raises(
        ContextOutputContractError,
        match="non-finite",
    ):
        validate_context_model_output(
            output
        )


def test_batch_sizes_must_match():
    output = ContextModelOutput(
        representation=torch.randn((3, 256)),
        state=torch.rand((2, 4)),
        confidence_score=torch.rand(3),
        confidence_calibrated=False,
        state_capability=StateCapability.RESEARCH,
    )

    with pytest.raises(
        ContextOutputContractError,
        match="batch sizes",
    ):
        validate_context_model_output(
            output
        )


def test_confidence_batch_size_must_match():
    output = ContextModelOutput(
        representation=torch.randn((3, 256)),
        state=torch.rand((3, 4)),
        confidence_score=torch.rand(2),
        confidence_calibrated=False,
        state_capability=StateCapability.RESEARCH,
    )

    with pytest.raises(
        ContextOutputContractError,
        match="Confidence batch size",
    ):
        validate_context_model_output(
            output
        )


def test_batched_output_cannot_be_published():
    output = ContextModelOutput(
        representation=torch.randn((2, 256)),
        state=torch.rand((2, 4)),
        confidence_score=torch.rand(2),
        confidence_calibrated=False,
        state_capability=StateCapability.VALIDATED,
    )

    with pytest.raises(
        ContextOutputContractError,
        match="one observation",
    ):
        publish_context_output(
            output
        )