import pytest
import torch

from app.services.context.model.state_contract import (
    STATE_DIMENSION,
    STATE_NAMES,
    StateCapability,
    StateContractError,
    StatePublicationError,
    require_state_publication_capability,
    state_tensor_to_estimate,
    validate_state_tensor,
)


def test_state_schema_is_frozen():
    assert STATE_DIMENSION == 4

    assert STATE_NAMES == (
        "energy",
        "stress",
        "positive_mood",
        "social_connection",
    )


def test_valid_single_state_is_accepted():
    state = torch.tensor(
        [0.2, 0.7, 0.8, 0.4],
        dtype=torch.float32,
    )

    validate_state_tensor(state)


def test_valid_batched_states_are_accepted():
    state = torch.tensor(
        [
            [0.2, 0.7, 0.8, 0.4],
            [0.9, 0.1, 0.6, 0.8],
        ],
        dtype=torch.float32,
    )

    validate_state_tensor(state)


def test_boundary_values_are_accepted():
    state = torch.tensor(
        [0.0, 1.0, 0.0, 1.0],
        dtype=torch.float32,
    )

    validate_state_tensor(state)


@pytest.mark.parametrize(
    "dimension",
    [
        1,
        3,
        5,
        256,
    ],
)
def test_wrong_state_dimension_is_rejected(
    dimension,
):
    state = torch.zeros(
        dimension,
        dtype=torch.float32,
    )

    with pytest.raises(
        StateContractError,
        match="final dimension",
    ):
        validate_state_tensor(state)


def test_more_than_two_dimensions_is_rejected():
    state = torch.zeros(
        (2, 3, 4),
        dtype=torch.float32,
    )

    with pytest.raises(
        StateContractError,
        match="must have shape",
    ):
        validate_state_tensor(state)


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_state_is_rejected(
    invalid_value,
):
    state = torch.tensor(
        [0.2, 0.4, invalid_value, 0.8],
        dtype=torch.float32,
    )

    with pytest.raises(
        StateContractError,
        match="non-finite",
    ):
        validate_state_tensor(state)


@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.01,
        1.01,
        -10.0,
        10.0,
    ],
)
def test_out_of_range_state_is_rejected(
    invalid_value,
):
    state = torch.tensor(
        [0.2, invalid_value, 0.5, 0.8],
        dtype=torch.float32,
    )

    with pytest.raises(
        StateContractError,
        match=r"\[0, 1\]",
    ):
        validate_state_tensor(state)


def test_integer_state_is_rejected():
    state = torch.tensor(
        [0, 1, 0, 1],
        dtype=torch.int64,
    )

    with pytest.raises(
        StateContractError,
        match="floating-point",
    ):
        validate_state_tensor(state)


def test_research_release_cannot_publish_state():
    with pytest.raises(
        StatePublicationError,
        match="not permitted",
    ):
        require_state_publication_capability(
            StateCapability.RESEARCH
        )


def test_validated_release_can_publish_state():
    require_state_publication_capability(
        StateCapability.VALIDATED
    )


def test_research_tensor_cannot_become_publishable_estimate():
    state = torch.tensor(
        [0.2, 0.7, 0.8, 0.4],
        dtype=torch.float32,
    )

    with pytest.raises(
        StatePublicationError,
        match="not permitted",
    ):
        state_tensor_to_estimate(
            state,
            capability=StateCapability.RESEARCH,
        )


def test_validated_tensor_becomes_named_estimate():
    state = torch.tensor(
        [0.2, 0.7, 0.8, 0.4],
        dtype=torch.float32,
    )

    estimate = state_tensor_to_estimate(
        state,
        capability=StateCapability.VALIDATED,
    )

    assert estimate.energy == pytest.approx(0.2)
    assert estimate.stress == pytest.approx(0.7)
    assert estimate.positive_mood == pytest.approx(0.8)
    assert estimate.social_connection == pytest.approx(0.4)


def test_batched_state_cannot_become_single_estimate():
    state = torch.tensor(
        [
            [0.2, 0.7, 0.8, 0.4],
            [0.4, 0.3, 0.6, 0.9],
        ],
        dtype=torch.float32,
    )

    with pytest.raises(
        StateContractError,
        match="exactly one observation",
    ):
        state_tensor_to_estimate(
            state,
            capability=StateCapability.VALIDATED,
        )