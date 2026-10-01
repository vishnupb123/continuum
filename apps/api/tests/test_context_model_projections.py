import pytest
import torch

from app.services.context.model.projections import (
    ContextModalityProjections,
    ModalityProjection,
)


def test_single_embedding_projects_768_to_256():
    projection = ModalityProjection()

    embedding = torch.ones(
        768,
        dtype=torch.float32,
    )

    output = projection(embedding)

    assert output.shape == (256,)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_batched_embeddings_project_to_256():
    projection = ModalityProjection()

    embeddings = torch.ones(
        (4, 768),
        dtype=torch.float32,
    )

    output = projection(embeddings)

    assert output.shape == (4, 256)
    assert torch.isfinite(output).all()


def test_projection_supports_gradient_flow():
    projection = ModalityProjection()

    embedding = torch.randn(
        (2, 768),
        dtype=torch.float32,
        requires_grad=True,
    )

    output = projection(embedding)

    loss = output.square().mean()
    loss.backward()

    assert embedding.grad is not None

    assert torch.isfinite(
        embedding.grad
    ).all()

    parameter_gradients = [
        parameter.grad
        for parameter in projection.parameters()
        if parameter.requires_grad
    ]

    assert parameter_gradients

    assert all(
        gradient is not None
        for gradient in parameter_gradients
    )


def test_text_and_audio_parameters_are_independent():
    projections = ContextModalityProjections()

    text_parameter_ids = {
        id(parameter)
        for parameter in projections.text.parameters()
    }

    audio_parameter_ids = {
        id(parameter)
        for parameter in projections.audio.parameters()
    }

    assert text_parameter_ids
    assert audio_parameter_ids

    assert text_parameter_ids.isdisjoint(
        audio_parameter_ids
    )


def test_eval_projection_is_deterministic():
    torch.manual_seed(42)

    projection = ModalityProjection()
    projection.eval()

    embedding = torch.randn(
        768,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        first = projection(embedding)
        second = projection(embedding)

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
        256,
        767,
        769,
    ],
)
def test_wrong_final_dimension_is_rejected(
    dimension,
):
    projection = ModalityProjection()

    embedding = torch.zeros(
        dimension,
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="final dimension",
    ):
        projection(embedding)


def test_more_than_two_dimensions_is_rejected():
    projection = ModalityProjection()

    embedding = torch.zeros(
        (2, 3, 768),
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="must have shape",
    ):
        projection(embedding)


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
    projection = ModalityProjection()

    embedding = torch.zeros(
        768,
        dtype=torch.float32,
    )

    embedding[100] = invalid_value

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        projection(embedding)


def test_integer_embedding_is_rejected():
    projection = ModalityProjection()

    embedding = torch.ones(
        768,
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match="floating-point",
    ):
        projection(embedding)


def test_text_and_audio_projection_paths_work():
    projections = ContextModalityProjections()

    text_embedding = torch.randn(
        768,
        dtype=torch.float32,
    )

    audio_embedding = torch.randn(
        768,
        dtype=torch.float32,
    )

    text_output = projections.project_text(
        text_embedding
    )

    audio_output = projections.project_audio(
        audio_embedding
    )

    assert text_output.shape == (256,)
    assert audio_output.shape == (256,)

    assert torch.isfinite(
        text_output
    ).all()

    assert torch.isfinite(
        audio_output
    ).all()