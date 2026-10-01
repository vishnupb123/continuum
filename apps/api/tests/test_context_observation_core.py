import torch

from app.services.context.model.core import (
    ContextObservationCore,
)


def test_text_path_produces_256_representation():
    core = ContextObservationCore()
    core.eval()

    embedding = torch.randn(
        768,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        representation = core.encode_text(
            embedding
        )

    assert representation.shape == (256,)
    assert representation.dtype == torch.float32
    assert torch.isfinite(
        representation
    ).all()


def test_batched_text_path_is_supported():
    core = ContextObservationCore()
    core.eval()

    embeddings = torch.randn(
        (4, 768),
        dtype=torch.float32,
    )

    with torch.inference_mode():
        representation = core.encode_text(
            embeddings
        )

    assert representation.shape == (4, 256)

    assert torch.isfinite(
        representation
    ).all()


def test_encode_text_matches_explicit_composition():
    torch.manual_seed(42)

    core = ContextObservationCore(
        dropout_probability=0.0,
    )

    core.eval()

    embedding = torch.randn(
        768,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        direct = core.encode_text(
            embedding
        )

        projected = core.project_text(
            embedding
        )

        explicit = (
            core.encode_shared_representation(
                projected
            )
        )

    torch.testing.assert_close(
        direct,
        explicit,
        rtol=0.0,
        atol=0.0,
    )


def test_audio_projection_does_not_implicitly_encode_observation():
    core = ContextObservationCore()
    core.eval()

    embedding = torch.randn(
        768,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        projected_audio = core.project_audio(
            embedding
        )

    assert projected_audio.shape == (256,)

    # M4.4 deliberately exposes only the projected audio
    # representation here. VOICE must go through M4.5 fusion
    # before becoming Rc.
    assert torch.isfinite(
        projected_audio
    ).all()


def test_text_path_supports_gradient_flow():
    core = ContextObservationCore(
        dropout_probability=0.0,
    )

    embedding = torch.randn(
        (2, 768),
        dtype=torch.float32,
        requires_grad=True,
    )

    representation = core.encode_text(
        embedding
    )

    loss = representation.square().mean()
    loss.backward()

    assert embedding.grad is not None

    assert torch.isfinite(
        embedding.grad
    ).all()

    text_gradients = [
        parameter.grad
        for parameter
        in core.projections.text.parameters()
        if parameter.requires_grad
    ]

    observation_gradients = [
        parameter.grad
        for parameter
        in core.observation.parameters()
        if parameter.requires_grad
    ]

    assert text_gradients
    assert observation_gradients

    assert all(
        gradient is not None
        for gradient in text_gradients
    )

    assert all(
        gradient is not None
        for gradient in observation_gradients
    )


def test_text_inference_is_deterministic_in_eval_mode():
    torch.manual_seed(42)

    core = ContextObservationCore()
    core.eval()

    embedding = torch.randn(
        768,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        first = core.encode_text(
            embedding
        )

        second = core.encode_text(
            embedding
        )

    torch.testing.assert_close(
        first,
        second,
        rtol=0.0,
        atol=0.0,
    )


def test_text_and_audio_projection_parameters_remain_independent():
    core = ContextObservationCore()

    text_parameters = {
        id(parameter)
        for parameter
        in core.projections.text.parameters()
    }

    audio_parameters = {
        id(parameter)
        for parameter
        in core.projections.audio.parameters()
    }

    assert text_parameters
    assert audio_parameters

    assert text_parameters.isdisjoint(
        audio_parameters
    )

def test_voice_path_produces_256_representation():
    core = ContextObservationCore()
    core.eval()

    text_embedding = torch.randn(
        768,
        dtype=torch.float32,
    )

    audio_embedding = torch.randn(
        768,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        representation = core.encode_voice(
            text_embedding,
            audio_embedding,
            text_quality="GOOD",
            audio_quality="GOOD",
        )

    assert representation.shape == (256,)
    assert representation.dtype == torch.float32
    assert torch.isfinite(
        representation
    ).all()


def test_batched_voice_path_is_supported():
    core = ContextObservationCore()
    core.eval()

    text_embeddings = torch.randn(
        (4, 768),
        dtype=torch.float32,
    )

    audio_embeddings = torch.randn(
        (4, 768),
        dtype=torch.float32,
    )

    with torch.inference_mode():
        representation = core.encode_voice(
            text_embeddings,
            audio_embeddings,
            text_quality="GOOD",
            audio_quality="DEGRADED",
        )

    assert representation.shape == (4, 256)

    assert torch.isfinite(
        representation
    ).all()


def test_voice_path_matches_explicit_composition():
    torch.manual_seed(42)

    core = ContextObservationCore(
        dropout_probability=0.0,
    )
    core.eval()

    text_embedding = torch.randn(
        768,
        dtype=torch.float32,
    )

    audio_embedding = torch.randn(
        768,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        direct = core.encode_voice(
            text_embedding,
            audio_embedding,
            text_quality="GOOD",
            audio_quality="DEGRADED",
        )

        projected_text = core.project_text(
            text_embedding
        )

        projected_audio = core.project_audio(
            audio_embedding
        )

        fused = core.fusion(
            projected_text,
            projected_audio,
            text_quality="GOOD",
            audio_quality="DEGRADED",
        )

        explicit = (
            core.encode_shared_representation(
                fused
            )
        )

    torch.testing.assert_close(
        direct,
        explicit,
        rtol=0.0,
        atol=0.0,
    )


def test_voice_path_supports_gradient_flow():
    core = ContextObservationCore(
        dropout_probability=0.0,
    )

    text_embedding = torch.randn(
        (2, 768),
        dtype=torch.float32,
        requires_grad=True,
    )

    audio_embedding = torch.randn(
        (2, 768),
        dtype=torch.float32,
        requires_grad=True,
    )

    representation = core.encode_voice(
        text_embedding,
        audio_embedding,
        text_quality="GOOD",
        audio_quality="GOOD",
    )

    loss = representation.square().mean()
    loss.backward()

    assert text_embedding.grad is not None
    assert audio_embedding.grad is not None

    assert torch.isfinite(
        text_embedding.grad
    ).all()

    assert torch.isfinite(
        audio_embedding.grad
    ).all()

    fusion_gradients = [
        parameter.grad
        for parameter
        in core.fusion.parameters()
        if parameter.requires_grad
    ]

    assert fusion_gradients

    assert all(
        gradient is not None
        for gradient in fusion_gradients
    )


def test_voice_inference_is_deterministic_in_eval_mode():
    torch.manual_seed(42)

    core = ContextObservationCore()
    core.eval()

    text_embedding = torch.randn(
        768,
        dtype=torch.float32,
    )

    audio_embedding = torch.randn(
        768,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        first = core.encode_voice(
            text_embedding,
            audio_embedding,
            text_quality="GOOD",
            audio_quality="DEGRADED",
        )

        second = core.encode_voice(
            text_embedding,
            audio_embedding,
            text_quality="GOOD",
            audio_quality="DEGRADED",
        )

    torch.testing.assert_close(
        first,
        second,
        rtol=0.0,
        atol=0.0,
    )
