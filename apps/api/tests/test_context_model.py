import pytest
import torch

from app.services.context.model.context_model import (
    ContextModel,
)
from app.services.context.model.output_contract import (
    publish_context_output,
)
from app.services.context.model.state_contract import (
    StateCapability,
    StatePublicationError,
)


def test_text_model_produces_complete_output():
    model = ContextModel()
    model.eval()

    embedding = torch.randn(768)

    with torch.inference_mode():
        output = model.forward_text(
            embedding,
            text_quality="GOOD",
        )

    assert output.representation.shape == (256,)
    assert output.state.shape == (4,)
    assert output.confidence_score.shape == ()

    assert torch.isfinite(
        output.representation
    ).all()

    assert torch.isfinite(
        output.state
    ).all()

    assert torch.isfinite(
        output.confidence_score
    )

    assert torch.all(output.state >= 0.0)
    assert torch.all(output.state <= 1.0)

    assert output.confidence_score.item() >= 0.0
    assert output.confidence_score.item() <= 1.0


def test_voice_model_produces_complete_output():
    model = ContextModel()
    model.eval()

    text = torch.randn(768)
    audio = torch.randn(768)

    with torch.inference_mode():
        output = model.forward_voice(
            text,
            audio,
            text_quality="GOOD",
            audio_quality="DEGRADED",
        )

    assert output.representation.shape == (256,)
    assert output.state.shape == (4,)
    assert output.confidence_score.shape == ()

    assert torch.isfinite(
        output.representation
    ).all()

    assert torch.isfinite(
        output.state
    ).all()

    assert torch.isfinite(
        output.confidence_score
    )


def test_text_batch_is_supported():
    model = ContextModel()
    model.eval()

    embedding = torch.randn((4, 768))

    with torch.inference_mode():
        output = model.forward_text(
            embedding,
            text_quality="GOOD",
        )

    assert output.representation.shape == (
        4,
        256,
    )

    assert output.state.shape == (
        4,
        4,
    )

    assert output.confidence_score.shape == (4,)


def test_voice_batch_is_supported():
    model = ContextModel()
    model.eval()

    text = torch.randn((3, 768))
    audio = torch.randn((3, 768))

    with torch.inference_mode():
        output = model.forward_voice(
            text,
            audio,
            text_quality="GOOD",
            audio_quality="GOOD",
        )

    assert output.representation.shape == (
        3,
        256,
    )

    assert output.state.shape == (
        3,
        4,
    )

    assert output.confidence_score.shape == (3,)


def test_voice_representation_matches_frozen_core_path():
    torch.manual_seed(42)

    model = ContextModel()
    model.eval()

    text = torch.randn(768)
    audio = torch.randn(768)

    with torch.inference_mode():
        integrated = model.forward_voice(
            text,
            audio,
            text_quality="GOOD",
            audio_quality="DEGRADED",
        )

        core_representation = (
            model.core.encode_voice(
                text,
                audio,
                text_quality="GOOD",
                audio_quality="DEGRADED",
            )
        )

    torch.testing.assert_close(
        integrated.representation,
        core_representation,
    )


def test_text_representation_matches_frozen_core_path():
    torch.manual_seed(42)

    model = ContextModel()
    model.eval()

    text = torch.randn(768)

    with torch.inference_mode():
        integrated = model.forward_text(
            text,
            text_quality="GOOD",
        )

        core_representation = (
            model.core.encode_text(
                text
            )
        )

    torch.testing.assert_close(
        integrated.representation,
        core_representation,
    )


def test_default_model_is_research_capability():
    model = ContextModel()

    assert (
        model.state_capability
        is StateCapability.RESEARCH
    )


def test_default_confidence_is_uncalibrated():
    model = ContextModel()

    assert model.confidence_calibrated is False


def test_research_model_output_cannot_be_published():
    model = ContextModel(
        state_capability=StateCapability.RESEARCH,
    )

    model.eval()

    with torch.inference_mode():
        output = model.forward_text(
            torch.randn(768),
            text_quality="GOOD",
        )

    with pytest.raises(
        StatePublicationError,
    ):
        publish_context_output(
            output
        )


def test_validated_model_output_can_cross_publication_boundary():
    model = ContextModel(
        state_capability=StateCapability.VALIDATED,
    )

    model.eval()

    with torch.inference_mode():
        output = model.forward_text(
            torch.randn(768),
            text_quality="GOOD",
        )

    published = publish_context_output(
        output
    )

    assert 0.0 <= published.state.energy <= 1.0
    assert 0.0 <= published.state.stress <= 1.0

    assert (
        0.0
        <= published.state.positive_mood
        <= 1.0
    )

    assert (
        0.0
        <= published.state.social_connection
        <= 1.0
    )

    assert (
        0.0
        <= published.confidence_score
        <= 1.0
    )


@pytest.mark.parametrize(
    "method",
    [
        "text",
        "voice",
    ],
)
def test_eval_inference_is_deterministic(
    method,
):
    torch.manual_seed(42)

    model = ContextModel()
    model.eval()

    text = torch.randn(768)

    with torch.inference_mode():
        if method == "text":
            first = model.forward_text(
                text,
                text_quality="GOOD",
            )

            second = model.forward_text(
                text,
                text_quality="GOOD",
            )

        else:
            audio = torch.randn(768)

            first = model.forward_voice(
                text,
                audio,
                text_quality="GOOD",
                audio_quality="GOOD",
            )

            second = model.forward_voice(
                text,
                audio,
                text_quality="GOOD",
                audio_quality="GOOD",
            )

    torch.testing.assert_close(
        first.representation,
        second.representation,
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        first.state,
        second.state,
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        first.confidence_score,
        second.confidence_score,
        rtol=0.0,
        atol=0.0,
    )
    