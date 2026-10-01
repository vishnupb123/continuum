import torch

from app.services.context.model.fusion import (
    GatedMultimodalFusion,
)


def test_single_fusion_produces_256_dimensions():
    fusion = GatedMultimodalFusion()

    text = torch.randn(256)
    audio = torch.randn(256)

    output = fusion(
        text,
        audio,
        text_quality="GOOD",
        audio_quality="GOOD",
    )

    assert output.shape == (256,)
    assert torch.isfinite(output).all()


def test_batched_fusion_is_supported():
    fusion = GatedMultimodalFusion()

    text = torch.randn((4, 256))
    audio = torch.randn((4, 256))

    output = fusion(
        text,
        audio,
        text_quality="GOOD",
        audio_quality="DEGRADED",
    )

    assert output.shape == (4, 256)
    assert torch.isfinite(output).all()


def test_gate_has_256_dimensions():
    fusion = GatedMultimodalFusion()

    text = torch.randn(256)
    audio = torch.randn(256)

    gate = fusion.compute_gate(
        text,
        audio,
        text_quality="GOOD",
        audio_quality="GOOD",
    )

    assert gate.shape == (256,)


def test_gate_is_bounded_between_zero_and_one():
    fusion = GatedMultimodalFusion()

    text = torch.randn((8, 256))
    audio = torch.randn((8, 256))

    gate = fusion.compute_gate(
        text,
        audio,
        text_quality="GOOD",
        audio_quality="DEGRADED",
    )

    assert torch.all(gate >= 0.0)
    assert torch.all(gate <= 1.0)


def test_fusion_equation_matches_gate_definition():
    torch.manual_seed(42)

    fusion = GatedMultimodalFusion()

    text = torch.randn(256)
    audio = torch.randn(256)

    gate = fusion.compute_gate(
        text,
        audio,
        text_quality="GOOD",
        audio_quality="GOOD",
    )

    output = fusion(
        text,
        audio,
        text_quality="GOOD",
        audio_quality="GOOD",
    )

    expected = (
        gate * text
        + (1.0 - gate) * audio
    )

    torch.testing.assert_close(
        output,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_zero_gate_selects_audio():
    fusion = GatedMultimodalFusion()

    with torch.no_grad():
        linear = fusion.gate[0]

        linear.weight.zero_()
        linear.bias.fill_(-100.0)

    text = torch.randn(256)
    audio = torch.randn(256)

    output = fusion(
        text,
        audio,
        text_quality="GOOD",
        audio_quality="GOOD",
    )

    torch.testing.assert_close(
        output,
        audio,
        rtol=0.0,
        atol=0.0,
    )


def test_one_gate_selects_text():
    fusion = GatedMultimodalFusion()

    with torch.no_grad():
        linear = fusion.gate[0]

        linear.weight.zero_()
        linear.bias.fill_(100.0)

    text = torch.randn(256)
    audio = torch.randn(256)

    output = fusion(
        text,
        audio,
        text_quality="GOOD",
        audio_quality="GOOD",
    )

    torch.testing.assert_close(
        output,
        text,
        rtol=0.0,
        atol=0.0,
    )


def test_half_gate_averages_modalities():
    fusion = GatedMultimodalFusion()

    with torch.no_grad():
        linear = fusion.gate[0]

        linear.weight.zero_()
        linear.bias.zero_()

    text = torch.randn(256)
    audio = torch.randn(256)

    output = fusion(
        text,
        audio,
        text_quality="GOOD",
        audio_quality="GOOD",
    )

    expected = (
        0.5 * text
        + 0.5 * audio
    )

    torch.testing.assert_close(
        output,
        expected,
    )


def test_fusion_supports_gradient_flow():
    fusion = GatedMultimodalFusion()

    text = torch.randn(
        (2, 256),
        requires_grad=True,
    )

    audio = torch.randn(
        (2, 256),
        requires_grad=True,
    )

    output = fusion(
        text,
        audio,
        text_quality="GOOD",
        audio_quality="GOOD",
    )

    loss = output.square().mean()
    loss.backward()

    assert text.grad is not None
    assert audio.grad is not None

    assert torch.isfinite(
        text.grad
    ).all()

    assert torch.isfinite(
        audio.grad
    ).all()

    gradients = [
        parameter.grad
        for parameter in fusion.parameters()
        if parameter.requires_grad
    ]

    assert gradients

    assert all(
        gradient is not None
        for gradient in gradients
    )


def test_eval_fusion_is_deterministic():
    torch.manual_seed(42)

    fusion = GatedMultimodalFusion()
    fusion.eval()

    text = torch.randn(256)
    audio = torch.randn(256)

    with torch.inference_mode():
        first = fusion(
            text,
            audio,
            text_quality="GOOD",
            audio_quality="DEGRADED",
        )

        second = fusion(
            text,
            audio,
            text_quality="GOOD",
            audio_quality="DEGRADED",
        )

    torch.testing.assert_close(
        first,
        second,
        rtol=0.0,
        atol=0.0,
    )


def test_quality_signal_reaches_gate_input():
    fusion = GatedMultimodalFusion()

    text = torch.randn(256)
    audio = torch.randn(256)

    good_good = fusion.compute_gate(
        text,
        audio,
        text_quality="GOOD",
        audio_quality="GOOD",
    )

    good_degraded = fusion.compute_gate(
        text,
        audio,
        text_quality="GOOD",
        audio_quality="DEGRADED",
    )

    # With ordinary randomly initialized weights, changing
    # the quality feature should affect the gate.
    assert not torch.equal(
        good_good,
        good_degraded,
    )
    