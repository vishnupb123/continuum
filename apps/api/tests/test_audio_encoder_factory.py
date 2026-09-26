import pytest

from app.services.features.audio_encoder import (
    AUDIO_ENCODER_DIMENSION,
)
from app.services.features.audio_encoder_factory import (
    _normalize_optional_string,
)
from app.services.features.mock_audio_encoder import (
    MockAudioEncoder,
)


def test_normalize_optional_string_none():
    assert (
        _normalize_optional_string(None)
        is None
    )


def test_normalize_optional_string_blank():
    assert (
        _normalize_optional_string("   ")
        is None
    )


def test_normalize_optional_string_strips_value():
    assert (
        _normalize_optional_string(
            "  revision-123  "
        )
        == "revision-123"
    )


def test_mock_encoder_implements_expected_contract():
    encoder = MockAudioEncoder()

    assert (
        encoder.embedding_dimension
        == AUDIO_ENCODER_DIMENSION
    )

    assert (
        encoder.encoder_name
        == "mock-audio-encoder"
    )