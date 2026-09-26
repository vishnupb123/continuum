import math

import pytest

from app.services.features.text_encoder.base import (
    TextEncodingResult,
)
from app.services.features.text_encoder.mock import (
    MOCK_TEXT_ENCODER_DIMENSION,
    MOCK_TEXT_ENCODER_NAME,
    MOCK_TEXT_ENCODER_REVISION,
    MOCK_TEXT_ENCODER_VERSION,
    MockTextEncoder,
)


def test_mock_encoder_is_deterministic():
    encoder = MockTextEncoder()

    first = encoder.encode(
        "Today was productive."
    )

    second = encoder.encode(
        "Today was productive."
    )

    assert first == second


def test_different_text_produces_different_embedding():
    encoder = MockTextEncoder()

    first = encoder.encode(
        "Today was productive."
    )

    second = encoder.encode(
        "Today was exhausting."
    )

    assert (
        first.embedding
        != second.embedding
    )


def test_mock_encoder_contract():
    encoder = MockTextEncoder()

    result = encoder.encode(
        "Today was productive."
    )

    assert (
        result.dimension
        == MOCK_TEXT_ENCODER_DIMENSION
    )

    assert (
        len(result.embedding)
        == MOCK_TEXT_ENCODER_DIMENSION
    )

    assert (
        result.encoder_name
        == MOCK_TEXT_ENCODER_NAME
    )

    assert (
        result.encoder_version
        == MOCK_TEXT_ENCODER_VERSION
    )

    assert (
        result.encoder_revision
        == MOCK_TEXT_ENCODER_REVISION
    )

    assert result.normalized is True


def test_mock_embedding_has_unit_norm():
    encoder = MockTextEncoder()

    result = encoder.encode(
        "Today was productive."
    )

    magnitude = math.sqrt(
        sum(
            value * value
            for value in result.embedding
        )
    )

    assert magnitude == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_empty_text_is_rejected():
    encoder = MockTextEncoder()

    with pytest.raises(
        ValueError,
        match="text must not be empty",
    ):
        encoder.encode("")


def test_non_string_input_is_rejected():
    encoder = MockTextEncoder()

    with pytest.raises(
        TypeError,
        match="text must be a string",
    ):
        encoder.encode(None)


def test_result_rejects_dimension_mismatch():
    with pytest.raises(
        ValueError,
        match="embedding length must match dimension",
    ):
        TextEncodingResult(
            embedding=(0.1, 0.2),
            dimension=3,
            encoder_name="test",
            encoder_version="test-v1",
            encoder_revision="test-revision",
            normalized=False,
        )


def test_result_rejects_invalid_dimension():
    with pytest.raises(
        ValueError,
        match="dimension must be greater than zero",
    ):
        TextEncodingResult(
            embedding=(),
            dimension=0,
            encoder_name="test",
            encoder_version="test-v1",
            encoder_revision="test-revision",
            normalized=False,
        )