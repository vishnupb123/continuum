import numpy as np
import pytest

from app.services.context.input_contract import (
    FeatureProvenance,
)
from app.services.context.input_validation import (
    ContextInputValidationError,
    validate_embedding,
    validate_feature_quality,
    validate_provenance,
)


def make_embedding():
    return np.linspace(
        0.0,
        1.0,
        768,
        dtype=np.float32,
    )


@pytest.mark.parametrize(
    "quality",
    [
        "GOOD",
        "DEGRADED",
    ],
)
def test_good_and_degraded_quality_are_accepted(
    quality,
):
    validate_feature_quality(
        quality=quality,
        modality="Text",
    )


@pytest.mark.parametrize(
    "quality",
    [
        "UNUSABLE",
        "UNKNOWN",
        "",
    ],
)
def test_invalid_quality_is_rejected(
    quality,
):
    with pytest.raises(
        ContextInputValidationError,
        match="quality is not usable",
    ):
        validate_feature_quality(
            quality=quality,
            modality="Text",
        )


def test_valid_embedding_returns_float32_copy():
    original = np.linspace(
        0.0,
        1.0,
        768,
        dtype=np.float64,
    )

    validated = validate_embedding(
        embedding=original,
        declared_dimension=768,
        modality="Text",
    )

    assert validated.shape == (768,)
    assert validated.dtype == np.float32

    assert (
        np.shares_memory(
            original,
            validated,
        )
        is False
    )


@pytest.mark.parametrize(
    "declared_dimension",
    [
        None,
        0,
        256,
        767,
        769,
    ],
)
def test_wrong_declared_dimension_is_rejected(
    declared_dimension,
):
    with pytest.raises(
        ContextInputValidationError,
        match="declared dimension",
    ):
        validate_embedding(
            embedding=make_embedding(),
            declared_dimension=declared_dimension,
            modality="Text",
        )


@pytest.mark.parametrize(
    "actual_dimension",
    [
        1,
        256,
        767,
        769,
    ],
)
def test_wrong_actual_dimension_is_rejected(
    actual_dimension,
):
    embedding = np.zeros(
        actual_dimension,
        dtype=np.float32,
    )

    with pytest.raises(
        ContextInputValidationError,
        match="embedding shape",
    ):
        validate_embedding(
            embedding=embedding,
            declared_dimension=768,
            modality="Audio",
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_non_finite_embedding_is_rejected(
    invalid_value,
):
    embedding = make_embedding()
    embedding[100] = invalid_value

    with pytest.raises(
        ContextInputValidationError,
        match="non-finite",
    ):
        validate_embedding(
            embedding=embedding,
            declared_dimension=768,
            modality="Text",
        )


def test_valid_provenance_is_accepted():
    provenance = FeatureProvenance(
        encoder_name="test-encoder",
        encoder_version="encoder-v1",
        encoder_revision="revision-1",
    )

    validate_provenance(
        provenance=provenance,
        modality="Text",
    )


@pytest.mark.parametrize(
    ("encoder_name", "encoder_version"),
    [
        ("", "encoder-v1"),
        ("   ", "encoder-v1"),
        ("test-encoder", ""),
        ("test-encoder", "   "),
    ],
)
def test_blank_provenance_is_rejected(
    encoder_name,
    encoder_version,
):
    provenance = FeatureProvenance(
        encoder_name=encoder_name,
        encoder_version=encoder_version,
        encoder_revision=None,
    )

    with pytest.raises(
        ContextInputValidationError,
        match="encoder",
    ):
        validate_provenance(
            provenance=provenance,
            modality="Text",
        )