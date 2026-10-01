from dataclasses import FrozenInstanceError
from uuid import uuid4

import numpy as np
import pytest

from app.services.context.input_contract import (
    ContextModelInput,
    FeatureProvenance,
)


def make_text_provenance():
    return FeatureProvenance(
        encoder_name=(
            "sentence-transformers/"
            "all-mpnet-base-v2"
        ),
        encoder_version="text-encoder-v1",
        encoder_revision="test-text-revision",
    )


def make_audio_provenance():
    return FeatureProvenance(
        encoder_name="microsoft/wavlm-base-plus",
        encoder_version="audio-encoder-v1",
        encoder_revision="test-audio-revision",
    )


def test_text_input_represents_true_audio_absence():
    model_input = ContextModelInput(
        journal_id=uuid4(),
        feature_set_id=uuid4(),
        entry_type="TEXT",
        feature_pipeline_version="m3-v1",
        source_hash="text-source-hash",
        text_embedding=np.ones(
            768,
            dtype=np.float32,
        ),
        text_quality="GOOD",
        text_provenance=make_text_provenance(),
        audio_embedding=None,
        audio_quality=None,
        audio_provenance=None,
    )

    assert model_input.entry_type == "TEXT"

    assert (
        model_input.text_embedding.shape
        == (768,)
    )

    assert (
        model_input.text_embedding.dtype
        == np.float32
    )

    assert model_input.audio_embedding is None
    assert model_input.audio_quality is None
    assert model_input.audio_provenance is None


def test_voice_input_can_represent_both_modalities():
    model_input = ContextModelInput(
        journal_id=uuid4(),
        feature_set_id=uuid4(),
        entry_type="VOICE",
        feature_pipeline_version="m3-v1",
        source_hash="voice-source-hash",
        text_embedding=np.ones(
            768,
            dtype=np.float32,
        ),
        text_quality="GOOD",
        text_provenance=make_text_provenance(),
        audio_embedding=np.ones(
            768,
            dtype=np.float32,
        ),
        audio_quality="DEGRADED",
        audio_provenance=make_audio_provenance(),
    )

    assert model_input.entry_type == "VOICE"

    assert (
        model_input.text_embedding.shape
        == (768,)
    )

    assert model_input.audio_embedding is not None

    assert (
        model_input.audio_embedding.shape
        == (768,)
    )

    assert model_input.text_quality == "GOOD"
    assert model_input.audio_quality == "DEGRADED"


def test_context_input_preserves_m3_lineage():
    journal_id = uuid4()
    feature_set_id = uuid4()

    model_input = ContextModelInput(
        journal_id=journal_id,
        feature_set_id=feature_set_id,
        entry_type="TEXT",
        feature_pipeline_version="m3-v7",
        source_hash="authoritative-source-hash",
        text_embedding=np.zeros(
            768,
            dtype=np.float32,
        ),
        text_quality="GOOD",
        text_provenance=make_text_provenance(),
        audio_embedding=None,
        audio_quality=None,
        audio_provenance=None,
    )

    assert model_input.journal_id == journal_id
    assert model_input.feature_set_id == feature_set_id

    assert (
        model_input.feature_pipeline_version
        == "m3-v7"
    )

    assert (
        model_input.source_hash
        == "authoritative-source-hash"
    )


def test_context_input_is_frozen():
    model_input = ContextModelInput(
        journal_id=uuid4(),
        feature_set_id=uuid4(),
        entry_type="TEXT",
        feature_pipeline_version="m3-v1",
        source_hash="immutable-source",
        text_embedding=np.zeros(
            768,
            dtype=np.float32,
        ),
        text_quality="GOOD",
        text_provenance=make_text_provenance(),
        audio_embedding=None,
        audio_quality=None,
        audio_provenance=None,
    )

    with pytest.raises(FrozenInstanceError):
        model_input.entry_type = "VOICE"