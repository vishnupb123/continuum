import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from app.services.features.text_encoder.chunking import (
    TokenChunk,
)
from app.services.features.text_encoder.mpnet import (
    EXPECTED_MPNet_DIMENSION,
    MPNetTextEncoder,
    TEXT_ENCODER_VERSION,
)


def make_mock_model():
    model = MagicMock()

    model.get_sentence_embedding_dimension.return_value = (
        EXPECTED_MPNet_DIMENSION
    )

    model.max_seq_length = 384

    tokenizer = MagicMock()
    tokenizer.num_special_tokens_to_add.return_value = 2

    model.tokenizer = tokenizer

    return model


@patch(
    "app.services.features.text_encoder.mpnet."
    "SentenceTransformer"
)
def test_mpnet_encoder_initializes_token_budget(
    sentence_transformer_cls,
):
    model = make_mock_model()

    sentence_transformer_cls.return_value = model

    encoder = MPNetTextEncoder(
        model_name="test-model",
        revision="test-revision",
        device="cpu",
        cache_folder="/tmp/models",
    )

    assert encoder.max_seq_length == 384
    assert encoder.special_token_count == 2
    assert encoder.max_content_tokens == 382


@patch(
    "app.services.features.text_encoder.mpnet."
    "SentenceTransformer"
)
def test_mpnet_rejects_wrong_dimension(
    sentence_transformer_cls,
):
    model = make_mock_model()

    model.get_sentence_embedding_dimension.return_value = 384

    sentence_transformer_cls.return_value = model

    with pytest.raises(
        RuntimeError,
        match="Unexpected MPNet embedding dimension",
    ):
        MPNetTextEncoder(
            model_name="wrong-model",
            revision="revision",
            device="cpu",
            cache_folder="/tmp/models",
        )


@patch(
    "app.services.features.text_encoder.mpnet."
    "SentenceTransformer"
)
def test_mpnet_encoder_contract(
    sentence_transformer_cls,
):
    model = make_mock_model()

    sentence_transformer_cls.return_value = model

    encoder = MPNetTextEncoder(
        model_name="test-model",
        revision="test-revision",
        device="cpu",
        cache_folder="/tmp/models",
    )

    encoder._tokenize_without_special_tokens = (
        MagicMock(
            return_value=(10, 20, 30)
        )
    )

    vector = np.zeros(
        (1, 768),
        dtype=np.float32,
    )
    vector[0, 0] = 1.0

    encoder._encode_chunks = MagicMock(
        return_value=vector
    )

    result = encoder.encode(
        "Today was productive."
    )

    assert result.dimension == 768
    assert len(result.embedding) == 768

    assert (
        result.encoder_version
        == TEXT_ENCODER_VERSION
    )

    assert (
        result.encoder_revision
        == "test-revision"
    )

    assert result.normalized is True

    assert result.embedding[0] == pytest.approx(
        1.0
    )


@patch(
    "app.services.features.text_encoder.mpnet."
    "SentenceTransformer"
)
def test_long_text_is_split_using_token_budget(
    sentence_transformer_cls,
):
    model = make_mock_model()

    sentence_transformer_cls.return_value = model

    encoder = MPNetTextEncoder(
        model_name="test-model",
        revision="revision",
        device="cpu",
        cache_folder="/tmp/models",
    )

    original_tokens = tuple(
        range(1000)
    )

    encoder._tokenize_without_special_tokens = (
        MagicMock(
            return_value=original_tokens
        )
    )

    captured_chunks = []

    def fake_encode_chunks(chunks):
        captured_chunks.extend(chunks)

        embeddings = np.zeros(
            (len(chunks), 768),
            dtype=np.float32,
        )

        embeddings[:, 0] = 1.0

        return embeddings

    encoder._encode_chunks = fake_encode_chunks

    encoder.encode("long journal")

    assert [
        chunk.token_count
        for chunk in captured_chunks
    ] == [382, 382, 236]

    reconstructed = tuple(
        token_id
        for chunk in captured_chunks
        for token_id in chunk.token_ids
    )

    assert reconstructed == original_tokens


@patch(
    "app.services.features.text_encoder.mpnet."
    "SentenceTransformer"
)
def test_weighted_aggregation_uses_token_counts(
    sentence_transformer_cls,
):
    model = make_mock_model()

    sentence_transformer_cls.return_value = model

    encoder = MPNetTextEncoder(
        model_name="test-model",
        revision="revision",
        device="cpu",
        cache_folder="/tmp/models",
    )

    chunks = (
        TokenChunk(
            token_ids=tuple(range(3))
        ),
        TokenChunk(
            token_ids=(100,),
        ),
    )

    embeddings = np.zeros(
        (2, 768),
        dtype=np.float32,
    )

    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0

    result = (
        encoder._aggregate_chunk_embeddings(
            embeddings,
            chunks,
        )
    )

    # Weighted pre-normalization vector:
    #
    # (3 * [1, 0] + 1 * [0, 1]) / 4
    # = [0.75, 0.25]
    #
    # Direction after normalization is [3, 1].

    expected_first = (
        3.0 / math.sqrt(10.0)
    )
    expected_second = (
        1.0 / math.sqrt(10.0)
    )

    assert result[0] == pytest.approx(
        expected_first,
        abs=1e-6,
    )

    assert result[1] == pytest.approx(
        expected_second,
        abs=1e-6,
    )

    assert np.linalg.norm(
        result
    ) == pytest.approx(
        1.0,
        abs=1e-6,
    )


@patch(
    "app.services.features.text_encoder.mpnet."
    "SentenceTransformer"
)
def test_aggregation_rejects_zero_vector(
    sentence_transformer_cls,
):
    model = make_mock_model()

    sentence_transformer_cls.return_value = model

    encoder = MPNetTextEncoder(
        model_name="test-model",
        revision="revision",
        device="cpu",
        cache_folder="/tmp/models",
    )

    chunks = (
        TokenChunk(
            token_ids=(1, 2, 3)
        ),
    )

    embeddings = np.zeros(
        (1, 768),
        dtype=np.float32,
    )

    with pytest.raises(
        RuntimeError,
        match="Cannot normalize aggregated",
    ):
        encoder._aggregate_chunk_embeddings(
            embeddings,
            chunks,
        )


@patch(
    "app.services.features.text_encoder.mpnet."
    "SentenceTransformer"
)
def test_final_result_has_unit_norm(
    sentence_transformer_cls,
):
    model = make_mock_model()

    sentence_transformer_cls.return_value = model

    encoder = MPNetTextEncoder(
        model_name="test-model",
        revision="revision",
        device="cpu",
        cache_folder="/tmp/models",
    )

    encoder._tokenize_without_special_tokens = (
        MagicMock(
            return_value=(1, 2, 3)
        )
    )

    value = 1.0 / math.sqrt(768)

    encoder._encode_chunks = MagicMock(
        return_value=np.full(
            (1, 768),
            value,
            dtype=np.float32,
        )
    )

    result = encoder.encode(
        "test journal"
    )

    norm = math.sqrt(
        sum(
            value * value
            for value in result.embedding
        )
    )

    assert norm == pytest.approx(
        1.0,
        abs=1e-5,
    )
    
@patch(
    "app.services.features.text_encoder.mpnet."
    "SentenceTransformer"
)
def test_tokenization_explicitly_disables_truncation(
    sentence_transformer_cls,
):
    model = make_mock_model()

    tokenizer = model.tokenizer

    tokenizer.return_value = {
        "input_ids": list(range(1000))
    }

    sentence_transformer_cls.return_value = model

    encoder = MPNetTextEncoder(
        model_name="test-model",
        revision="revision",
        device="cpu",
        cache_folder="/tmp/models",
    )

    result = (
        encoder._tokenize_without_special_tokens(
            "long journal"
        )
    )

    tokenizer.assert_called_once_with(
        "long journal",
        add_special_tokens=False,
        truncation=False,
    )

    assert len(result) == 1000
    assert result == tuple(range(1000))