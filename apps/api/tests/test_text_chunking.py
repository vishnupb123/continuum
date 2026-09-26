import pytest

from app.services.features.text_encoder.chunking import (
    TEXT_CHUNKING_VERSION,
    chunk_token_ids,
)


def test_chunking_version_is_frozen():
    assert (
        TEXT_CHUNKING_VERSION
        == "text-chunking-v1"
    )


def test_empty_tokens_produce_no_chunks():
    chunks = chunk_token_ids(
        [],
        max_content_tokens=382,
    )

    assert chunks == tuple()


def test_short_sequence_produces_one_chunk():
    tokens = list(range(100))

    chunks = chunk_token_ids(
        tokens,
        max_content_tokens=382,
    )

    assert len(chunks) == 1
    assert chunks[0].token_count == 100
    assert list(chunks[0].token_ids) == tokens


def test_exact_limit_produces_one_chunk():
    tokens = list(range(382))

    chunks = chunk_token_ids(
        tokens,
        max_content_tokens=382,
    )

    assert len(chunks) == 1
    assert chunks[0].token_count == 382


def test_limit_plus_one_produces_two_chunks():
    tokens = list(range(383))

    chunks = chunk_token_ids(
        tokens,
        max_content_tokens=382,
    )

    assert len(chunks) == 2

    assert chunks[0].token_count == 382
    assert chunks[1].token_count == 1


def test_long_sequence_preserves_every_token_once():
    tokens = list(range(9780))

    chunks = chunk_token_ids(
        tokens,
        max_content_tokens=382,
    )

    reconstructed = [
        token_id
        for chunk in chunks
        for token_id in chunk.token_ids
    ]

    assert reconstructed == tokens

    assert sum(
        chunk.token_count
        for chunk in chunks
    ) == len(tokens)


def test_invalid_content_limit_is_rejected():
    with pytest.raises(ValueError):
        chunk_token_ids(
            [1, 2, 3],
            max_content_tokens=0,
        )