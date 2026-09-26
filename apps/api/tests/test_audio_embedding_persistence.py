import numpy as np
import pytest
from sqlalchemy import select

from app.models.audio_feature import (
    AUDIO_EMBEDDING_DIMENSION,
    AudioFeature,
)
from app.models.journal import JournalEntry
from app.models.journal_feature_set import JournalFeatureSet
from app.models.user import User


def _normalized_embedding(
    index: int,
) -> np.ndarray:
    embedding = np.zeros(
        AUDIO_EMBEDDING_DIMENSION,
        dtype=np.float32,
    )

    embedding[index] = 1.0

    return embedding


def test_audio_embedding_pgvector_round_trip(
    db_session,
):
    user = User(
        email="audio-vector@example.com",
        password_hash="test-hash",
        display_name="Audio Vector Test",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="VOICE",
        raw_text="Audio embedding persistence test.",
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.flush()
    feature_set = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash="audio-vector-test-source",
        status="PROCESSING",
    )

    db_session.add(feature_set)
    db_session.flush()

    expected_embedding = _normalized_embedding(
        0
    )

    audio_feature = AudioFeature(
        feature_set_id=feature_set.id,
        preprocessing_version="audio-preprocess-v1",
        encoder_name="microsoft/wavlm-base-plus",
        encoder_version="audio-encoder-v1",
        encoder_revision=(
            "4c66d4806a428f2e922ccfa1a962776e232d487b"
        ),
        embedding_dimension=768,
        embedding=expected_embedding.tolist(),
        duration_seconds=1.0,
        speech_ratio=None,
        sample_rate_hz=16_000,
        quality_status="GOOD",
        feature_metadata={},
    )

    db_session.add(audio_feature)
    db_session.commit()

    db_session.expire_all()

    persisted = db_session.scalar(
        select(AudioFeature).where(
            AudioFeature.id
            == audio_feature.id
        )
    )

    assert persisted is not None

    assert (
        persisted.encoder_name
        == "microsoft/wavlm-base-plus"
    )

    assert (
        persisted.encoder_version
        == "audio-encoder-v1"
    )

    assert (
        persisted.encoder_revision
        == (
            "4c66d4806a428f2e922ccfa1a962776e232d487b"
        )
    )

    assert (
        persisted.embedding_dimension
        == 768
    )

    assert persisted.embedding is not None

    actual_embedding = np.asarray(
        persisted.embedding,
        dtype=np.float32,
    )

    assert actual_embedding.shape == (
        768,
    )

    assert (
        actual_embedding.dtype
        == np.float32
    )

    np.testing.assert_allclose(
        actual_embedding,
        expected_embedding,
        rtol=0.0,
        atol=1e-7,
    )

    norm = np.linalg.norm(
        actual_embedding.astype(
            np.float64
        )
    )

    assert norm == pytest.approx(
        1.0,
        abs=1e-7,
    )


def test_audio_embedding_cosine_distance(
    db_session,
):
    user = User(
        email="audio-cosine@example.com",
        password_hash="test-hash",
        display_name="Audio Cosine Test",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="VOICE",
        raw_text="Cosine distance test.",
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.flush()

    feature_set = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash="audio-cosine-test-source",
        status="PROCESSING",
    )

    db_session.add(feature_set)
    db_session.flush()

    stored_embedding = _normalized_embedding(
        0
    )

    audio_feature = AudioFeature(
        feature_set_id=feature_set.id,
        preprocessing_version="audio-preprocess-v1",
        encoder_name="microsoft/wavlm-base-plus",
        encoder_version="audio-encoder-v1",
        encoder_revision=(
            "4c66d4806a428f2e922ccfa1a962776e232d487b"
        ),
        embedding_dimension=768,
        embedding=stored_embedding.tolist(),
        duration_seconds=1.0,
        speech_ratio=None,
        sample_rate_hz=16_000,
        quality_status="GOOD",
        feature_metadata={},
    )

    db_session.add(audio_feature)
    db_session.commit()

    same_embedding = _normalized_embedding(
        0
    )

    orthogonal_embedding = (
        _normalized_embedding(1)
    )

    same_distance = db_session.scalar(
        select(
            AudioFeature.embedding.cosine_distance(
                same_embedding.tolist()
            )
        ).where(
            AudioFeature.id
            == audio_feature.id
        )
    )

    orthogonal_distance = db_session.scalar(
        select(
            AudioFeature.embedding.cosine_distance(
                orthogonal_embedding.tolist()
            )
        ).where(
            AudioFeature.id
            == audio_feature.id
        )
    )

    assert same_distance == pytest.approx(
        0.0,
        abs=1e-6,
    )

    assert orthogonal_distance == pytest.approx(
        1.0,
        abs=1e-6,
    )