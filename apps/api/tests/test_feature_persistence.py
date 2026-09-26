import hashlib

import pytest
from sqlalchemy.exc import IntegrityError
import numpy as np

from app.models.audio_feature import AudioFeature
from app.models.journal import JournalEntry
from app.models.journal_feature_set import JournalFeatureSet
from app.models.text_feature import TextFeature
from app.models.user import User


def make_user(db_session, email="features@example.com"):
    user = User(
        email=email,
        password_hash="test-password-hash",
        display_name="Feature Test User",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def make_journal(
    db_session,
    user,
    *,
    entry_type="TEXT",
    text="Today was a productive day.",
):
    journal = JournalEntry(
        user_id=user.id,
        entry_type=entry_type,
        raw_text=text,
        status="COMPLETED",
    )
    db_session.add(journal)
    db_session.commit()
    db_session.refresh(journal)
    return journal


def source_hash(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def test_text_feature_set_persists(db_session):
    user = make_user(db_session)
    journal = make_journal(
        db_session,
        user,
    )

    feature_set = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash=source_hash(
            journal.raw_text
        ),
        status="COMPLETED",
    )

    feature_set.text_feature = TextFeature(
        source_type="RAW_TEXT",
        preprocessing_version="text-preprocess-v1",
        encoder_name=None,
        encoder_version=None,
        embedding_dimension=None,
        word_count=5,
        character_count=len(
            journal.raw_text
        ),
        quality_status="GOOD",
        feature_metadata={
            "test": True,
        },
    )

    db_session.add(feature_set)
    db_session.commit()

    db_session.expire_all()

    stored = db_session.get(
        JournalFeatureSet,
        feature_set.id,
    )

    assert stored is not None
    assert stored.journal_id == journal.id
    assert stored.pipeline_version == "m3-v1"
    assert stored.status == "COMPLETED"

    assert stored.text_feature is not None
    assert (
        stored.text_feature.source_type
        == "RAW_TEXT"
    )
    assert (
        stored.text_feature.word_count
        == 5
    )
    assert (
        stored.text_feature.feature_metadata
        == {"test": True}
    )

    assert stored.audio_feature is None
    
def test_text_embedding_persists(db_session):
    user = make_user(
        db_session,
        email="embedding-persistence@example.com",
    )

    journal = make_journal(
        db_session,
        user,
        text="Continuum should persist semantic embeddings.",
    )

    embedding = [
        float(index) / 768.0
        for index in range(768)
    ]

    feature_set = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash=source_hash(
            journal.raw_text
        ),
        status="COMPLETED",
    )

    feature_set.text_feature = TextFeature(
        source_type="RAW_TEXT",
        preprocessing_version="text-preprocess-v1",
        encoder_name="sentence-transformers/all-mpnet-base-v2",
        encoder_version="text-encoder-v1",
        encoder_revision=(
          "e8c3b32edf5434bc2275fc9bab85f82640a19130"
        ),
        embedding_dimension=768,
        embedding=embedding,
        word_count=5,
        character_count=len(
            journal.raw_text
        ),
        quality_status="GOOD",
        feature_metadata={
            "test": "embedding-persistence",
        },
    )

    db_session.add(feature_set)
    db_session.commit()

    text_feature_id = feature_set.text_feature.id

    db_session.expire_all()

    stored = db_session.get(
        TextFeature,
        text_feature_id,
    )

    assert stored is not None
    assert stored.embedding is not None

    assert stored.embedding_dimension == 768
    assert len(stored.embedding) == 768

    assert (
        stored.encoder_name
        == "sentence-transformers/all-mpnet-base-v2"
    )
    assert (
        stored.encoder_version
        == "text-encoder-v1"
    )
    assert (
        stored.encoder_revision
        == "e8c3b32edf5434bc2275fc9bab85f82640a19130"
    )

    np.testing.assert_allclose(
        np.asarray(
            stored.embedding,
            dtype=np.float32,
        ),
        np.asarray(
            embedding,
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_voice_feature_set_supports_both_modalities(
    db_session,
):
    user = make_user(
        db_session,
        email="voice-features@example.com",
    )

    journal = make_journal(
        db_session,
        user,
        entry_type="VOICE",
        text="This is a voice transcript.",
    )

    feature_set = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash=source_hash(
            "voice-source"
        ),
        status="COMPLETED",
    )

    feature_set.text_feature = TextFeature(
        source_type="TRANSCRIPT",
        preprocessing_version="text-preprocess-v1",
        word_count=5,
        character_count=len(
            journal.raw_text
        ),
        quality_status="GOOD",
        feature_metadata={},
    )

    feature_set.audio_feature = AudioFeature(
        preprocessing_version="audio-preprocess-v1",
        duration_seconds=23.46,
        speech_ratio=0.91,
        sample_rate_hz=16000,
        quality_status="GOOD",
        feature_metadata={},
    )

    db_session.add(feature_set)
    db_session.commit()

    db_session.expire_all()

    stored = db_session.get(
        JournalFeatureSet,
        feature_set.id,
    )

    assert stored.text_feature is not None
    assert stored.audio_feature is not None

    assert (
        stored.text_feature.source_type
        == "TRANSCRIPT"
    )

    assert (
        stored.audio_feature.duration_seconds
        == pytest.approx(23.46)
    )

    assert (
        stored.audio_feature.speech_ratio
        == pytest.approx(0.91)
    )

    assert (
        stored.audio_feature.sample_rate_hz
        == 16000
    )


def test_same_generation_cannot_be_duplicated(
    db_session,
):
    user = make_user(
        db_session,
        email="duplicate@example.com",
    )

    journal = make_journal(
        db_session,
        user,
    )

    fingerprint = source_hash(
        journal.raw_text
    )

    first = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash=fingerprint,
        status="PENDING",
    )

    db_session.add(first)
    db_session.commit()

    duplicate = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash=fingerprint,
        status="PENDING",
    )

    db_session.add(duplicate)

    with pytest.raises(IntegrityError):
        db_session.commit()

    db_session.rollback()


def test_same_pipeline_allows_changed_source(
    db_session,
):
    user = make_user(
        db_session,
        email="changed-source@example.com",
    )

    journal = make_journal(
        db_session,
        user,
    )

    first = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash=source_hash(
            "source version one"
        ),
        status="COMPLETED",
    )

    second = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash=source_hash(
            "source version two"
        ),
        status="COMPLETED",
    )

    db_session.add_all(
        [first, second]
    )
    db_session.commit()

    assert first.id != second.id

    assert (
        len(journal.feature_sets)
        == 2
    )


def test_new_pipeline_allows_same_source(
    db_session,
):
    user = make_user(
        db_session,
        email="pipeline-upgrade@example.com",
    )

    journal = make_journal(
        db_session,
        user,
    )

    fingerprint = source_hash(
        journal.raw_text
    )

    v1 = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash=fingerprint,
        status="COMPLETED",
    )

    v2 = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v2",
        source_hash=fingerprint,
        status="COMPLETED",
    )

    db_session.add_all([v1, v2])
    db_session.commit()

    assert v1.id != v2.id

    assert (
        len(journal.feature_sets)
        == 2
    )


def test_deleting_journal_cascades_feature_data(
    db_session,
):
    user = make_user(
        db_session,
        email="cascade@example.com",
    )

    journal = make_journal(
        db_session,
        user,
        entry_type="VOICE",
        text="Cascade test transcript.",
    )

    feature_set = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash=source_hash(
            "cascade-source"
        ),
        status="COMPLETED",
    )

    feature_set.text_feature = TextFeature(
        source_type="TRANSCRIPT",
        preprocessing_version="text-preprocess-v1",
        word_count=3,
        character_count=len(
            journal.raw_text
        ),
        quality_status="GOOD",
        feature_metadata={},
    )

    feature_set.audio_feature = AudioFeature(
        preprocessing_version="audio-preprocess-v1",
        duration_seconds=10.0,
        speech_ratio=0.9,
        sample_rate_hz=16000,
        quality_status="GOOD",
        feature_metadata={},
    )

    db_session.add(feature_set)
    db_session.commit()

    feature_set_id = feature_set.id
    text_feature_id = (
        feature_set.text_feature.id
    )
    audio_feature_id = (
        feature_set.audio_feature.id
    )

    db_session.delete(journal)
    db_session.commit()

    assert (
        db_session.get(
            JournalFeatureSet,
            feature_set_id,
        )
        is None
    )

    assert (
        db_session.get(
            TextFeature,
            text_feature_id,
        )
        is None
    )

    assert (
        db_session.get(
            AudioFeature,
            audio_feature_id,
        )
        is None
    )
    
def test_text_embeddings_support_cosine_similarity(
    db_session,
):
    user = make_user(
        db_session,
        email="cosine-similarity@example.com",
    )

    journal_a = make_journal(
        db_session,
        user,
        text="Vector A",
    )

    journal_b = make_journal(
        db_session,
        user,
        text="Vector B",
    )

    journal_c = make_journal(
        db_session,
        user,
        text="Vector C",
    )

    vector_a = [0.0] * 768
    vector_b = [0.0] * 768
    vector_c = [0.0] * 768

    # A and B point in nearly the same direction.
    vector_a[0] = 1.0

    vector_b[0] = 0.9
    vector_b[1] = 0.1

    # C points in a completely different direction.
    vector_c[1] = 1.0

    def add_feature_set(journal, embedding):
        feature_set = JournalFeatureSet(
            journal_id=journal.id,
            pipeline_version="m3-v1",
            source_hash=source_hash(
                journal.raw_text
            ),
            status="COMPLETED",
        )

        feature_set.text_feature = TextFeature(
            source_type="RAW_TEXT",
            preprocessing_version="text-preprocess-v1",
            encoder_name=(
                "sentence-transformers/"
                "all-mpnet-base-v2"
            ),
            encoder_version="text-encoder-v1",
            encoder_revision=(
                "e8c3b32edf5434bc2275fc9bab85f82640a19130"
            ),
            embedding_dimension=768,
            embedding=embedding,
            word_count=2,
            character_count=len(
                journal.raw_text
            ),
            quality_status="GOOD",
            feature_metadata={},
        )

        db_session.add(feature_set)

    add_feature_set(journal_a, vector_a)
    add_feature_set(journal_b, vector_b)
    add_feature_set(journal_c, vector_c)

    db_session.commit()

    results = (
        db_session.query(TextFeature)
        .order_by(
            TextFeature.embedding.cosine_distance(
                vector_a
            )
        )
        .all()
    )

    assert len(results) == 3

    # Exact/self match should be closest.
    assert (
        results[0]
        .feature_set
        .journal_id
        == journal_a.id
    )

    # Similar vector should come next.
    assert (
        results[1]
        .feature_set
        .journal_id
        == journal_b.id
    )

    # Orthogonal vector should be least similar.
    assert (
        results[2]
        .feature_set
        .journal_id
        == journal_c.id
    )