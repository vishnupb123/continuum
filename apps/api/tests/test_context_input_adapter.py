import hashlib

import numpy as np
import pytest

from app.models.journal import JournalEntry
from app.models.journal_feature_set import JournalFeatureSet
from app.models.text_feature import TextFeature
from app.models.user import User
from app.services.context.input_adapter import (
    ContextInputUnavailableError,
    build_context_model_input,
)
from app.services.features.fingerprinting import (
    calculate_journal_source_hash,
)
from app.models.audio_feature import AudioFeature
from app.services.context.input_validation import (
    ContextInputValidationError,
)
from app.models.journal_audio import JournalAudio


class FakeAudioStorage:
    def __init__(
        self,
        audio_bytes: bytes,
    ) -> None:
        self.audio_bytes = audio_bytes

    def get(
        self,
        key: str,
    ) -> bytes:
        return self.audio_bytes


def make_journal(
    db_session,
    *,
    email="context-adapter@example.com",
    text="Context adapter test journal.",
):
    user = User(
        email=email,
        password_hash="test-password-hash",
        display_name="Context Adapter Test User",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="TEXT",
        raw_text=text,
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.commit()
    db_session.refresh(journal)

    return journal

def make_voice_journal(
    db_session,
    *,
    email: str,
):
    user = User(
        email=email,
        password_hash="test-password-hash",
        display_name="Context Voice Test User",
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="VOICE",
        raw_text=(
            "This is a transcript for the "
            "M4 context adapter test."
        ),
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.flush()

    audio_bytes = (
        b"context-adapter-test-audio"
    )

    journal.audio = JournalAudio(
        journal_id=journal.id,
        storage_key=(
            f"tests/context/{journal.id}/audio.wav"
        ),
        original_filename="audio.wav",
        mime_type="audio/wav",
        size_bytes=len(audio_bytes),
        transcription_status="TRANSCRIBED",
    )

    db_session.commit()
    db_session.refresh(
        journal,
        attribute_names=["audio"],
    )

    return journal, audio_bytes


def make_completed_text_feature_set(
    db_session,
    *,
    journal,
    source_hash=None,
):
    if source_hash is None:
        source_hash = (
            calculate_journal_source_hash(
                journal
            )
        )

    embedding = np.linspace(
        0.0,
        1.0,
        768,
        dtype=np.float32,
    ).tolist()

    feature_set = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash=source_hash,
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
        encoder_revision="test-text-revision",
        embedding_dimension=768,
        embedding=embedding,
        word_count=4,
        character_count=len(
            journal.raw_text
        ),
        quality_status="GOOD",
        feature_metadata={},
    )

    db_session.add(feature_set)
    db_session.commit()
    db_session.refresh(feature_set)

    return feature_set


def test_loader_builds_input_from_current_completed_generation(
    db_session,
):
    journal = make_journal(
        db_session,
        email="context-current@example.com",
    )

    feature_set = make_completed_text_feature_set(
        db_session,
        journal=journal,
    )

    model_input = build_context_model_input(
        db_session,
        journal=journal,
    )

    assert model_input.journal_id == journal.id

    assert (
        model_input.feature_set_id
        == feature_set.id
    )

    assert model_input.entry_type == "TEXT"

    assert (
        model_input.feature_pipeline_version
        == "m3-v1"
    )

    assert (
        model_input.source_hash
        == feature_set.source_hash
    )

    assert (
        model_input.text_embedding.shape
        == (768,)
    )

    assert (
        model_input.text_embedding.dtype
        == np.float32
    )

    assert model_input.text_quality == "GOOD"

    assert (
        model_input.text_provenance.encoder_version
        == "text-encoder-v1"
    )

    assert model_input.audio_embedding is None
    assert model_input.audio_quality is None
    assert model_input.audio_provenance is None


def test_loader_rejects_when_no_current_generation_exists(
    db_session,
):
    journal = make_journal(
        db_session,
        email="context-no-generation@example.com",
    )

    with pytest.raises(
        ContextInputUnavailableError,
        match=(
            "Current completed feature "
            "generation is unavailable"
        ),
    ):
        build_context_model_input(
            db_session,
            journal=journal,
        )


def test_loader_does_not_fall_back_to_historical_generation(
    db_session,
):
    journal = make_journal(
        db_session,
        email="context-stale@example.com",
        text="Original journal content.",
    )

    historical = (
        make_completed_text_feature_set(
            db_session,
            journal=journal,
        )
    )

    historical_id = historical.id

    journal.raw_text = (
        "Edited journal content that changes "
        "the authoritative source identity."
    )

    db_session.commit()
    db_session.refresh(journal)

    assert (
        calculate_journal_source_hash(journal)
        != historical.source_hash
    )

    with pytest.raises(
        ContextInputUnavailableError,
        match=(
            "Current completed feature "
            "generation is unavailable"
        ),
    ):
        build_context_model_input(
            db_session,
            journal=journal,
        )

    preserved = db_session.get(
        JournalFeatureSet,
        historical_id,
    )

    assert preserved is not None
    assert preserved.status == "COMPLETED"


def test_loader_rejects_feature_set_without_text_feature(
    db_session,
):
    journal = make_journal(
        db_session,
        email="context-no-text@example.com",
    )

    feature_set = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash=(
            calculate_journal_source_hash(
                journal
            )
        ),
        status="COMPLETED",
    )

    db_session.add(feature_set)
    db_session.commit()

    with pytest.raises(
        ContextInputUnavailableError,
        match="has no text feature",
    ):
        build_context_model_input(
            db_session,
            journal=journal,
        )


def test_loader_rejects_missing_text_encoder_provenance(
    db_session,
):
    journal = make_journal(
        db_session,
        email="context-no-provenance@example.com",
    )

    feature_set = (
        make_completed_text_feature_set(
            db_session,
            journal=journal,
        )
    )

    feature_set.text_feature.encoder_name = None

    db_session.commit()

    with pytest.raises(
        ContextInputValidationError,
        match="Text encoder name is unavailable",
    ):
        build_context_model_input(
            db_session,
            journal=journal,
        )
        
def make_completed_voice_feature_set(
    db_session,
    *,
    journal,
):
    feature_set = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash=(
            calculate_journal_source_hash(
                journal
            )
        ),
        status="COMPLETED",
    )

    text_embedding = np.linspace(
        0.0,
        1.0,
        768,
        dtype=np.float32,
    ).tolist()

    audio_embedding = np.linspace(
        1.0,
        0.0,
        768,
        dtype=np.float32,
    ).tolist()

    feature_set.text_feature = TextFeature(
        source_type="TRANSCRIPT",
        preprocessing_version="text-preprocess-v1",
        encoder_name="test-text-encoder",
        encoder_version="text-encoder-v1",
        encoder_revision="text-revision",
        embedding_dimension=768,
        embedding=text_embedding,
        word_count=4,
        character_count=len(
            journal.raw_text
        ),
        quality_status="GOOD",
        feature_metadata={},
    )

    feature_set.audio_feature = AudioFeature(
        preprocessing_version="audio-preprocess-v1",
        encoder_name="test-audio-encoder",
        encoder_version="audio-encoder-v1",
        encoder_revision="audio-revision",
        embedding_dimension=768,
        embedding=audio_embedding,
        duration_seconds=5.0,
        speech_ratio=0.9,
        sample_rate_hz=16000,
        quality_status="GOOD",
        feature_metadata={},
    )

    db_session.add(feature_set)
    db_session.commit()
    db_session.refresh(feature_set)

    return feature_set


def test_voice_requires_audio_feature(
    db_session,
    monkeypatch,
):
    journal, audio_bytes = make_voice_journal(
        db_session,
        email="context-voice-no-audio@example.com",
    )

    fake_storage = FakeAudioStorage(
        audio_bytes
    )

    monkeypatch.setattr(
        "app.services.features.fingerprinting."
        "get_audio_storage",
        lambda: fake_storage,
    )

    make_completed_text_feature_set(
        db_session,
        journal=journal,
    )

    with pytest.raises(
        ContextInputUnavailableError,
        match=(
            "VOICE feature generation "
            "has no audio feature"
        ),
    ):
        build_context_model_input(
            db_session,
            journal=journal,
        )

def test_voice_builds_both_modalities(
    db_session,
    monkeypatch,
):
    journal, audio_bytes = make_voice_journal(
        db_session,
        email="context-voice@example.com",
    )

    fake_storage = FakeAudioStorage(
        audio_bytes
    )

    monkeypatch.setattr(
        "app.services.features.fingerprinting."
        "get_audio_storage",
        lambda: fake_storage,
    )

    feature_set = (
        make_completed_voice_feature_set(
            db_session,
            journal=journal,
        )
    )

    model_input = build_context_model_input(
        db_session,
        journal=journal,
    )

    assert (
        model_input.feature_set_id
        == feature_set.id
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
    assert model_input.audio_quality == "GOOD"

    assert (
        model_input.audio_provenance
        is not None
    )

    assert (
        model_input.audio_provenance.encoder_version
        == "audio-encoder-v1"
    )


def test_degraded_evidence_is_accepted(
    db_session,
):
    journal = make_journal(
        db_session,
        email="context-degraded@example.com",
    )

    feature_set = make_completed_text_feature_set(
        db_session,
        journal=journal,
    )

    feature_set.text_feature.quality_status = (
        "DEGRADED"
    )

    db_session.commit()

    model_input = build_context_model_input(
        db_session,
        journal=journal,
    )

    assert (
        model_input.text_quality
        == "DEGRADED"
    )


def test_unusable_evidence_is_rejected(
    db_session,
):
    journal = make_journal(
        db_session,
        email="context-unusable@example.com",
    )

    feature_set = make_completed_text_feature_set(
        db_session,
        journal=journal,
    )

    feature_set.text_feature.quality_status = (
        "UNUSABLE"
    )

    db_session.commit()

    with pytest.raises(
        ContextInputValidationError,
        match="quality is not usable",
    ):
        build_context_model_input(
            db_session,
            journal=journal,
        )


def test_wrong_declared_dimension_is_rejected(
    db_session,
):
    journal = make_journal(
        db_session,
        email="context-wrong-dimension@example.com",
    )

    feature_set = make_completed_text_feature_set(
        db_session,
        journal=journal,
    )

    feature_set.text_feature.embedding_dimension = (
        256
    )

    db_session.commit()

    with pytest.raises(
        ContextInputValidationError,
        match="declared dimension",
    ):
        build_context_model_input(
            db_session,
            journal=journal,
        )


# def test_non_finite_embedding_is_rejected(
#     db_session,
# ):
#     journal = make_journal(
#         db_session,
#         email="context-nonfinite@example.com",
#     )

#     feature_set = make_completed_text_feature_set(
#         db_session,
#         journal=journal,
#     )

#     embedding = np.zeros(
#         768,
#         dtype=np.float32,
#     )

#     embedding[100] = np.nan

#     feature_set.text_feature.embedding = (
#         embedding.tolist()
#     )

#     db_session.commit()

#     with pytest.raises(
#         ContextInputValidationError,
#         match="non-finite",
#     ):
#         build_context_model_input(
#             db_session,
#             journal=journal,
#         )


def test_text_does_not_consume_audio_even_if_present(
    db_session,
):
    journal = make_journal(
        db_session,
        email="context-text-extra-audio@example.com",
    )

    feature_set = make_completed_text_feature_set(
        db_session,
        journal=journal,
    )

    feature_set.audio_feature = AudioFeature(
        preprocessing_version="audio-preprocess-v1",
        encoder_name="test-audio-encoder",
        encoder_version="audio-encoder-v1",
        encoder_revision="audio-revision",
        embedding_dimension=768,
        embedding=np.zeros(
            768,
            dtype=np.float32,
        ).tolist(),
        duration_seconds=5.0,
        speech_ratio=0.9,
        sample_rate_hz=16000,
        quality_status="GOOD",
        feature_metadata={},
    )

    db_session.commit()

    model_input = build_context_model_input(
        db_session,
        journal=journal,
    )

    assert model_input.entry_type == "TEXT"
    assert model_input.audio_embedding is None
    assert model_input.audio_quality is None
    assert model_input.audio_provenance is None
        