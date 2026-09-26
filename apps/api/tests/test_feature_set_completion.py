import uuid

import pytest

from app.models.audio_feature import AudioFeature
from app.models.feature_constants import (
    FEATURE_QUALITY_GOOD,
    FEATURE_STATUS_COMPLETED,
    FEATURE_STATUS_PROCESSING,
)
from app.models.journal_feature_set import (
    JournalFeatureSet,
)
from app.models.text_feature import TextFeature
from app.services.features.feature_sets import (
    complete_feature_set_if_ready,
)


def make_feature_set() -> JournalFeatureSet:
    return JournalFeatureSet(
        id=uuid.uuid4(),
        journal_id=uuid.uuid4(),
        pipeline_version="m3-v1",
        source_hash=str(uuid.uuid4()),
        status=FEATURE_STATUS_PROCESSING,
    )


def make_text_feature() -> TextFeature:
    return TextFeature(
        id=uuid.uuid4(),
        source_type="RAW_TEXT",
        preprocessing_version="text-preprocess-v1",
        encoder_name="test-encoder",
        encoder_revision="test-revision",
        encoder_version="text-encoder-v1",
        embedding_dimension=768,
        embedding=[0.0] * 768,
        word_count=10,
        character_count=50,
        quality_status=FEATURE_QUALITY_GOOD,
        feature_metadata={},
    )


def make_audio_feature() -> AudioFeature:
    return AudioFeature(
        id=uuid.uuid4(),
        preprocessing_version="audio-preprocess-v1",
        encoder_name=None,
        encoder_version=None,
        embedding_dimension=None,
        duration_seconds=3.0,
        speech_ratio=None,
        sample_rate_hz=16_000,
        quality_status=FEATURE_QUALITY_GOOD,
        feature_metadata={},
    )


def test_text_requires_text_feature():
    feature_set = make_feature_set()

    completed = complete_feature_set_if_ready(
        feature_set,
        entry_type="TEXT",
    )

    assert completed is False

    assert (
        feature_set.status
        == FEATURE_STATUS_PROCESSING
    )

    assert feature_set.completed_at is None


def test_text_completes_with_text_feature():
    feature_set = make_feature_set()

    feature_set.text_feature = (
        make_text_feature()
    )

    completed = complete_feature_set_if_ready(
        feature_set,
        entry_type="TEXT",
    )

    assert completed is True

    assert (
        feature_set.status
        == FEATURE_STATUS_COMPLETED
    )

    assert (
        feature_set.completed_at
        is not None
    )


def test_voice_does_not_complete_with_text_only():
    feature_set = make_feature_set()

    feature_set.text_feature = (
        make_text_feature()
    )

    completed = complete_feature_set_if_ready(
        feature_set,
        entry_type="VOICE",
    )

    assert completed is False

    assert (
        feature_set.status
        == FEATURE_STATUS_PROCESSING
    )

    assert feature_set.completed_at is None


def test_voice_does_not_complete_with_audio_only():
    feature_set = make_feature_set()

    feature_set.audio_feature = (
        make_audio_feature()
    )

    completed = complete_feature_set_if_ready(
        feature_set,
        entry_type="VOICE",
    )

    assert completed is False

    assert (
        feature_set.status
        == FEATURE_STATUS_PROCESSING
    )

    assert feature_set.completed_at is None


def test_voice_completes_with_both_modalities():
    feature_set = make_feature_set()

    feature_set.text_feature = (
        make_text_feature()
    )

    feature_set.audio_feature = (
        make_audio_feature()
    )

    completed = complete_feature_set_if_ready(
        feature_set,
        entry_type="VOICE",
    )

    assert completed is True

    assert (
        feature_set.status
        == FEATURE_STATUS_COMPLETED
    )

    assert (
        feature_set.completed_at
        is not None
    )


def test_unsupported_entry_type_is_rejected():
    feature_set = make_feature_set()

    with pytest.raises(
        ValueError,
        match="Unsupported journal entry type",
    ):
        complete_feature_set_if_ready(
            feature_set,
            entry_type="VIDEO",
        )
        
def test_voice_completes_text_then_audio():
    feature_set = make_feature_set()

    # Text finishes first.
    feature_set.text_feature = (
        make_text_feature()
    )

    completed_after_text = (
        complete_feature_set_if_ready(
            feature_set,
            entry_type="VOICE",
        )
    )

    assert completed_after_text is False
    assert (
        feature_set.status
        == FEATURE_STATUS_PROCESSING
    )
    assert feature_set.completed_at is None

    # Audio finishes later.
    feature_set.audio_feature = (
        make_audio_feature()
    )

    completed_after_audio = (
        complete_feature_set_if_ready(
            feature_set,
            entry_type="VOICE",
        )
    )

    assert completed_after_audio is True
    assert (
        feature_set.status
        == FEATURE_STATUS_COMPLETED
    )
    assert feature_set.completed_at is not None


def test_voice_completes_audio_then_text():
    feature_set = make_feature_set()

    # Audio finishes first.
    feature_set.audio_feature = (
        make_audio_feature()
    )

    completed_after_audio = (
        complete_feature_set_if_ready(
            feature_set,
            entry_type="VOICE",
        )
    )

    assert completed_after_audio is False
    assert (
        feature_set.status
        == FEATURE_STATUS_PROCESSING
    )
    assert feature_set.completed_at is None

    # Text finishes later.
    feature_set.text_feature = (
        make_text_feature()
    )

    completed_after_text = (
        complete_feature_set_if_ready(
            feature_set,
            entry_type="VOICE",
        )
    )

    assert completed_after_text is True
    assert (
        feature_set.status
        == FEATURE_STATUS_COMPLETED
    )
    assert feature_set.completed_at is not None