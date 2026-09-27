import io
import wave

import numpy as np
import pytest

from app.models.journal import JournalEntry
from app.models.journal_audio import JournalAudio
from app.models.journal_feature_set import (
    JournalFeatureSet,
)
from app.models.user import User

from app.services.features.audio_features import (
    extract_audio_features,
)
from app.services.features.feature_sets import (
    get_current_completed_feature_set,
)
from app.services.features.fingerprinting import (
    calculate_journal_source_hash,
)
from app.services.features.mock_audio_encoder import (
    MockAudioEncoder,
)
from app.services.features.text_features import (
    extract_text_features,
)


# ============================================================
# TEST AUDIO
# ============================================================


def _make_wav_bytes(
    *,
    sample_rate: int = 16_000,
    duration_seconds: float = 1.0,
    frequency_hz: float = 440.0,
) -> bytes:
    """
    Create a small deterministic PCM WAV signal.

    The signal is intentionally valid and usable so the real
    M3 audio preprocessing and quality pipeline can process it.
    """

    sample_count = int(
        sample_rate
        * duration_seconds
    )

    time = (
        np.arange(
            sample_count,
            dtype=np.float64,
        )
        / sample_rate
    )

    waveform = (
        0.25
        * np.sin(
            2.0
            * np.pi
            * frequency_hz
            * time
        )
    )

    pcm = (
        waveform
        * 32767.0
    ).astype(
        np.int16
    )

    buffer = io.BytesIO()

    with wave.open(
        buffer,
        "wb",
    ) as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(
            sample_rate
        )
        wav_file.writeframes(
            pcm.tobytes()
        )

    return buffer.getvalue()


# ============================================================
# FAKE STORAGE BOUNDARY
# ============================================================


class FakeAudioStorage:
    """
    Deterministic storage adapter used by the multimodal
    integration test.

    Both source fingerprinting and audio feature extraction
    must observe the exact same stored audio bytes.
    """

    def __init__(
        self,
        audio_bytes: bytes,
    ) -> None:
        self.audio_bytes = audio_bytes

    def get(
        self,
        key: str,
    ) -> bytes:
        assert (
            key
            == "test/multimodal.wav"
        )

        return self.audio_bytes


# ============================================================
# VOICE JOURNAL FIXTURE
# ============================================================


def _create_voice_generation(
    db_session,
):
    """
    Create the completed M2 portion of a VOICE journal.

    Important:

    This helper intentionally does NOT create the M3
    JournalFeatureSet.

    VOICE generation identity depends on:

        transcript + exact stored audio bytes

    Therefore the fake storage boundary must be installed
    before calculate_journal_source_hash() is called.
    """

    user = User(
        email=(
            "multimodal-generation@example.com"
        ),
        password_hash=(
            "test-password-hash"
        ),
    )

    db_session.add(user)
    db_session.flush()

    journal = JournalEntry(
        user_id=user.id,
        entry_type="VOICE",
        raw_text=(
            "Today was a demanding day, "
            "but I managed to finish my work "
            "and I feel better now."
        ),
        status="COMPLETED",
    )

    db_session.add(journal)
    db_session.flush()

    audio_bytes = _make_wav_bytes()

    journal_audio = JournalAudio(
        journal_id=journal.id,
        storage_key=(
            "test/multimodal.wav"
        ),
        original_filename=(
            "multimodal.wav"
        ),
        mime_type="audio/wav",
        size_bytes=len(
            audio_bytes
        ),
        transcription_status=(
            "TRANSCRIBED"
        ),
    )

    db_session.add(
        journal_audio
    )

    db_session.flush()

    # calculate_journal_source_hash() intentionally reads
    # journal.audio for VOICE journals.
    #
    # Refresh the relationship so this JournalEntry instance
    # reflects the persisted JournalAudio row.
    db_session.refresh(
        journal,
        attribute_names=["audio"],
    )

    return (
        journal,
        audio_bytes,
    )


# ============================================================
# M3.10 MULTIMODAL END-TO-END ACCEPTANCE
# ============================================================


def test_voice_generation_completes_with_text_then_audio(
    db_session,
    monkeypatch,
):
    """
    M3.10 acceptance test.

    Proves that a completed M2 VOICE journal can produce one
    coherent M3 multimodal generation containing:

        - semantic text representation
        - deterministic acoustic representation
        - learned audio representation
        - representation provenance
        - completed generation state

    It then proves that the authoritative M3 -> M4 lookup
    returns that exact completed generation.
    """

    # ---------------------------------------------------------
    # Completed M2 VOICE journal
    # ---------------------------------------------------------

    (
        journal,
        audio_bytes,
    ) = _create_voice_generation(
        db_session
    )

    fake_storage = FakeAudioStorage(
        audio_bytes
    )

    # ---------------------------------------------------------
    # Shared storage boundary
    # ---------------------------------------------------------
    #
    # Source fingerprinting and audio feature extraction must
    # observe exactly the same stored audio object.
    # ---------------------------------------------------------

    monkeypatch.setattr(
        "app.services.features.fingerprinting."
        "get_audio_storage",
        lambda: fake_storage,
    )

    monkeypatch.setattr(
        "app.services.features.audio_features."
        "get_audio_storage",
        lambda: fake_storage,
    )

    # ---------------------------------------------------------
    # Canonical M3 generation identity
    # ---------------------------------------------------------

    source_hash = (
        calculate_journal_source_hash(
            journal
        )
    )

    feature_set = JournalFeatureSet(
        journal_id=journal.id,
        pipeline_version="m3-v1",
        source_hash=source_hash,
        status="PENDING",
    )

    db_session.add(
        feature_set
    )

    db_session.flush()

    # ---------------------------------------------------------
    # Deterministic learned-audio boundary
    # ---------------------------------------------------------
    #
    # The integration test exercises the real audio decoding,
    # preprocessing, quality and acoustic feature pipeline.
    #
    # Only the expensive learned encoder is substituted with
    # the deterministic contract-compatible MockAudioEncoder.
    # ---------------------------------------------------------

    mock_audio_encoder = (
        MockAudioEncoder()
    )

    monkeypatch.setattr(
        "app.services.features.audio_features."
        "get_audio_encoder",
        lambda: mock_audio_encoder,
    )

    # ---------------------------------------------------------
    # Text representation
    # ---------------------------------------------------------

    text_result = (
        extract_text_features(
            db_session,
            journal=journal,
            feature_set=feature_set,
        )
    )

    assert (
        text_result.text_feature
        is not None
    )

    assert (
        feature_set.text_feature
        is not None
    )

    # VOICE is not complete yet because the
    # audio representation is still missing.
    assert (
        feature_set.status
        == "PROCESSING"
    )

    assert (
        feature_set.completed_at
        is None
    )

    # ---------------------------------------------------------
    # Audio representations
    # ---------------------------------------------------------

    audio_result = (
        extract_audio_features(
            db_session,
            journal=journal,
            feature_set=feature_set,
        )
    )

    assert (
        audio_result.audio_feature
        is not None
    )

    assert (
        feature_set.audio_feature
        is not None
    )

    # Both modalities now exist.
    assert (
        feature_set.status
        == "COMPLETED"
    )

    assert (
        feature_set.completed_at
        is not None
    )

    assert (
        feature_set.text_feature
        is text_result.text_feature
    )

    assert (
        feature_set.audio_feature
        is audio_result.audio_feature
    )

    # ---------------------------------------------------------
    # M2 lifecycle isolation
    # ---------------------------------------------------------
    #
    # Successful M3 processing must not mutate the completed
    # journal lifecycle owned by M2.
    # ---------------------------------------------------------

    assert (
        journal.status
        == "COMPLETED"
    )

    # ---------------------------------------------------------
    # Text representation contract
    # ---------------------------------------------------------

    assert (
        feature_set
        .text_feature
        .embedding
        is not None
    )

    assert (
        feature_set
        .text_feature
        .embedding_dimension
        == 768
    )

    # ---------------------------------------------------------
    # M3.5 deterministic acoustic representation
    # ---------------------------------------------------------

    metadata = (
        feature_set
        .audio_feature
        .feature_metadata
    )

    assert (
        "acoustic_features"
        in metadata
    )

    acoustic = metadata[
        "acoustic_features"
    ]

    assert (
        acoustic[
            "feature_version"
        ]
        == "acoustic-features-v1"
    )

    assert (
        acoustic[
            "frame_count"
        ]
        > 0
    )

    assert (
        len(
            acoustic[
                "mfcc_mean"
            ]
        )
        == 13
    )

    assert (
        len(
            acoustic[
                "mfcc_std"
            ]
        )
        == 13
    )

    # M3.5 signal activity must not be
    # misrepresented as speech/VAD.
    assert (
        feature_set
        .audio_feature
        .speech_ratio
        is None
    )

    # ---------------------------------------------------------
    # M3.6 learned audio representation
    # ---------------------------------------------------------

    audio_feature = (
        feature_set.audio_feature
    )

    assert (
        audio_feature.encoder_name
        == "mock-audio-encoder"
    )

    assert (
        audio_feature.encoder_version
        == "audio-encoder-v1"
    )

    assert (
        audio_feature.encoder_revision
        == "deterministic-v1"
    )

    assert (
        audio_feature.embedding_dimension
        == 768
    )

    assert (
        audio_feature.embedding
        is not None
    )

    audio_embedding = np.asarray(
        audio_feature.embedding,
        dtype=np.float32,
    )

    assert (
        audio_embedding.shape
        == (768,)
    )

    assert np.all(
        np.isfinite(
            audio_embedding
        )
    )

    assert (
        np.linalg.norm(
            audio_embedding.astype(
                np.float64
            )
        )
        == pytest.approx(
            1.0,
            abs=1e-6,
        )
    )

    # Returned representation must match
    # what was persisted.
    np.testing.assert_allclose(
        audio_embedding,
        audio_result
        .audio_encoding
        .embedding,
        rtol=0.0,
        atol=1e-7,
    )

    # ---------------------------------------------------------
    # M3.6 representation provenance
    # ---------------------------------------------------------

    assert (
        "audio_embedding"
        in metadata
    )

    embedding_metadata = metadata[
        "audio_embedding"
    ]

    assert (
        embedding_metadata[
            "encoder_name"
        ]
        == "mock-audio-encoder"
    )

    assert (
        embedding_metadata[
            "encoder_version"
        ]
        == "audio-encoder-v1"
    )

    assert (
        embedding_metadata[
            "encoder_revision"
        ]
        == "deterministic-v1"
    )

    assert (
        embedding_metadata[
            "embedding_dimension"
        ]
        == 768
    )

    assert (
        embedding_metadata[
            "sample_rate_hz"
        ]
        == 16_000
    )

    assert (
        embedding_metadata[
            "pooling_strategy"
        ]
        == "masked_temporal_mean"
    )

    assert (
        embedding_metadata[
            "normalization"
        ]
        == "l2"
    )

    # ---------------------------------------------------------
    # Final multimodal contract
    # ---------------------------------------------------------

    assert (
        feature_set
        .text_feature
        .embedding
        is not None
    )

    assert (
        feature_set
        .audio_feature
        .embedding
        is not None
    )

    assert (
        feature_set.status
        == "COMPLETED"
    )

    # ---------------------------------------------------------
    # M3.10 authoritative M3 -> M4 handoff
    # ---------------------------------------------------------
    #
    # Persist the completed generation, then retrieve it using
    # the same authoritative lookup that downstream M4 code
    # will use.
    #
    # The lookup recalculates the current source fingerprint,
    # so the same fake storage remains installed here.
    # ---------------------------------------------------------

    feature_set_id = (
        feature_set.id
    )

    db_session.commit()
    db_session.expire_all()

    current_feature_set = (
        get_current_completed_feature_set(
            db_session,
            journal=journal,
        )
    )

    assert (
        current_feature_set
        is not None
    )

    # The M4-facing lookup must return the exact generation
    # produced by this multimodal extraction.
    assert (
        current_feature_set.id
        == feature_set_id
    )

    assert (
        current_feature_set.status
        == "COMPLETED"
    )

    # ---------------------------------------------------------
    # Both modality representations survive the handoff
    # ---------------------------------------------------------

    assert (
        current_feature_set
        .text_feature
        is not None
    )

    assert (
        current_feature_set
        .audio_feature
        is not None
    )

    assert (
        current_feature_set
        .text_feature
        .embedding
        is not None
    )

    assert (
        current_feature_set
        .audio_feature
        .embedding
        is not None
    )

    # ---------------------------------------------------------
    # Representation contracts remain M4-consumable
    # ---------------------------------------------------------

    assert (
        current_feature_set
        .text_feature
        .embedding_dimension
        == 768
    )

    assert (
        current_feature_set
        .audio_feature
        .embedding_dimension
        == 768
    )

    assert (
        current_feature_set
        .audio_feature
        .encoder_version
        == "audio-encoder-v1"
    )

    # ---------------------------------------------------------
    # Generation identity survives persistence / handoff
    # ---------------------------------------------------------

    assert (
        current_feature_set.source_hash
        == source_hash
    )

    assert (
        current_feature_set.pipeline_version
        == "m3-v1"
    )