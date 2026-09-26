import hashlib

import numpy as np

import hashlib

import numpy as np

from app.services.features.audio_encoder import (
    AUDIO_ENCODER_DIMENSION,
    AUDIO_ENCODER_SAMPLE_RATE_HZ,
    AUDIO_ENCODER_VERSION,
    AUDIO_NORMALIZATION,
    AUDIO_POOLING_STRATEGY,
    AudioEncodingResult,
    l2_normalize_embedding,
    validate_audio_encoder_input,
)


MOCK_AUDIO_ENCODER_NAME = "mock-audio-encoder"
MOCK_AUDIO_ENCODER_REVISION = "deterministic-v1"


class MockAudioEncoder:
    """
    Deterministic test encoder implementing the production contract.

    This is NOT intended to approximate WavLM semantics.

    It exists so orchestration and persistence can be tested without:
    - downloading model weights
    - loading PyTorch models
    - requiring Hugging Face
    - introducing network access into unit tests
    """

    @property
    def encoder_name(self) -> str:
        return MOCK_AUDIO_ENCODER_NAME

    @property
    def encoder_version(self) -> str:
        return AUDIO_ENCODER_VERSION

    @property
    def encoder_revision(self) -> str:
        return MOCK_AUDIO_ENCODER_REVISION

    @property
    def embedding_dimension(self) -> int:
        return AUDIO_ENCODER_DIMENSION

    def encode(
        self,
        waveform: np.ndarray,
        *,
        sample_rate_hz: int,
    ) -> AudioEncodingResult:
        validate_audio_encoder_input(
            waveform,
            sample_rate_hz=sample_rate_hz,
        )

        digest = hashlib.sha256(
            waveform.tobytes()
        ).digest()

        seed = int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        )

        rng = np.random.default_rng(
            seed
        )

        raw_embedding = rng.standard_normal(
            AUDIO_ENCODER_DIMENSION
        )

        embedding = l2_normalize_embedding(
            raw_embedding
        )

        return AudioEncodingResult(
            embedding=embedding,
            encoder_name=self.encoder_name,
            encoder_version=self.encoder_version,
            encoder_revision=self.encoder_revision,
            embedding_dimension=(
                self.embedding_dimension
            ),
            sample_rate_hz=(
                AUDIO_ENCODER_SAMPLE_RATE_HZ
            ),
            pooling_strategy=(
                AUDIO_POOLING_STRATEGY
            ),
            normalization=(
                AUDIO_NORMALIZATION
            ),
        )