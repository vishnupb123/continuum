from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


AUDIO_ENCODER_VERSION = "audio-encoder-v1"

AUDIO_ENCODER_DIMENSION = 768
AUDIO_ENCODER_SAMPLE_RATE_HZ = 16_000

AUDIO_POOLING_STRATEGY = "masked_temporal_mean"
AUDIO_NORMALIZATION = "l2"


@dataclass(frozen=True)
class AudioEncodingResult:
    """
    Journal-level representation produced by a learned audio encoder.

    This contract contains representation metadata only.

    It does not contain:
    - transcription
    - emotion labels
    - speaker identity
    - psychological conclusions
    - diagnostic conclusions
    """

    embedding: np.ndarray

    encoder_name: str
    encoder_version: str
    encoder_revision: str | None

    embedding_dimension: int
    sample_rate_hz: int

    pooling_strategy: str
    normalization: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.embedding,
            np.ndarray,
        ):
            raise TypeError(
                "embedding must be a numpy array"
            )

        if self.embedding.dtype != np.float32:
            raise ValueError(
                "embedding must have dtype float32"
            )

        if self.embedding.ndim != 1:
            raise ValueError(
                "embedding must be one-dimensional"
            )

        if (
            self.embedding_dimension
            != AUDIO_ENCODER_DIMENSION
        ):
            raise ValueError(
                "embedding_dimension must be 768"
            )

        if self.embedding.shape != (
            self.embedding_dimension,
        ):
            raise ValueError(
                "embedding shape does not match "
                "embedding_dimension"
            )

        if not np.all(
            np.isfinite(self.embedding)
        ):
            raise ValueError(
                "embedding must contain only finite values"
            )

        if (
            self.sample_rate_hz
            != AUDIO_ENCODER_SAMPLE_RATE_HZ
        ):
            raise ValueError(
                "audio encoder requires 16 kHz audio"
            )

        if (
            self.pooling_strategy
            != AUDIO_POOLING_STRATEGY
        ):
            raise ValueError(
                "unsupported pooling strategy"
            )

        if (
            self.normalization
            != AUDIO_NORMALIZATION
        ):
            raise ValueError(
                "unsupported normalization"
            )

        norm = float(
            np.linalg.norm(
                self.embedding.astype(
                    np.float64
                )
            )
        )

        if not np.isclose(
            norm,
            1.0,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError(
                "embedding must be L2 normalized"
            )


class AudioEncoder(Protocol):
    """
    Stable interface consumed by the feature pipeline.

    M3.6C+ can provide WavLM without exposing Hugging Face
    implementation details to the rest of Continuum.
    """

    @property
    def encoder_name(self) -> str:
        ...

    @property
    def encoder_version(self) -> str:
        ...

    @property
    def encoder_revision(self) -> str | None:
        ...

    @property
    def embedding_dimension(self) -> int:
        ...

    def encode(
        self,
        waveform: np.ndarray,
        *,
        sample_rate_hz: int,
    ) -> AudioEncodingResult:
        ...


def validate_audio_encoder_input(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int,
) -> None:
    """
    Validate the canonical waveform expected by all M3.6 encoders.
    """

    if not isinstance(
        waveform,
        np.ndarray,
    ):
        raise TypeError(
            "waveform must be a numpy array"
        )

    if waveform.dtype != np.float32:
        raise ValueError(
            "waveform must have dtype float32"
        )

    if waveform.ndim != 1:
        raise ValueError(
            "waveform must be one-dimensional"
        )

    if waveform.size == 0:
        raise ValueError(
            "waveform cannot be empty"
        )

    if not np.all(
        np.isfinite(waveform)
    ):
        raise ValueError(
            "waveform must contain only finite values"
        )

    if (
        sample_rate_hz
        != AUDIO_ENCODER_SAMPLE_RATE_HZ
    ):
        raise ValueError(
            "audio encoder requires 16 kHz audio"
        )


def l2_normalize_embedding(
    embedding: np.ndarray,
) -> np.ndarray:
    """
    Convert a one-dimensional representation to float32 and
    L2-normalize it.

    A zero vector is invalid because it cannot represent a valid
    cosine-comparable embedding.
    """

    if not isinstance(
        embedding,
        np.ndarray,
    ):
        raise TypeError(
            "embedding must be a numpy array"
        )

    if embedding.ndim != 1:
        raise ValueError(
            "embedding must be one-dimensional"
        )

    if not np.all(
        np.isfinite(embedding)
    ):
        raise ValueError(
            "embedding must contain only finite values"
        )

    vector = embedding.astype(
        np.float64,
        copy=False,
    )

    norm = float(
        np.linalg.norm(vector)
    )

    if norm <= 0.0:
        raise ValueError(
            "cannot normalize a zero embedding"
        )

    normalized = (
        vector / norm
    ).astype(
        np.float32
    )

    return normalized