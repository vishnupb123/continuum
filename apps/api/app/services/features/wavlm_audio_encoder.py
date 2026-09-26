from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    WavLMModel,
)

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


WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"

WAVLM_MODEL_REVISION = (
    "4c66d4806a428f2e922ccfa1a962776e232d487b"
)

WAVLM_HIDDEN_SIZE = 768

WAVLM_DEVICE_CPU = "cpu"


@lru_cache(maxsize=4)
def _load_wavlm_components(
    model_name: str,
    revision: str,
    device: str,
    cache_dir: str | None,
) -> tuple[
    Wav2Vec2FeatureExtractor,
    WavLMModel,
]:
    """
    Load and cache the immutable WavLM feature extractor and model.

    The cache is process-local. A Celery worker therefore loads a
    particular model/revision/device combination only once.
    """

    if device != WAVLM_DEVICE_CPU:
        raise ValueError(
            "M3.6C currently supports CPU inference only"
        )

    feature_extractor = (
        Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
        )
    )

    model = WavLMModel.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=cache_dir,
    )

    model.eval()
    model.to(device)

    if (
        model.config.hidden_size
        != WAVLM_HIDDEN_SIZE
    ):
        raise RuntimeError(
            "unexpected WavLM hidden dimension: "
            f"{model.config.hidden_size}"
        )

    return (
        feature_extractor,
        model,
    )


class WavLMAudioEncoder:
    """
    Frozen WavLM Base+ journal-level audio encoder.

    Contract:

        canonical 16 kHz float32 waveform
                    |
                    v
              WavLM Base+
                    |
                    v
        last_hidden_state [1, T, 768]
                    |
                    v
          temporal mean pooling
                    |
                    v
                 [768]
                    |
                    v
            L2 normalization
                    |
                    v
                 [768]

    This adapter performs representation extraction only.

    It does not perform:
    - transcription
    - VAD
    - emotion classification
    - speaker classification
    - psychological inference
    - diagnostic inference
    """

    def __init__(
        self,
        *,
        model_name: str = WAVLM_MODEL_NAME,
        revision: str = WAVLM_MODEL_REVISION,
        device: str = WAVLM_DEVICE_CPU,
        cache_dir: str | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError(
                "model_name cannot be empty"
            )

        if not revision.strip():
            raise ValueError(
                "revision cannot be empty"
            )

        if device != WAVLM_DEVICE_CPU:
            raise ValueError(
                "M3.6C currently supports CPU inference only"
            )

        self._model_name = model_name
        self._revision = revision
        self._device = device
        self._cache_dir = cache_dir

    @property
    def encoder_name(self) -> str:
        return self._model_name

    @property
    def encoder_version(self) -> str:
        return AUDIO_ENCODER_VERSION

    @property
    def encoder_revision(self) -> str:
        return self._revision

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

        (
            feature_extractor,
            model,
        ) = _load_wavlm_components(
            self._model_name,
            self._revision,
            self._device,
            self._cache_dir,
        )

        # Copy intentionally:
        # the external feature extractor must never be allowed
        # to mutate Continuum's canonical waveform.
        waveform_input = waveform.copy()

        inputs = feature_extractor(
            waveform_input,
            sampling_rate=sample_rate_hz,
            return_tensors="pt",
            padding=False,
        )

        input_values = inputs[
            "input_values"
        ].to(
            self._device
        )

        with torch.inference_mode():
            outputs = model(
                input_values=input_values,
            )

        hidden_state = (
            outputs.last_hidden_state
        )

        if hidden_state.ndim != 3:
            raise RuntimeError(
                "WavLM last_hidden_state must "
                "be three-dimensional"
            )

        if hidden_state.shape[0] != 1:
            raise RuntimeError(
                "WavLM adapter expects exactly "
                "one waveform"
            )

        if (
            hidden_state.shape[2]
            != WAVLM_HIDDEN_SIZE
        ):
            raise RuntimeError(
                "unexpected WavLM output dimension"
            )

        if hidden_state.shape[1] == 0:
            raise RuntimeError(
                "WavLM produced no temporal frames"
            )

        # M3.6 currently encodes exactly one unpadded waveform.
        #
        # Therefore every emitted WavLM temporal position is valid
        # and temporal mean is mathematically identical to masked
        # temporal mean with an all-ones mask.
        #
        # When batched/padded inference is introduced, this boundary
        # must construct and apply a feature-vector attention mask.
        pooled = hidden_state.mean(
            dim=1
        )

        embedding = (
            pooled[0]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        normalized_embedding = (
            l2_normalize_embedding(
                embedding
            )
        )

        return AudioEncodingResult(
            embedding=normalized_embedding,
            encoder_name=self.encoder_name,
            encoder_version=(
                self.encoder_version
            ),
            encoder_revision=(
                self.encoder_revision
            ),
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