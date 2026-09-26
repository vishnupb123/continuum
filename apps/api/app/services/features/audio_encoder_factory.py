from functools import lru_cache

from app.core.config import settings
from app.services.features.audio_encoder import (
    AudioEncoder,
)
from app.services.features.mock_audio_encoder import (
    MockAudioEncoder,
)
from app.services.features.wavlm_audio_encoder import (
    WavLMAudioEncoder,
)


AUDIO_ENCODER_PROVIDER_WAVLM = "wavlm"
AUDIO_ENCODER_PROVIDER_MOCK = "mock"


def _normalize_optional_string(
    value: str | None,
) -> str | None:
    if value is None:
        return None

    normalized = value.strip()

    if not normalized:
        return None

    return normalized


@lru_cache(maxsize=1)
def get_audio_encoder() -> AudioEncoder:
    """
    Return the process-level configured audio encoder.

    The application depends on the AudioEncoder contract rather than
    directly on WavLM or Hugging Face.
    """

    provider = (
        settings.audio_encoder_provider
        .strip()
        .lower()
    )

    if (
        provider
        == AUDIO_ENCODER_PROVIDER_MOCK
    ):
        return MockAudioEncoder()

    if (
        provider
        == AUDIO_ENCODER_PROVIDER_WAVLM
    ):
        revision = (
            _normalize_optional_string(
                settings.audio_encoder_revision
            )
        )

        if revision is None:
            raise ValueError(
                "AUDIO_ENCODER_REVISION is required "
                "for the wavlm provider"
            )

        model_name = (
            settings.audio_encoder_model
            .strip()
        )

        if not model_name:
            raise ValueError(
                "AUDIO_ENCODER_MODEL is required "
                "for the wavlm provider"
            )

        device = (
            settings.audio_encoder_device
            .strip()
            .lower()
        )

        cache_path = (
            _normalize_optional_string(
                settings.audio_encoder_cache_path
            )
        )

        return WavLMAudioEncoder(
            model_name=model_name,
            revision=revision,
            device=device,
            cache_dir=cache_path,
        )

    raise ValueError(
        "Unsupported audio encoder provider: "
        f"{provider}"
    )