from functools import lru_cache

from app.core.config import settings
from app.services.features.text_encoder.base import (
    TextEncoder,
)
from app.services.features.text_encoder.mock import (
    MockTextEncoder,
)
from app.services.features.text_encoder.mpnet import (
    MPNetTextEncoder,
)


@lru_cache(maxsize=1)
def _get_mpnet_encoder(
    model_name: str,
    revision: str | None,
    device: str,
    cache_folder: str,
) -> MPNetTextEncoder:
    return MPNetTextEncoder(
        model_name=model_name,
        revision=revision,
        device=device,
        cache_folder=cache_folder,
    )


@lru_cache(maxsize=1)
def _get_mock_encoder() -> MockTextEncoder:
    return MockTextEncoder()


def get_text_encoder() -> TextEncoder:
    provider = (
        settings.text_encoder_provider
        .strip()
        .lower()
    )

    if provider == "mock":
        return _get_mock_encoder()

    if provider == "mpnet":
        revision = (
            settings.text_encoder_revision.strip()
            if settings.text_encoder_revision
            else None
        )

        return _get_mpnet_encoder(
            settings.text_encoder_model,
            revision,
            settings.text_encoder_device,
            settings.text_encoder_cache_path,
        )

    raise ValueError(
        f"Unsupported text encoder provider: "
        f"{provider}"
    )