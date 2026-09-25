from functools import lru_cache

from app.core.config import settings
from app.services.transcription.base import TranscriptionProvider
from app.services.transcription.local_whisper import (
    LocalWhisperTranscriptionProvider,
)
from app.services.transcription.mock import MockTranscriptionProvider
from app.services.transcription.openai_provider import (
    OpenAITranscriptionProvider,
)


@lru_cache(maxsize=1)
def _get_local_whisper_provider(
    model_name: str,
    device: str,
    compute_type: str,
    download_root: str,
) -> LocalWhisperTranscriptionProvider:
    """
    Return one LocalWhisperTranscriptionProvider per worker process
    for a given configuration.

    Celery prefork workers do not share Python memory, so this cache
    is intentionally process-local.
    """
    return LocalWhisperTranscriptionProvider(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        download_root=download_root,
    )


def get_transcription_provider() -> TranscriptionProvider:
    provider = settings.transcription_provider.strip().lower()

    if provider == "mock":
        return MockTranscriptionProvider()

    if provider == "local_whisper":
        return _get_local_whisper_provider(
            model_name=settings.local_whisper_model,
            device=settings.local_whisper_device,
            compute_type=settings.local_whisper_compute_type,
            download_root=settings.local_whisper_download_root,
        )

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when "
                "TRANSCRIPTION_PROVIDER=openai"
            )

        return OpenAITranscriptionProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_transcription_model,
        )

    raise ValueError(
        f"Unsupported transcription provider: {provider}"
    )