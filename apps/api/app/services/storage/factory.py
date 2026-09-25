from app.core.config import settings
from app.services.storage.base import AudioStorage
from app.services.storage.local import LocalAudioStorage


def get_audio_storage() -> AudioStorage:
    if settings.audio_storage_backend == "local":
        return LocalAudioStorage(settings.audio_storage_path)

    raise ValueError(
        f"Unsupported audio storage backend: "
        f"{settings.audio_storage_backend}"
    )