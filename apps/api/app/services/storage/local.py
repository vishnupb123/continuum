from pathlib import Path

from app.services.storage.base import AudioStorage


class LocalAudioStorage(AudioStorage):
    def __init__(self, root_path: str):
        self.root = Path(root_path).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve_key(self, key: str) -> Path:
        if not key or key.strip() == "":
            raise ValueError("Storage key cannot be empty")

        relative = Path(key)

        if relative.is_absolute():
            raise ValueError("Absolute storage keys are not allowed")

        target = (self.root / relative).resolve()

        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Invalid storage key") from exc

        return target

    def put(self, key: str, data: bytes) -> str:
        if not isinstance(data, bytes):
            raise TypeError("Audio data must be bytes")

        path = self._resolve_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_bytes(data)

        return key

    def get(self, key: str) -> bytes:
        path = self._resolve_key(key)

        if not path.is_file():
            raise FileNotFoundError(f"Audio object not found: {key}")

        return path.read_bytes()

    def delete(self, key: str) -> None:
        path = self._resolve_key(key)

        if path.is_file():
            path.unlink()

    def exists(self, key: str) -> bool:
        return self._resolve_key(key).is_file()