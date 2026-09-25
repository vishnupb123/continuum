from abc import ABC, abstractmethod


class AudioStorage(ABC):
    """Storage contract for private Continuum audio objects."""

    @abstractmethod
    def put(self, key: str, data: bytes) -> str:
        """Store audio bytes and return the storage key."""
        raise NotImplementedError

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Retrieve an audio object."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete an audio object."""
        raise NotImplementedError

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return whether an audio object exists."""
        raise NotImplementedError