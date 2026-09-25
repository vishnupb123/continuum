from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    duration_seconds: float | None = None
    language: str | None = None


class TranscriptionProvider(ABC):
    @abstractmethod
    def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str,
    ) -> TranscriptionResult:
        """
        Convert audio bytes into normalized transcription output.

        Implementations must either return a TranscriptionResult
        or raise an exception when transcription cannot be completed.
        """
        raise NotImplementedError