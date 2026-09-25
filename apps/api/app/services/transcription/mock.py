from app.services.transcription.base import (
    TranscriptionProvider,
    TranscriptionResult,
)


class MockTranscriptionProvider(TranscriptionProvider):
    def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str,
    ) -> TranscriptionResult:
        if not isinstance(audio_bytes, bytes):
            raise TypeError("audio_bytes must be bytes")

        if not audio_bytes:
            raise ValueError("Audio cannot be empty")

        if not mime_type or not mime_type.strip():
            raise ValueError("mime_type cannot be empty")

        return TranscriptionResult(
            text=(
                "This is a deterministic mock transcription "
                "for Continuum."
            ),
            duration_seconds=None,
            language="en",
        )