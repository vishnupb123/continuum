from io import BytesIO

from openai import OpenAI

from app.services.transcription.base import (
    TranscriptionProvider,
    TranscriptionResult,
)


class OpenAITranscriptionProvider(
    TranscriptionProvider
):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-transcribe",
    ):
        if not api_key or not api_key.strip():
            raise ValueError(
                "OpenAI API key is required"
            )

        self.client = OpenAI(
            api_key=api_key
        )

        self.model = model

    def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str,
    ) -> TranscriptionResult:
        if not isinstance(
            audio_bytes,
            bytes,
        ):
            raise TypeError(
                "audio_bytes must be bytes"
            )

        if not audio_bytes:
            raise ValueError(
                "Audio cannot be empty"
            )

        if not mime_type:
            raise ValueError(
                "MIME type is required"
            )

        extension = self._extension_for_mime(
            mime_type
        )

        audio_file = BytesIO(
            audio_bytes
        )

        audio_file.name = (
            f"continuum-audio.{extension}"
        )

        response = (
            self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
            )
        )

        text = (
            response.text or ""
        ).strip()

        if not text:
            raise ValueError(
                "Transcription provider "
                "returned an empty transcript"
            )

        return TranscriptionResult(
            text=text,
            duration_seconds=None,
            language=None,
        )

    @staticmethod
    def _extension_for_mime(
        mime_type: str,
    ) -> str:
        base_mime = (
            mime_type
            .lower()
            .split(";", 1)[0]
            .strip()
        )

        extensions = {
            "audio/webm": "webm",
            "audio/mpeg": "mp3",
            "audio/mp3": "mp3",
            "audio/mp4": "m4a",
            "audio/m4a": "m4a",
            "audio/wav": "wav",
            "audio/x-wav": "wav",
        }

        extension = extensions.get(
            base_mime
        )

        if extension is None:
            raise ValueError(
                "Unsupported audio MIME type"
            )

        return extension