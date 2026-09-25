import os
import tempfile

from faster_whisper import WhisperModel

from app.services.transcription.base import (
    TranscriptionProvider,
    TranscriptionResult,
)


class LocalWhisperTranscriptionProvider(
    TranscriptionProvider
):
    def __init__(
        self,
        model_name: str = "small.en",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root

        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )

    def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str,
    ) -> TranscriptionResult:
        if not isinstance(audio_bytes, bytes):
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

        temp_path: str | None = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=f".{extension}",
                delete=False,
            ) as temp_file:
                temp_file.write(audio_bytes)
                temp_file.flush()

                temp_path = temp_file.name

            segments, info = (
                self.model.transcribe(
                    temp_path,
                    beam_size=5,
                    vad_filter=True,
                )
            )

            text_parts: list[str] = []

            for segment in segments:
                text = (
                    segment.text or ""
                ).strip()

                if text:
                    text_parts.append(text)

            transcript = " ".join(
                text_parts
            ).strip()

            if not transcript:
                raise ValueError(
                    "Transcription provider "
                    "returned an empty transcript"
                )

            duration = getattr(
                info,
                "duration",
                None,
            )

            language = getattr(
                info,
                "language",
                None,
            )

            return TranscriptionResult(
                text=transcript,
                duration_seconds=duration,
                language=language,
            )

        finally:
            if (
                temp_path is not None
                and os.path.exists(temp_path)
            ):
                os.remove(temp_path)

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