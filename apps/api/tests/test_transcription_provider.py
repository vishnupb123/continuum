import pytest

from app.core.config import settings
from app.services.transcription.base import (
    TranscriptionProvider,
    TranscriptionResult,
)
from app.services.transcription.factory import (
    get_transcription_provider,
)
from app.services.transcription.mock import (
    MockTranscriptionProvider,
)


def test_mock_provider_implements_transcription_contract():
    provider = MockTranscriptionProvider()

    assert isinstance(provider, TranscriptionProvider)


def test_mock_provider_returns_transcription_result():
    provider = MockTranscriptionProvider()

    result = provider.transcribe(
        audio_bytes=b"fake-audio",
        mime_type="audio/webm",
    )

    assert isinstance(result, TranscriptionResult)

    assert result.text == (
        "This is a deterministic mock transcription "
        "for Continuum."
    )

    assert result.duration_seconds is None
    assert result.language == "en"


def test_mock_provider_is_deterministic():
    provider = MockTranscriptionProvider()

    first = provider.transcribe(
        audio_bytes=b"first-audio",
        mime_type="audio/webm",
    )

    second = provider.transcribe(
        audio_bytes=b"different-audio",
        mime_type="audio/mpeg",
    )

    assert first == second


def test_mock_provider_rejects_empty_audio():
    provider = MockTranscriptionProvider()

    with pytest.raises(
        ValueError,
        match="Audio cannot be empty",
    ):
        provider.transcribe(
            audio_bytes=b"",
            mime_type="audio/webm",
        )


def test_mock_provider_rejects_non_bytes_audio():
    provider = MockTranscriptionProvider()

    with pytest.raises(
        TypeError,
        match="audio_bytes must be bytes",
    ):
        provider.transcribe(
            audio_bytes="not-bytes",  # type: ignore[arg-type]
            mime_type="audio/webm",
        )


def test_mock_provider_rejects_empty_mime_type():
    provider = MockTranscriptionProvider()

    with pytest.raises(
        ValueError,
        match="mime_type cannot be empty",
    ):
        provider.transcribe(
            audio_bytes=b"fake-audio",
            mime_type="",
        )


def test_factory_returns_mock_provider(monkeypatch):
    monkeypatch.setattr(
        settings,
        "transcription_provider",
        "mock",
    )

    provider = get_transcription_provider()

    assert isinstance(
        provider,
        MockTranscriptionProvider,
    )


def test_factory_normalizes_provider_name(monkeypatch):
    monkeypatch.setattr(
        settings,
        "transcription_provider",
        "  MOCK  ",
    )

    provider = get_transcription_provider()

    assert isinstance(
        provider,
        MockTranscriptionProvider,
    )


def test_factory_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(
        settings,
        "transcription_provider",
        "unknown-provider",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported transcription provider",
    ):
        get_transcription_provider()