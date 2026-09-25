from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from app.services.transcription.openai_provider import (
    OpenAITranscriptionProvider,
)


def make_provider(
    model: str = "gpt-transcribe",
):
    mock_client = Mock()

    with patch(
        "app.services.transcription.openai_provider.OpenAI",
        return_value=mock_client,
    ):
        provider = OpenAITranscriptionProvider(
            api_key="test-api-key",
            model=model,
        )

    return provider, mock_client


def test_transcribes_webm_audio():
    provider, mock_client = make_provider()

    mock_client.audio.transcriptions.create.return_value = (
        SimpleNamespace(
            text="This is a real transcript."
        )
    )

    result = provider.transcribe(
        audio_bytes=b"fake-webm-bytes",
        mime_type="audio/webm",
    )

    assert (
        result.text
        == "This is a real transcript."
    )

    assert result.duration_seconds is None
    assert result.language is None

    mock_client.audio.transcriptions.create.assert_called_once()

    call_kwargs = (
        mock_client
        .audio
        .transcriptions
        .create
        .call_args
        .kwargs
    )

    assert (
        call_kwargs["model"]
        == "gpt-transcribe"
    )

    uploaded_file = call_kwargs["file"]

    assert (
        uploaded_file.name
        == "continuum-audio.webm"
    )

    assert (
        uploaded_file.getvalue()
        == b"fake-webm-bytes"
    )


def test_parameterized_webm_mime_type_is_supported():
    provider, mock_client = make_provider()

    mock_client.audio.transcriptions.create.return_value = (
        SimpleNamespace(
            text="Browser recording transcript"
        )
    )

    result = provider.transcribe(
        audio_bytes=b"browser-audio",
        mime_type=(
            "audio/webm;codecs=opus"
        ),
    )

    assert (
        result.text
        == "Browser recording transcript"
    )

    call_kwargs = (
        mock_client
        .audio
        .transcriptions
        .create
        .call_args
        .kwargs
    )

    assert (
        call_kwargs["file"].name
        == "continuum-audio.webm"
    )


@pytest.mark.parametrize(
    ("mime_type", "expected_extension"),
    [
        ("audio/webm", "webm"),
        ("audio/mpeg", "mp3"),
        ("audio/mp3", "mp3"),
        ("audio/mp4", "m4a"),
        ("audio/m4a", "m4a"),
        ("audio/wav", "wav"),
        ("audio/x-wav", "wav"),
    ],
)
def test_supported_mime_types_use_correct_extension(
    mime_type,
    expected_extension,
):
    provider, mock_client = make_provider()

    mock_client.audio.transcriptions.create.return_value = (
        SimpleNamespace(
            text="Transcript"
        )
    )

    provider.transcribe(
        audio_bytes=b"audio",
        mime_type=mime_type,
    )

    call_kwargs = (
        mock_client
        .audio
        .transcriptions
        .create
        .call_args
        .kwargs
    )

    assert (
        call_kwargs["file"].name
        == (
            "continuum-audio."
            f"{expected_extension}"
        )
    )


def test_custom_model_is_forwarded():
    provider, mock_client = make_provider(
        model="custom-transcription-model"
    )

    mock_client.audio.transcriptions.create.return_value = (
        SimpleNamespace(
            text="Transcript"
        )
    )

    provider.transcribe(
        audio_bytes=b"audio",
        mime_type="audio/webm",
    )

    call_kwargs = (
        mock_client
        .audio
        .transcriptions
        .create
        .call_args
        .kwargs
    )

    assert (
        call_kwargs["model"]
        == "custom-transcription-model"
    )


def test_transcript_is_trimmed():
    provider, mock_client = make_provider()

    mock_client.audio.transcriptions.create.return_value = (
        SimpleNamespace(
            text="   Hello Continuum.   "
        )
    )

    result = provider.transcribe(
        audio_bytes=b"audio",
        mime_type="audio/webm",
    )

    assert result.text == "Hello Continuum."


def test_empty_audio_is_rejected_before_api_call():
    provider, mock_client = make_provider()

    with pytest.raises(
        ValueError,
        match="Audio cannot be empty",
    ):
        provider.transcribe(
            audio_bytes=b"",
            mime_type="audio/webm",
        )

    (
        mock_client
        .audio
        .transcriptions
        .create
        .assert_not_called()
    )


def test_non_bytes_audio_is_rejected():
    provider, mock_client = make_provider()

    with pytest.raises(
        TypeError,
        match="audio_bytes must be bytes",
    ):
        provider.transcribe(
            audio_bytes="not-bytes",
            mime_type="audio/webm",
        )

    (
        mock_client
        .audio
        .transcriptions
        .create
        .assert_not_called()
    )


def test_missing_mime_type_is_rejected():
    provider, mock_client = make_provider()

    with pytest.raises(
        ValueError,
        match="MIME type is required",
    ):
        provider.transcribe(
            audio_bytes=b"audio",
            mime_type="",
        )

    (
        mock_client
        .audio
        .transcriptions
        .create
        .assert_not_called()
    )


def test_unsupported_mime_type_is_rejected():
    provider, mock_client = make_provider()

    with pytest.raises(
        ValueError,
        match="Unsupported audio MIME type",
    ):
        provider.transcribe(
            audio_bytes=b"audio",
            mime_type="audio/ogg",
        )

    (
        mock_client
        .audio
        .transcriptions
        .create
        .assert_not_called()
    )


def test_empty_provider_transcript_is_rejected():
    provider, mock_client = make_provider()

    mock_client.audio.transcriptions.create.return_value = (
        SimpleNamespace(
            text="   "
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Transcription provider returned "
            "an empty transcript"
        ),
    ):
        provider.transcribe(
            audio_bytes=b"audio",
            mime_type="audio/webm",
        )


def test_api_key_is_required():
    with pytest.raises(
        ValueError,
        match="OpenAI API key is required",
    ):
        OpenAITranscriptionProvider(
            api_key=""
        )