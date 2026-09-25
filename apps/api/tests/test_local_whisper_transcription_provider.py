import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from app.services.transcription.local_whisper import (
    LocalWhisperTranscriptionProvider,
)


def make_provider():
    mock_model = Mock()

    with patch(
        "app.services.transcription."
        "local_whisper.WhisperModel",
        return_value=mock_model,
    ) as mocked_whisper:
        provider = (
            LocalWhisperTranscriptionProvider(
                model_name="small.en",
                device="cpu",
                compute_type="int8",
                download_root="/models",
            )
        )

    return (
        provider,
        mock_model,
        mocked_whisper,
    )


def test_model_is_initialized_with_configuration():
    (
        provider,
        mock_model,
        mocked_whisper,
    ) = make_provider()

    mocked_whisper.assert_called_once_with(
        "small.en",
        device="cpu",
        compute_type="int8",
        download_root="/models",
    )


def test_transcribes_webm_audio():
    provider, mock_model, _ = (
        make_provider()
    )

    mock_model.transcribe.return_value = (
        iter(
            [
                SimpleNamespace(
                    text=" Hello"
                ),
                SimpleNamespace(
                    text=" Continuum. "
                ),
            ]
        ),
        SimpleNamespace(
            duration=12.5,
            language="en",
        ),
    )

    result = provider.transcribe(
        audio_bytes=b"fake-webm-audio",
        mime_type="audio/webm",
    )

    assert (
        result.text
        == "Hello Continuum."
    )

    assert (
        result.duration_seconds
        == 12.5
    )

    assert result.language == "en"

    mock_model.transcribe.assert_called_once()

    args = (
        mock_model
        .transcribe
        .call_args
        .args
    )

    kwargs = (
        mock_model
        .transcribe
        .call_args
        .kwargs
    )

    temp_path = args[0]

    assert temp_path.endswith(
        ".webm"
    )

    assert kwargs["beam_size"] == 5
    assert kwargs["vad_filter"] is True

    # Provider must remove temporary audio
    # after transcription.
    assert not os.path.exists(
        temp_path
    )


def test_parameterized_webm_is_supported():
    provider, mock_model, _ = (
        make_provider()
    )

    mock_model.transcribe.return_value = (
        iter(
            [
                SimpleNamespace(
                    text="Browser recording"
                )
            ]
        ),
        SimpleNamespace(
            duration=5.0,
            language="en",
        ),
    )

    result = provider.transcribe(
        audio_bytes=b"audio",
        mime_type=(
            "audio/webm;codecs=opus"
        ),
    )

    assert (
        result.text
        == "Browser recording"
    )

    temp_path = (
        mock_model
        .transcribe
        .call_args
        .args[0]
    )

    assert temp_path.endswith(
        ".webm"
    )


@pytest.mark.parametrize(
    ("mime_type", "extension"),
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
def test_supported_mime_extensions(
    mime_type,
    extension,
):
    provider, mock_model, _ = (
        make_provider()
    )

    mock_model.transcribe.return_value = (
        iter(
            [
                SimpleNamespace(
                    text="Transcript"
                )
            ]
        ),
        SimpleNamespace(
            duration=1.0,
            language="en",
        ),
    )

    provider.transcribe(
        audio_bytes=b"audio",
        mime_type=mime_type,
    )

    temp_path = (
        mock_model
        .transcribe
        .call_args
        .args[0]
    )

    assert temp_path.endswith(
        f".{extension}"
    )


def test_empty_audio_is_rejected():
    provider, mock_model, _ = (
        make_provider()
    )

    with pytest.raises(
        ValueError,
        match="Audio cannot be empty",
    ):
        provider.transcribe(
            audio_bytes=b"",
            mime_type="audio/webm",
        )

    mock_model.transcribe.assert_not_called()


def test_non_bytes_audio_is_rejected():
    provider, mock_model, _ = (
        make_provider()
    )

    with pytest.raises(
        TypeError,
        match="audio_bytes must be bytes",
    ):
        provider.transcribe(
            audio_bytes="audio",
            mime_type="audio/webm",
        )

    mock_model.transcribe.assert_not_called()


def test_missing_mime_is_rejected():
    provider, mock_model, _ = (
        make_provider()
    )

    with pytest.raises(
        ValueError,
        match="MIME type is required",
    ):
        provider.transcribe(
            audio_bytes=b"audio",
            mime_type="",
        )

    mock_model.transcribe.assert_not_called()


def test_unsupported_mime_is_rejected():
    provider, mock_model, _ = (
        make_provider()
    )

    with pytest.raises(
        ValueError,
        match="Unsupported audio MIME type",
    ):
        provider.transcribe(
            audio_bytes=b"audio",
            mime_type="audio/ogg",
        )

    mock_model.transcribe.assert_not_called()


def test_empty_transcript_is_rejected():
    provider, mock_model, _ = (
        make_provider()
    )

    mock_model.transcribe.return_value = (
        iter(
            [
                SimpleNamespace(
                    text="   "
                )
            ]
        ),
        SimpleNamespace(
            duration=1.0,
            language="en",
        ),
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