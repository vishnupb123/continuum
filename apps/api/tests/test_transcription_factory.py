from unittest.mock import patch

import pytest

from app.services.transcription.factory import (
    _get_local_whisper_provider,
    get_transcription_provider,
)
from app.services.transcription.mock import (
    MockTranscriptionProvider,
)
from app.services.transcription.openai_provider import (
    OpenAITranscriptionProvider,
)


def test_factory_returns_mock_provider():
    with patch(
        "app.services.transcription.factory."
        "settings.transcription_provider",
        "mock",
    ):
        provider = get_transcription_provider()

    assert isinstance(
        provider,
        MockTranscriptionProvider,
    )


def test_factory_normalizes_provider_name():
    with patch(
        "app.services.transcription.factory."
        "settings.transcription_provider",
        "  MOCK  ",
    ):
        provider = get_transcription_provider()

    assert isinstance(
        provider,
        MockTranscriptionProvider,
    )


def test_factory_returns_openai_provider():
    with (
        patch(
            "app.services.transcription.factory."
            "settings.transcription_provider",
            "openai",
        ),
        patch(
            "app.services.transcription.factory."
            "settings.openai_api_key",
            "test-api-key",
        ),
        patch(
            "app.services.transcription.factory."
            "settings.openai_transcription_model",
            "gpt-transcribe",
        ),
        patch(
            "app.services.transcription.factory."
            "OpenAITranscriptionProvider"
        ) as mocked_provider,
    ):
        provider = get_transcription_provider()

    mocked_provider.assert_called_once_with(
        api_key="test-api-key",
        model="gpt-transcribe",
    )

    assert (
        provider
        == mocked_provider.return_value
    )


def test_openai_provider_requires_api_key():
    with (
        patch(
            "app.services.transcription.factory."
            "settings.transcription_provider",
            "openai",
        ),
        patch(
            "app.services.transcription.factory."
            "settings.openai_api_key",
            None,
        ),
    ):
        with pytest.raises(
            ValueError,
            match="OPENAI_API_KEY is required",
        ):
            get_transcription_provider()


def test_openai_provider_rejects_empty_api_key():
    with (
        patch(
            "app.services.transcription.factory."
            "settings.transcription_provider",
            "openai",
        ),
        patch(
            "app.services.transcription.factory."
            "settings.openai_api_key",
            "",
        ),
    ):
        with pytest.raises(
            ValueError,
            match="OPENAI_API_KEY is required",
        ):
            get_transcription_provider()


def test_factory_forwards_configured_model():
    with (
        patch(
            "app.services.transcription.factory."
            "settings.transcription_provider",
            "openai",
        ),
        patch(
            "app.services.transcription.factory."
            "settings.openai_api_key",
            "test-api-key",
        ),
        patch(
            "app.services.transcription.factory."
            "settings.openai_transcription_model",
            "another-model",
        ),
        patch(
            "app.services.transcription.factory."
            "OpenAITranscriptionProvider"
        ) as mocked_provider,
    ):
        get_transcription_provider()

    mocked_provider.assert_called_once_with(
        api_key="test-api-key",
        model="another-model",
    )


def test_factory_rejects_unknown_provider():
    with patch(
        "app.services.transcription.factory."
        "settings.transcription_provider",
        "unknown-provider",
    ):
        with pytest.raises(
            ValueError,
            match=(
                "Unsupported transcription provider"
            ),
        ):
            get_transcription_provider()
            
def test_factory_returns_local_whisper_provider():
    _get_local_whisper_provider.cache_clear()

    try:
        with (
            patch(
                "app.services.transcription.factory."
                "settings.transcription_provider",
                "local_whisper",
            ),
            patch(
                "app.services.transcription.factory."
                "settings.local_whisper_model",
                "small.en",
            ),
            patch(
                "app.services.transcription.factory."
                "settings.local_whisper_device",
                "cpu",
            ),
            patch(
                "app.services.transcription.factory."
                "settings.local_whisper_compute_type",
                "int8",
            ),
            patch(
                "app.services.transcription.factory."
                "settings.local_whisper_download_root",
                "/models",
            ),
            patch(
                "app.services.transcription.factory."
                "LocalWhisperTranscriptionProvider"
            ) as mocked_provider,
        ):
            provider = get_transcription_provider()

        mocked_provider.assert_called_once_with(
            model_name="small.en",
            device="cpu",
            compute_type="int8",
            download_root="/models",
        )

        assert provider == mocked_provider.return_value

    finally:
        _get_local_whisper_provider.cache_clear()
        

def test_local_whisper_provider_is_reused():
    _get_local_whisper_provider.cache_clear()

    try:
        with (
            patch(
                "app.services.transcription.factory."
                "settings.transcription_provider",
                "local_whisper",
            ),
            patch(
                "app.services.transcription.factory."
                "settings.local_whisper_model",
                "small.en",
            ),
            patch(
                "app.services.transcription.factory."
                "settings.local_whisper_device",
                "cpu",
            ),
            patch(
                "app.services.transcription.factory."
                "settings.local_whisper_compute_type",
                "int8",
            ),
            patch(
                "app.services.transcription.factory."
                "settings.local_whisper_download_root",
                "/models",
            ),
            patch(
                "app.services.transcription.factory."
                "LocalWhisperTranscriptionProvider"
            ) as mocked_provider,
        ):
            first = get_transcription_provider()
            second = get_transcription_provider()
            third = get_transcription_provider()

        assert first is second
        assert second is third

        mocked_provider.assert_called_once_with(
            model_name="small.en",
            device="cpu",
            compute_type="int8",
            download_root="/models",
        )

    finally:
        _get_local_whisper_provider.cache_clear()