from app.services.features.fingerprinting import (
    FINGERPRINT_VERSION,
    fingerprint_text_journal,
    fingerprint_voice_journal,
)


def test_text_fingerprint_is_deterministic():
    first = fingerprint_text_journal(
        "Today was productive."
    )
    second = fingerprint_text_journal(
        "Today was productive."
    )

    assert first == second
    assert len(first.value) == 64
    assert first.version == FINGERPRINT_VERSION


def test_text_fingerprint_changes_when_text_changes():
    first = fingerprint_text_journal(
        "Today was productive."
    )
    second = fingerprint_text_journal(
        "Today was productive!"
    )

    assert first.value != second.value


def test_text_fingerprint_preserves_exact_source_identity():
    first = fingerprint_text_journal(
        "Today was productive."
    )
    second = fingerprint_text_journal(
        " Today was productive. "
    )

    assert first.value != second.value


def test_voice_fingerprint_is_deterministic():
    first = fingerprint_voice_journal(
        transcript="This is my journal.",
        audio_bytes=b"audio-data",
    )
    second = fingerprint_voice_journal(
        transcript="This is my journal.",
        audio_bytes=b"audio-data",
    )

    assert first == second


def test_voice_fingerprint_changes_when_transcript_changes():
    first = fingerprint_voice_journal(
        transcript="Original transcript.",
        audio_bytes=b"same-audio",
    )
    second = fingerprint_voice_journal(
        transcript="Changed transcript.",
        audio_bytes=b"same-audio",
    )

    assert first.value != second.value


def test_voice_fingerprint_changes_when_audio_changes():
    first = fingerprint_voice_journal(
        transcript="Same transcript.",
        audio_bytes=b"audio-version-one",
    )
    second = fingerprint_voice_journal(
        transcript="Same transcript.",
        audio_bytes=b"audio-version-two",
    )

    assert first.value != second.value


def test_text_and_voice_sources_cannot_collide_by_content():
    text = fingerprint_text_journal(
        "identical-content"
    )

    voice = fingerprint_voice_journal(
        transcript="identical-content",
        audio_bytes=b"",
    )

    assert text.value != voice.value