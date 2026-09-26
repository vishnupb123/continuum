import pytest

from app.models.feature_constants import (
    FEATURE_QUALITY_DEGRADED,
    FEATURE_QUALITY_GOOD,
    FEATURE_QUALITY_UNUSABLE,
)
from app.services.features.text_preprocessing import (
    TEXT_PREPROCESSING_VERSION,
    preprocess_text,
)


def test_preprocessing_is_deterministic():
    text = "Today   was\tproductive."

    first = preprocess_text(text)
    second = preprocess_text(text)

    assert first == second


def test_normalizes_horizontal_whitespace():
    result = preprocess_text(
        "Today    was\t\tvery   productive."
    )

    assert (
        result.text
        == "Today was very productive."
    )


def test_normalizes_line_endings():
    result = preprocess_text(
        "First line\r\nSecond line\rThird line"
    )

    assert result.text == (
        "First line\n"
        "Second line\n"
        "Third line"
    )


def test_limits_excessive_blank_lines():
    result = preprocess_text(
        "First paragraph.\n\n\n\n"
        "Second paragraph."
    )

    assert result.text == (
        "First paragraph.\n\n"
        "Second paragraph."
    )


def test_removes_control_characters():
    result = preprocess_text(
        "Today\x00 was\x07 productive."
    )

    assert (
        result.text
        == "Today was productive."
    )


def test_preserves_semantic_information():
    original = (
        "I am NOT feeling great 😔, "
        "but I can't explain why!"
    )

    result = preprocess_text(original)

    assert result.text == original

    assert "NOT" in result.text
    assert "😔" in result.text
    assert "can't" in result.text
    assert "!" in result.text


def test_applies_nfkc_unicode_normalization():
    result = preprocess_text(
        "Ｔｏｄａｙ was productive."
    )

    assert (
        result.text
        == "Today was productive."
    )


def test_good_quality_text():
    result = preprocess_text(
        "Today was productive."
    )

    assert (
        result.quality_status
        == FEATURE_QUALITY_GOOD
    )

    assert result.quality_reasons == ()
    assert result.is_usable is True


def test_one_word_is_degraded_but_usable():
    result = preprocess_text("fine")

    assert (
        result.quality_status
        == FEATURE_QUALITY_DEGRADED
    )

    assert result.quality_reasons == (
        "VERY_SHORT_TEXT",
    )

    assert result.is_usable is True


def test_two_words_are_degraded_but_usable():
    result = preprocess_text(
        "pretty tired"
    )

    assert (
        result.quality_status
        == FEATURE_QUALITY_DEGRADED
    )

    assert result.is_usable is True


def test_empty_text_is_unusable():
    result = preprocess_text(
        "   \t\n\n   "
    )

    assert result.text == ""
    assert result.word_count == 0
    assert result.character_count == 0

    assert (
        result.quality_status
        == FEATURE_QUALITY_UNUSABLE
    )

    assert result.quality_reasons == (
        "EMPTY_AFTER_PREPROCESSING",
    )

    assert result.is_usable is False


def test_statistics_match_processed_text():
    result = preprocess_text(
        "Today   was productive."
    )

    assert result.text == (
        "Today was productive."
    )

    assert result.word_count == 3

    assert (
        result.character_count
        == len(result.text)
    )


def test_does_not_silently_truncate_long_text():
    original = "word " * 5000

    result = preprocess_text(
        original
    )

    assert result.word_count == 5000

    assert result.text.endswith(
        "word"
    )


def test_non_string_input_is_rejected():
    with pytest.raises(
        TypeError,
        match="text must be a string",
    ):
        preprocess_text(None)
        
def test_tabs_preserve_word_boundaries():
    result = preprocess_text(
        "one\ttwo\tthree"
    )

    assert result.text == (
        "one two three"
    )

    assert result.word_count == 3