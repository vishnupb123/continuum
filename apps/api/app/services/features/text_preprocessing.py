import re
import unicodedata
from dataclasses import dataclass

from app.models.feature_constants import (
    FEATURE_QUALITY_DEGRADED,
    FEATURE_QUALITY_GOOD,
    FEATURE_QUALITY_UNUSABLE,
)


TEXT_PREPROCESSING_VERSION = "text-preprocess-v1"

_HORIZONTAL_WHITESPACE_RE = re.compile(r"[^\S\n]+")
_EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class TextPreprocessingResult:
    text: str
    word_count: int
    character_count: int
    quality_status: str
    quality_reasons: tuple[str, ...]
    preprocessing_version: str = TEXT_PREPROCESSING_VERSION

    @property
    def is_usable(self) -> bool:
        return self.quality_status != FEATURE_QUALITY_UNUSABLE


def _remove_control_characters(text: str) -> str:
    """
    Remove Unicode control characters while preserving semantic
    whitespace boundaries.

    Newlines are preserved as line boundaries.
    Tabs are converted to spaces so adjacent words cannot be merged.
    Other control characters are removed.
    """
    characters: list[str] = []

    for character in text:
        if character == "\n":
            characters.append("\n")
            continue

        if character == "\t":
            characters.append(" ")
            continue

        if unicodedata.category(character) == "Cc":
            continue

        characters.append(character)

    return "".join(characters)


def _normalize_line_whitespace(text: str) -> str:
    lines = text.split("\n")

    normalized_lines = [
        _HORIZONTAL_WHITESPACE_RE.sub(
            " ",
            line,
        ).strip()
        for line in lines
    ]

    return "\n".join(normalized_lines)


def _count_words(text: str) -> int:
    if not text:
        return 0

    return len(text.split())


def _assess_quality(
    *,
    text: str,
    word_count: int,
) -> tuple[str, tuple[str, ...]]:
    if not text:
        return (
            FEATURE_QUALITY_UNUSABLE,
            ("EMPTY_AFTER_PREPROCESSING",),
        )

    if word_count <= 2:
        return (
            FEATURE_QUALITY_DEGRADED,
            ("VERY_SHORT_TEXT",),
        )

    return (
        FEATURE_QUALITY_GOOD,
        (),
    )


def preprocess_text(
    text: str,
) -> TextPreprocessingResult:
    if not isinstance(text, str):
        raise TypeError(
            "text must be a string"
        )

    normalized = unicodedata.normalize(
        "NFKC",
        text,
    )

    normalized = normalized.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    normalized = _remove_control_characters(
        normalized
    )

    normalized = _normalize_line_whitespace(
        normalized
    )

    normalized = _EXCESSIVE_NEWLINES_RE.sub(
        "\n\n",
        normalized,
    )

    normalized = normalized.strip()

    word_count = _count_words(
        normalized
    )

    character_count = len(
        normalized
    )

    quality_status, quality_reasons = (
        _assess_quality(
            text=normalized,
            word_count=word_count,
        )
    )

    return TextPreprocessingResult(
        text=normalized,
        word_count=word_count,
        character_count=character_count,
        quality_status=quality_status,
        quality_reasons=quality_reasons,
    )