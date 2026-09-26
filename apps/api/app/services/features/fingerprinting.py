import hashlib
import json
from dataclasses import dataclass
from app.models.journal import JournalEntry
from app.services.storage.factory import get_audio_storage


FINGERPRINT_VERSION = "source-fingerprint-v1"


@dataclass(frozen=True)
class SourceFingerprint:
    value: str
    version: str = FINGERPRINT_VERSION


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_json(value: dict) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def fingerprint_text_journal(
    text: str,
) -> SourceFingerprint:
    """
    Fingerprint the source content of a TEXT journal.

    This intentionally fingerprints the exact persisted source text rather
    than a normalized/preprocessed representation. Preprocessing has its own
    independent version.
    """
    payload = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "entry_type": "TEXT",
        "text_sha256": _sha256_text(text),
    }

    return SourceFingerprint(
        value=_sha256_bytes(
            _canonical_json(payload)
        )
    )


def fingerprint_voice_journal(
    transcript: str,
    audio_bytes: bytes,
) -> SourceFingerprint:
    """
    Fingerprint both source modalities of a VOICE journal.

    A change to either the transcript or the stored audio must produce a
    different feature generation.
    """
    payload = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "entry_type": "VOICE",
        "transcript_sha256": _sha256_text(
            transcript
        ),
        "audio_sha256": _sha256_bytes(
            audio_bytes
        ),
    }

    return SourceFingerprint(
        value=_sha256_bytes(
            _canonical_json(payload)
        )
    )
    
def calculate_journal_source_hash(
    journal: JournalEntry,
) -> str:
    """
    Calculate the canonical feature-generation source hash
    from the exact persisted journal source.

    TEXT:
        exact persisted raw text

    VOICE:
        exact persisted transcript + stored audio bytes

    This function is the single source of truth for journal
    feature-generation identity.
    """

    if journal.entry_type == "TEXT":
        if journal.raw_text is None:
            raise ValueError(
                "TEXT journal has no source text"
            )

        return fingerprint_text_journal(
            journal.raw_text
        ).value

    if journal.entry_type == "VOICE":
        if (
            journal.raw_text is None
            or not journal.raw_text.strip()
        ):
            raise ValueError(
                "VOICE journal has no transcript"
            )

        if journal.audio is None:
            raise ValueError(
                "VOICE journal has no audio"
            )

        storage = get_audio_storage()

        audio_bytes = storage.get(
            journal.audio.storage_key
        )

        return fingerprint_voice_journal(
            journal.raw_text,
            audio_bytes,
        ).value

    raise ValueError(
        "Unsupported journal entry type: "
        f"{journal.entry_type}"
    )