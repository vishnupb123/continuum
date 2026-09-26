import hashlib
import json
from dataclasses import dataclass


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