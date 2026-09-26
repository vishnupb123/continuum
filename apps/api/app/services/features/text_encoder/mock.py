import hashlib
import math

from app.services.features.text_encoder.base import (
    TextEncoder,
    TextEncodingResult,
)


MOCK_TEXT_ENCODER_NAME = "mock-text-encoder"
MOCK_TEXT_ENCODER_VERSION = "mock-v1"
MOCK_TEXT_ENCODER_REVISION = "deterministic-sha256-v1"
MOCK_TEXT_ENCODER_DIMENSION = 8


class MockTextEncoder(TextEncoder):
    def encode(
        self,
        text: str,
    ) -> TextEncodingResult:
        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if not text:
            raise ValueError(
                "text must not be empty"
            )

        digest = hashlib.sha256(
            text.encode("utf-8")
        ).digest()

        values = []

        for index in range(
            MOCK_TEXT_ENCODER_DIMENSION
        ):
            start = index * 4
            chunk = digest[start:start + 4]

            integer = int.from_bytes(
                chunk,
                byteorder="big",
                signed=False,
            )

            value = (
                integer / 0xFFFFFFFF
            ) * 2.0 - 1.0

            values.append(value)

        magnitude = math.sqrt(
            sum(
                value * value
                for value in values
            )
        )

        if magnitude == 0:
            raise RuntimeError(
                "cannot normalize zero embedding"
            )

        normalized = tuple(
            value / magnitude
            for value in values
        )

        return TextEncodingResult(
            embedding=normalized,
            dimension=MOCK_TEXT_ENCODER_DIMENSION,
            encoder_name=MOCK_TEXT_ENCODER_NAME,
            encoder_version=MOCK_TEXT_ENCODER_VERSION,
            encoder_revision=MOCK_TEXT_ENCODER_REVISION,
            normalized=True,
        )