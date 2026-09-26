from dataclasses import dataclass
from typing import Sequence


TEXT_CHUNKING_VERSION = "text-chunking-v1"


@dataclass(frozen=True)
class TokenChunk:
    token_ids: tuple[int, ...]

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


def chunk_token_ids(
    token_ids: Sequence[int],
    *,
    max_content_tokens: int,
) -> tuple[TokenChunk, ...]:
    if max_content_tokens <= 0:
        raise ValueError(
            "max_content_tokens must be greater than zero"
        )

    if not token_ids:
        return tuple()

    chunks = []

    for start in range(
        0,
        len(token_ids),
        max_content_tokens,
    ):
        chunk = tuple(
            int(token_id)
            for token_id in token_ids[
                start : start + max_content_tokens
            ]
        )

        chunks.append(
            TokenChunk(token_ids=chunk)
        )

    return tuple(chunks)