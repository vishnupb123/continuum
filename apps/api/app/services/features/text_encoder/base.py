from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class TextEncodingResult:
    embedding: tuple[float, ...]
    dimension: int

    encoder_name: str
    encoder_version: str
    encoder_revision: str

    normalized: bool

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError(
                "dimension must be greater than zero"
            )

        if len(self.embedding) != self.dimension:
            raise ValueError(
                "embedding length must match dimension"
            )


class TextEncoder(ABC):
    @abstractmethod
    def encode(
        self,
        text: str,
    ) -> TextEncodingResult:
        raise NotImplementedError