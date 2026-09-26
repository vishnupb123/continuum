from typing import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from transformers.utils import logging as transformers_logging

from app.services.features.text_encoder.base import (
    TextEncoder,
    TextEncodingResult,
)
from app.services.features.text_encoder.chunking import (
    TEXT_CHUNKING_VERSION,
    TokenChunk,
    chunk_token_ids,
)


TEXT_ENCODER_VERSION = "text-encoder-v1"
EXPECTED_MPNet_DIMENSION = 768


class MPNetTextEncoder(TextEncoder):
    def __init__(
        self,
        *,
        model_name: str,
        revision: str | None,
        device: str,
        cache_folder: str,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device

        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device=device,
            cache_folder=cache_folder,
        )

        dimension = (
            self.model
            .get_sentence_embedding_dimension()
        )

        if dimension != EXPECTED_MPNet_DIMENSION:
            raise RuntimeError(
                "Unexpected MPNet embedding "
                f"dimension: {dimension}"
            )

        self.dimension = dimension

        self.tokenizer = self.model.tokenizer
        self.max_seq_length = (
            self.model.max_seq_length
        )

        self.special_token_count = (
            self.tokenizer
            .num_special_tokens_to_add(
                pair=False
            )
        )

        self.max_content_tokens = (
            self.max_seq_length
            - self.special_token_count
        )

        if self.max_content_tokens <= 0:
            raise RuntimeError(
                "Invalid MPNet content-token budget"
            )

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

        content_token_ids = (
            self._tokenize_without_special_tokens(
                text
            )
        )

        if not content_token_ids:
            raise ValueError(
                "text produced no content tokens"
            )

        chunks = chunk_token_ids(
            content_token_ids,
            max_content_tokens=(
                self.max_content_tokens
            ),
        )

        chunk_embeddings = (
            self._encode_chunks(chunks)
        )

        embedding = (
            self._aggregate_chunk_embeddings(
                chunk_embeddings,
                chunks,
            )
        )

        return TextEncodingResult(
            embedding=tuple(
                float(value)
                for value in embedding
            ),
            dimension=self.dimension,
            encoder_name=self.model_name,
            encoder_version=TEXT_ENCODER_VERSION,
            encoder_revision=(
                self.revision
                or "UNPINNED"
            ),
            normalized=True,
        )

    def _tokenize_without_special_tokens(
        self,
        text: str,
    ) -> tuple[int, ...]:
      encoded = self.tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
     )

      token_ids = encoded["input_ids"]

      return tuple(
        int(token_id)
        for token_id in token_ids
    )
      
      
    def _encode_chunks(
        self,
        chunks: Sequence[TokenChunk],
    ) -> np.ndarray:
        prepared_chunks = []

        for chunk in chunks:
            prepared = (
                self.tokenizer.prepare_for_model(
                    list(chunk.token_ids),
                    add_special_tokens=True,
                    truncation=False,
                    return_attention_mask=True,
                )
            )

            if (
                len(prepared["input_ids"])
                > self.max_seq_length
            ):
                raise RuntimeError(
                    "Prepared chunk exceeds "
                    "MPNet sequence limit"
                )

            prepared_chunks.append(
                prepared
            )

        batch = self.tokenizer.pad(
            prepared_chunks,
            padding=True,
            return_tensors="pt",
        )

        batch = {
            key: value.to(self.device)
            for key, value in batch.items()
        }

        self.model.eval()

        with torch.no_grad():
            output = self.model(batch)

        if "sentence_embedding" not in output:
            raise RuntimeError(
                "SentenceTransformer output "
                "did not contain sentence_embedding"
            )

        embeddings = (
            output["sentence_embedding"]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        if embeddings.ndim != 2:
            raise RuntimeError(
                "Expected two-dimensional "
                "chunk embeddings"
            )

        if embeddings.shape[0] != len(chunks):
            raise RuntimeError(
                "Chunk embedding count mismatch"
            )

        if embeddings.shape[1] != self.dimension:
            raise RuntimeError(
                "Unexpected embedding dimension"
            )

        return embeddings

    def _aggregate_chunk_embeddings(
        self,
        embeddings: np.ndarray,
        chunks: Sequence[TokenChunk],
    ) -> np.ndarray:
        if len(chunks) == 0:
            raise ValueError(
                "At least one chunk is required"
            )

        if embeddings.shape != (
            len(chunks),
            self.dimension,
        ):
            raise RuntimeError(
                "Invalid chunk embedding shape"
            )

        weights = np.asarray(
            [
                chunk.token_count
                for chunk in chunks
            ],
            dtype=np.float32,
        )

        if np.any(weights <= 0):
            raise RuntimeError(
                "Chunk weights must be positive"
            )

        weighted_embedding = np.average(
            embeddings,
            axis=0,
            weights=weights,
        ).astype(
            np.float32,
            copy=False,
        )

        norm = float(
            np.linalg.norm(
                weighted_embedding
            )
        )

        if not np.isfinite(norm) or norm <= 0.0:
            raise RuntimeError(
                "Cannot normalize aggregated "
                "text embedding"
            )

        normalized = (
            weighted_embedding / norm
        ).astype(
            np.float32,
            copy=False,
        )

        if normalized.shape != (
            self.dimension,
        ):
            raise RuntimeError(
                "Unexpected aggregated "
                "embedding dimension"
            )

        return normalized