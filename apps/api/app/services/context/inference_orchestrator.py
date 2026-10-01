import numpy as np
import torch
from sqlalchemy.orm import Session

from app.models.context_inference import ContextInference
from app.models.journal import JournalEntry
from app.models.journal_feature_set import JournalFeatureSet
from app.services.context.inference_factory import (
    build_pending_context_inference,
)
from app.services.context.inference_lifecycle import (
    complete_context_inference,
    fail_context_inference,
    mark_context_inference_processing,
)
from app.services.context.input_adapter import (
    build_context_model_input,
)
from app.services.context.model.context_model import (
    ContextModel,
)
from app.services.context.model_release import (
    ContextModelRelease,
)


_CONTEXT_INFERENCE_FAILURE_MESSAGE = (
    "Context inference execution failed"
)


class ContextInferenceOrchestrationError(RuntimeError):
    """Raised when M4 orchestration invariants are violated."""


def run_text_context_inference(
    db: Session,
    *,
    journal: JournalEntry,
    release: ContextModelRelease,
    model: ContextModel,
) -> ContextInference:
    """
    Execute one TEXT M4 inference.

    Responsibilities:
        - consume the authoritative M3 input boundary
        - create the matching M4 inference generation
        - transition PENDING -> PROCESSING
        - execute deterministic TEXT inference
        - publish COMPLETED on success
        - publish FAILED on post-creation execution failure

    This function intentionally does not:
        - load model artifacts
        - verify artifact hashes
        - commit the transaction
        - resolve duplicate/concurrent generations
        - fall back to historical M3 generations

    Failures before ContextInference creation do not create
    an M4 generation.
    """

    if not isinstance(journal, JournalEntry):
        raise ContextInferenceOrchestrationError(
            "journal must be a JournalEntry"
        )

    if not isinstance(release, ContextModelRelease):
        raise ContextInferenceOrchestrationError(
            "release must be a ContextModelRelease"
        )

    if not isinstance(model, ContextModel):
        raise ContextInferenceOrchestrationError(
            "model must be a ContextModel"
        )

    if journal.entry_type != "TEXT":
        raise ContextInferenceOrchestrationError(
            "TEXT orchestration requires a TEXT journal entry"
        )

    _validate_model_matches_release(
        model=model,
        release=release,
    )

    model_input = build_context_model_input(
        db,
        journal=journal,
    )

    if model_input.entry_type != "TEXT":
        raise ContextInferenceOrchestrationError(
            "Authoritative M4 input is not TEXT"
        )

    if model_input.audio_embedding is not None:
        raise ContextInferenceOrchestrationError(
            "TEXT M4 input must not contain an audio embedding"
        )

    feature_set = db.get(
        JournalFeatureSet,
        model_input.feature_set_id,
    )

    if feature_set is None:
        raise ContextInferenceOrchestrationError(
            "Authoritative feature generation is unavailable"
        )

    inference = build_pending_context_inference(
        feature_set=feature_set,
        release=release,
    )

    db.add(inference)
    db.flush()

    text_tensor = _embedding_to_tensor(
        model_input.text_embedding
    )

    model.eval()

    def execute_model():
        with torch.inference_mode():
            return model.forward_text(
                text_tensor,
                text_quality=model_input.text_quality,
            )

    return _execute_context_inference(
        db,
        inference=inference,
        execute_model=execute_model,
    )


def run_voice_context_inference(
    db: Session,
    *,
    journal: JournalEntry,
    release: ContextModelRelease,
    model: ContextModel,
) -> ContextInference:
    """
    Execute one VOICE M4 inference.

    VOICE requires both authoritative text and audio M3 evidence.
    Missing audio must never silently degrade into TEXT inference.

    Responsibilities:
        - consume the authoritative M3 input boundary
        - enforce the VOICE multimodal contract
        - create the matching M4 inference generation
        - transition PENDING -> PROCESSING
        - execute deterministic VOICE inference
        - publish COMPLETED on success
        - publish FAILED on post-creation execution failure

    This function intentionally does not:
        - load model artifacts
        - verify artifact hashes
        - commit the transaction
        - resolve duplicate/concurrent generations
        - fall back to TEXT inference
        - fall back to historical M3 generations

    Failures before ContextInference creation do not create
    an M4 generation.
    """

    if not isinstance(journal, JournalEntry):
        raise ContextInferenceOrchestrationError(
            "journal must be a JournalEntry"
        )

    if not isinstance(release, ContextModelRelease):
        raise ContextInferenceOrchestrationError(
            "release must be a ContextModelRelease"
        )

    if not isinstance(model, ContextModel):
        raise ContextInferenceOrchestrationError(
            "model must be a ContextModel"
        )

    if journal.entry_type != "VOICE":
        raise ContextInferenceOrchestrationError(
            "VOICE orchestration requires a VOICE journal entry"
        )

    _validate_model_matches_release(
        model=model,
        release=release,
    )

    model_input = build_context_model_input(
        db,
        journal=journal,
    )

    if model_input.entry_type != "VOICE":
        raise ContextInferenceOrchestrationError(
            "Authoritative M4 input is not VOICE"
        )

    if model_input.audio_embedding is None:
        raise ContextInferenceOrchestrationError(
            "VOICE M4 input requires an audio embedding"
        )

    if model_input.audio_quality is None:
        raise ContextInferenceOrchestrationError(
            "VOICE M4 input requires audio quality evidence"
        )

    if model_input.audio_provenance is None:
        raise ContextInferenceOrchestrationError(
            "VOICE M4 input requires audio provenance"
        )

    feature_set = db.get(
        JournalFeatureSet,
        model_input.feature_set_id,
    )

    if feature_set is None:
        raise ContextInferenceOrchestrationError(
            "Authoritative feature generation is unavailable"
        )

    inference = build_pending_context_inference(
        feature_set=feature_set,
        release=release,
    )

    db.add(inference)
    db.flush()

    text_tensor = _embedding_to_tensor(
        model_input.text_embedding
    )

    audio_tensor = _embedding_to_tensor(
        model_input.audio_embedding
    )

    model.eval()

    def execute_model():
        with torch.inference_mode():
            return model.forward_voice(
                text_tensor,
                audio_tensor,
                text_quality=model_input.text_quality,
                audio_quality=model_input.audio_quality,
            )

    return _execute_context_inference(
        db,
        inference=inference,
        execute_model=execute_model,
    )


def _execute_context_inference(
    db: Session,
    *,
    inference: ContextInference,
    execute_model,
) -> ContextInference:
    """
    Execute the model after an M4 inference generation exists.

    Successful execution:
        PROCESSING -> COMPLETED

    Failed execution:
        PROCESSING -> FAILED

    The original exception is re-raised to the caller while only a
    generic safe error message is persisted.

    Transaction commit remains the responsibility of the caller.
    """

    try:
        mark_context_inference_processing(
            inference
        )

        output = execute_model()

        complete_context_inference(
            inference,
            output,
        )

        db.flush()

        return inference

    except Exception:
        fail_context_inference(
            inference,
            error_message=(
                _CONTEXT_INFERENCE_FAILURE_MESSAGE
            ),
        )

        db.flush()

        raise


def _validate_model_matches_release(
    *,
    model: ContextModel,
    release: ContextModelRelease,
) -> None:
    """
    Ensure the supplied runtime model matches the configured release
    semantics before an inference generation is created.
    """

    if model.state_capability is not release.state_capability:
        raise ContextInferenceOrchestrationError(
            "Model state capability does not match release"
        )

    if (
        model.confidence_calibrated
        != release.confidence_calibrated
    ):
        raise ContextInferenceOrchestrationError(
            "Model confidence calibration does not match release"
        )


def _embedding_to_tensor(
    embedding: np.ndarray,
) -> torch.Tensor:
    """
    Convert one validated M3 learned embedding into the tensor shape
    required by the M4 model boundary.

    This is a defensive orchestration check. M4.3 remains the
    authoritative validation boundary for M3 evidence.
    """

    array = np.asarray(
        embedding,
        dtype=np.float32,
    )

    if array.shape != (768,):
        raise ContextInferenceOrchestrationError(
            "M4 embedding must have shape (768,)"
        )

    if not np.isfinite(array).all():
        raise ContextInferenceOrchestrationError(
            "M4 embedding must contain only finite values"
        )

    return torch.from_numpy(
        array.copy()
    )