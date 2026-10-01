import uuid
from unittest.mock import patch

import numpy as np
import pytest
import torch

from app.models.context_constants import (
    CONTEXT_ARCHITECTURE_VERSION,
    CONTEXT_STATUS_COMPLETED,
)
from app.models.journal import JournalEntry
from app.models.journal_feature_set import JournalFeatureSet
from app.services.context.input_contract import (
    ContextModelInput,
    FeatureProvenance,
)
from app.services.context.inference_orchestrator import (
    ContextInferenceOrchestrationError,
    run_text_context_inference,
)
from app.services.context.model.context_model import (
    ContextModel,
)
from app.services.context.model.state_contract import (
    StateCapability,
)
from app.services.context.model_release import (
    ContextModelRelease,
)


def make_release(
    *,
    capability=StateCapability.RESEARCH,
    calibrated=False,
):
    return ContextModelRelease(
        architecture_version=CONTEXT_ARCHITECTURE_VERSION,
        model_revision="test-r1",
        model_artifact_hash="abc123",
        state_capability=capability,
        confidence_calibrated=calibrated,
    )


def make_model(
    *,
    capability=StateCapability.RESEARCH,
    calibrated=False,
):
    model = ContextModel(
        state_capability=capability,
        confidence_calibrated=calibrated,
    )
    model.eval()
    return model


def make_text_input(
    *,
    journal_id,
    feature_set_id,
):
    return ContextModelInput(
        journal_id=journal_id,
        feature_set_id=feature_set_id,
        entry_type="TEXT",
        feature_pipeline_version="test-pipeline-v1",
        source_hash="source-hash",
        text_embedding=np.linspace(
            -1.0,
            1.0,
            768,
            dtype=np.float32,
        ),
        text_quality="GOOD",
        text_provenance=FeatureProvenance(
            encoder_name="test-text-encoder",
            encoder_version="test-v1",
            encoder_revision="test-revision",
        ),
        audio_embedding=None,
        audio_quality=None,
        audio_provenance=None,
    )


def make_journal():
    journal = JournalEntry()
    journal.id = uuid.uuid4()
    journal.entry_type = "TEXT"
    return journal


def make_feature_set():
    feature_set = JournalFeatureSet()
    feature_set.id = uuid.uuid4()
    return feature_set


class FakeSession:
    def __init__(self, feature_set):
        self.feature_set = feature_set
        self.added = []
        self.flush_count = 0

    def get(self, model_type, object_id):
        if (
            model_type is JournalFeatureSet
            and object_id == self.feature_set.id
        ):
            return self.feature_set

        return None

    def add(self, value):
        self.added.append(value)

    def flush(self):
        self.flush_count += 1


def test_text_orchestration_completes_research_inference():
    journal = make_journal()
    feature_set = make_feature_set()
    db = FakeSession(feature_set)

    model_input = make_text_input(
        journal_id=journal.id,
        feature_set_id=feature_set.id,
    )

    with patch(
        "app.services.context.inference_orchestrator."
        "build_context_model_input",
        return_value=model_input,
    ):
        inference = run_text_context_inference(
            db,
            journal=journal,
            release=make_release(),
            model=make_model(),
        )

    assert inference.status == CONTEXT_STATUS_COMPLETED
    assert inference.feature_set_id == feature_set.id

    assert inference.state_capability == "RESEARCH"

    assert inference.representation_dimension == 256
    assert len(inference.representation) == 256

    assert inference.energy is None
    assert inference.stress is None
    assert inference.positive_mood is None
    assert inference.social_connection is None

    assert inference.confidence_score is None
    assert inference.confidence_calibrated is False

    assert db.added == [inference]
    assert db.flush_count == 2


def test_text_orchestration_can_publish_validated_output():
    journal = make_journal()
    feature_set = make_feature_set()
    db = FakeSession(feature_set)

    model_input = make_text_input(
        journal_id=journal.id,
        feature_set_id=feature_set.id,
    )

    release = make_release(
        capability=StateCapability.VALIDATED,
        calibrated=True,
    )

    model = make_model(
        capability=StateCapability.VALIDATED,
        calibrated=True,
    )

    with patch(
        "app.services.context.inference_orchestrator."
        "build_context_model_input",
        return_value=model_input,
    ):
        inference = run_text_context_inference(
            db,
            journal=journal,
            release=release,
            model=model,
        )

    assert inference.status == CONTEXT_STATUS_COMPLETED
    assert inference.state_capability == "VALIDATED"

    assert inference.energy is not None
    assert inference.stress is not None
    assert inference.positive_mood is not None
    assert inference.social_connection is not None

    assert 0.0 <= inference.energy <= 1.0
    assert 0.0 <= inference.stress <= 1.0
    assert 0.0 <= inference.positive_mood <= 1.0
    assert 0.0 <= inference.social_connection <= 1.0

    assert inference.confidence_score is not None
    assert 0.0 <= inference.confidence_score <= 1.0

    assert inference.confidence_calibrated is True


def test_text_orchestration_rejects_voice_journal():
    journal = make_journal()
    journal.entry_type = "VOICE"

    with pytest.raises(
        ContextInferenceOrchestrationError,
        match="requires a TEXT",
    ):
        run_text_context_inference(
            FakeSession(make_feature_set()),
            journal=journal,
            release=make_release(),
            model=make_model(),
        )


def test_text_orchestration_rejects_model_capability_mismatch():
    journal = make_journal()

    with pytest.raises(
        ContextInferenceOrchestrationError,
        match="state capability does not match",
    ):
        run_text_context_inference(
            FakeSession(make_feature_set()),
            journal=journal,
            release=make_release(
                capability=StateCapability.RESEARCH,
            ),
            model=make_model(
                capability=StateCapability.VALIDATED,
            ),
        )


def test_text_orchestration_rejects_calibration_mismatch():
    journal = make_journal()

    with pytest.raises(
        ContextInferenceOrchestrationError,
        match="confidence calibration does not match",
    ):
        run_text_context_inference(
            FakeSession(make_feature_set()),
            journal=journal,
            release=make_release(
                calibrated=False,
            ),
            model=make_model(
                calibrated=True,
            ),
        )


def test_text_orchestration_rejects_audio_in_text_input():
    journal = make_journal()
    feature_set = make_feature_set()
    db = FakeSession(feature_set)

    model_input = make_text_input(
        journal_id=journal.id,
        feature_set_id=feature_set.id,
    )

    object.__setattr__(
        model_input,
        "audio_embedding",
        np.zeros(768, dtype=np.float32),
    )

    with patch(
        "app.services.context.inference_orchestrator."
        "build_context_model_input",
        return_value=model_input,
    ):
        with pytest.raises(
            ContextInferenceOrchestrationError,
            match="must not contain an audio embedding",
        ):
            run_text_context_inference(
                db,
                journal=journal,
                release=make_release(),
                model=make_model(),
            )


def test_text_orchestration_does_not_commit_transaction():
    journal = make_journal()
    feature_set = make_feature_set()
    db = FakeSession(feature_set)

    model_input = make_text_input(
        journal_id=journal.id,
        feature_set_id=feature_set.id,
    )

    assert not hasattr(db, "commit")

    with patch(
        "app.services.context.inference_orchestrator."
        "build_context_model_input",
        return_value=model_input,
    ):
        inference = run_text_context_inference(
            db,
            journal=journal,
            release=make_release(),
            model=make_model(),
        )

    assert inference.status == CONTEXT_STATUS_COMPLETED
    
def test_text_model_failure_marks_inference_failed():
    journal = make_journal()
    feature_set = make_feature_set()
    db = FakeSession(feature_set)

    model_input = make_text_input(
        journal_id=journal.id,
        feature_set_id=feature_set.id,
    )

    model = make_model()

    with patch(
        "app.services.context.inference_orchestrator."
        "build_context_model_input",
        return_value=model_input,
    ), patch.object(
        model,
        "forward_text",
        side_effect=RuntimeError(
            "sensitive runtime detail"
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="sensitive runtime detail",
        ):
            run_text_context_inference(
                db,
                journal=journal,
                release=make_release(),
                model=model,
            )

    inference = db.added[0]

    assert inference.status == "FAILED"
    assert (
        inference.error_message
        == "Context inference execution failed"
    )

    assert "sensitive" not in inference.error_message

    assert inference.representation is None
    assert inference.energy is None
    assert inference.stress is None
    assert inference.positive_mood is None
    assert inference.social_connection is None
    assert inference.confidence_score is None

    assert db.flush_count == 2


def test_input_failure_creates_no_context_inference():
    journal = make_journal()
    feature_set = make_feature_set()
    db = FakeSession(feature_set)

    with patch(
        "app.services.context.inference_orchestrator."
        "build_context_model_input",
        side_effect=RuntimeError(
            "input unavailable"
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="input unavailable",
        ):
            run_text_context_inference(
                db,
                journal=journal,
                release=make_release(),
                model=make_model(),
            )

    assert db.added == []
    assert db.flush_count == 0