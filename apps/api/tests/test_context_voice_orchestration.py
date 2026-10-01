import uuid
from unittest.mock import patch

import numpy as np
import pytest

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
    run_voice_context_inference,
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


def make_voice_input(
    *,
    journal_id,
    feature_set_id,
):
    return ContextModelInput(
        journal_id=journal_id,
        feature_set_id=feature_set_id,
        entry_type="VOICE",
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
            encoder_revision="text-revision",
        ),
        audio_embedding=np.linspace(
            1.0,
            -1.0,
            768,
            dtype=np.float32,
        ),
        audio_quality="GOOD",
        audio_provenance=FeatureProvenance(
            encoder_name="test-audio-encoder",
            encoder_version="test-v1",
            encoder_revision="audio-revision",
        ),
    )


def make_journal():
    journal = JournalEntry()
    journal.id = uuid.uuid4()
    journal.entry_type = "VOICE"
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


def test_voice_orchestration_completes_research_inference():
    journal = make_journal()
    feature_set = make_feature_set()
    db = FakeSession(feature_set)

    model_input = make_voice_input(
        journal_id=journal.id,
        feature_set_id=feature_set.id,
    )

    with patch(
        "app.services.context.inference_orchestrator."
        "build_context_model_input",
        return_value=model_input,
    ):
        inference = run_voice_context_inference(
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


def test_voice_orchestration_can_publish_validated_output():
    journal = make_journal()
    feature_set = make_feature_set()
    db = FakeSession(feature_set)

    model_input = make_voice_input(
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
        inference = run_voice_context_inference(
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


def test_voice_orchestration_rejects_text_journal():
    journal = make_journal()
    journal.entry_type = "TEXT"

    with pytest.raises(
        ContextInferenceOrchestrationError,
        match="requires a VOICE",
    ):
        run_voice_context_inference(
            FakeSession(make_feature_set()),
            journal=journal,
            release=make_release(),
            model=make_model(),
        )


def test_voice_orchestration_rejects_missing_audio_embedding():
    journal = make_journal()
    feature_set = make_feature_set()
    db = FakeSession(feature_set)

    model_input = make_voice_input(
        journal_id=journal.id,
        feature_set_id=feature_set.id,
    )

    object.__setattr__(
        model_input,
        "audio_embedding",
        None,
    )

    with patch(
        "app.services.context.inference_orchestrator."
        "build_context_model_input",
        return_value=model_input,
    ):
        with pytest.raises(
            ContextInferenceOrchestrationError,
            match="requires an audio embedding",
        ):
            run_voice_context_inference(
                db,
                journal=journal,
                release=make_release(),
                model=make_model(),
            )


def test_voice_orchestration_rejects_missing_audio_quality():
    journal = make_journal()
    feature_set = make_feature_set()
    db = FakeSession(feature_set)

    model_input = make_voice_input(
        journal_id=journal.id,
        feature_set_id=feature_set.id,
    )

    object.__setattr__(
        model_input,
        "audio_quality",
        None,
    )

    with patch(
        "app.services.context.inference_orchestrator."
        "build_context_model_input",
        return_value=model_input,
    ):
        with pytest.raises(
            ContextInferenceOrchestrationError,
            match="requires audio quality",
        ):
            run_voice_context_inference(
                db,
                journal=journal,
                release=make_release(),
                model=make_model(),
            )


def test_voice_orchestration_rejects_missing_audio_provenance():
    journal = make_journal()
    feature_set = make_feature_set()
    db = FakeSession(feature_set)

    model_input = make_voice_input(
        journal_id=journal.id,
        feature_set_id=feature_set.id,
    )

    object.__setattr__(
        model_input,
        "audio_provenance",
        None,
    )

    with patch(
        "app.services.context.inference_orchestrator."
        "build_context_model_input",
        return_value=model_input,
    ):
        with pytest.raises(
            ContextInferenceOrchestrationError,
            match="requires audio provenance",
        ):
            run_voice_context_inference(
                db,
                journal=journal,
                release=make_release(),
                model=make_model(),
            )


def test_voice_orchestration_rejects_model_capability_mismatch():
    journal = make_journal()

    with pytest.raises(
        ContextInferenceOrchestrationError,
        match="state capability does not match",
    ):
        run_voice_context_inference(
            FakeSession(make_feature_set()),
            journal=journal,
            release=make_release(
                capability=StateCapability.RESEARCH,
            ),
            model=make_model(
                capability=StateCapability.VALIDATED,
            ),
        )


def test_voice_orchestration_rejects_calibration_mismatch():
    journal = make_journal()

    with pytest.raises(
        ContextInferenceOrchestrationError,
        match="confidence calibration does not match",
    ):
        run_voice_context_inference(
            FakeSession(make_feature_set()),
            journal=journal,
            release=make_release(
                calibrated=False,
            ),
            model=make_model(
                calibrated=True,
            ),
        )


def test_voice_orchestration_does_not_commit_transaction():
    journal = make_journal()
    feature_set = make_feature_set()
    db = FakeSession(feature_set)

    model_input = make_voice_input(
        journal_id=journal.id,
        feature_set_id=feature_set.id,
    )

    assert not hasattr(db, "commit")

    with patch(
        "app.services.context.inference_orchestrator."
        "build_context_model_input",
        return_value=model_input,
    ):
        inference = run_voice_context_inference(
            db,
            journal=journal,
            release=make_release(),
            model=make_model(),
        )

    assert inference.status == CONTEXT_STATUS_COMPLETED

def test_voice_model_failure_marks_inference_failed():
    journal = make_journal()
    feature_set = make_feature_set()
    db = FakeSession(feature_set)

    model_input = make_voice_input(
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
        "forward_voice",
        side_effect=RuntimeError(
            "private audio runtime detail"
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="private audio runtime detail",
        ):
            run_voice_context_inference(
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

    assert "private audio" not in inference.error_message

    assert inference.representation is None
    assert inference.energy is None
    assert inference.stress is None
    assert inference.positive_mood is None
    assert inference.social_connection is None
    assert inference.confidence_score is None

    assert db.flush_count == 2