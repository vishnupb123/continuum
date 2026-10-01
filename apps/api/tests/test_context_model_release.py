import pytest

from app.models.context_constants import (
    CONTEXT_ARCHITECTURE_VERSION,
)
from app.services.context.model.state_contract import (
    StateCapability,
)
from app.services.context.model_release import (
    ContextModelRelease,
    ContextModelReleaseError,
    build_context_model,
    build_research_model_release,
)


def test_valid_research_release():
    release = build_research_model_release(
        model_revision="research-test-r1",
        model_artifact_hash="abc123",
    )

    assert (
        release.architecture_version
        == CONTEXT_ARCHITECTURE_VERSION
    )

    assert (
        release.model_revision
        == "research-test-r1"
    )

    assert (
        release.model_artifact_hash
        == "abc123"
    )

    assert (
        release.state_capability
        is StateCapability.RESEARCH
    )

    assert release.confidence_calibrated is False


def test_release_is_immutable():
    release = build_research_model_release(
        model_revision="research-test-r1",
        model_artifact_hash="abc123",
    )

    with pytest.raises(
        AttributeError,
    ):
        release.model_revision = "changed"


@pytest.mark.parametrize(
    "field_value",
    [
        "",
        "   ",
    ],
)
def test_blank_revision_is_rejected(
    field_value,
):
    with pytest.raises(
        ContextModelReleaseError,
        match="model_revision",
    ):
        build_research_model_release(
            model_revision=field_value,
            model_artifact_hash="abc123",
        )


@pytest.mark.parametrize(
    "field_value",
    [
        "",
        "   ",
    ],
)
def test_blank_artifact_hash_is_rejected(
    field_value,
):
    with pytest.raises(
        ContextModelReleaseError,
        match="model_artifact_hash",
    ):
        build_research_model_release(
            model_revision="research-test-r1",
            model_artifact_hash=field_value,
        )


def test_unsupported_architecture_is_rejected():
    with pytest.raises(
        ContextModelReleaseError,
        match="architecture version",
    ):
        ContextModelRelease(
            architecture_version="some-other-model",
            model_revision="r1",
            model_artifact_hash="abc123",
            state_capability=StateCapability.RESEARCH,
        )


def test_invalid_state_capability_is_rejected():
    with pytest.raises(
        ContextModelReleaseError,
        match="state_capability",
    ):
        ContextModelRelease(
            architecture_version=(
                CONTEXT_ARCHITECTURE_VERSION
            ),
            model_revision="r1",
            model_artifact_hash="abc123",
            state_capability="RESEARCH",
        )


def test_invalid_calibration_flag_is_rejected():
    with pytest.raises(
        ContextModelReleaseError,
        match="confidence_calibrated",
    ):
        ContextModelRelease(
            architecture_version=(
                CONTEXT_ARCHITECTURE_VERSION
            ),
            model_revision="r1",
            model_artifact_hash="abc123",
            state_capability=StateCapability.RESEARCH,
            confidence_calibrated="false",
        )


def test_model_factory_preserves_release_capability():
    release = ContextModelRelease(
        architecture_version=(
            CONTEXT_ARCHITECTURE_VERSION
        ),
        model_revision="validated-test-r1",
        model_artifact_hash="abc123",
        state_capability=StateCapability.VALIDATED,
        confidence_calibrated=True,
    )

    model = build_context_model(
        release
    )

    assert (
        model.state_capability
        is StateCapability.VALIDATED
    )

    assert model.confidence_calibrated is True


def test_model_factory_preserves_research_boundary():
    release = build_research_model_release(
        model_revision="research-test-r1",
        model_artifact_hash="abc123",
    )

    model = build_context_model(
        release
    )

    assert (
        model.state_capability
        is StateCapability.RESEARCH
    )

    assert model.confidence_calibrated is False
    