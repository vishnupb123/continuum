from dataclasses import dataclass

from app.models.context_constants import (
    CONTEXT_ARCHITECTURE_VERSION,
)
from app.services.context.model.state_contract import (
    StateCapability,
)

from app.services.context.model.context_model import (
    ContextModel,
)


class ContextModelReleaseError(ValueError):
    """Raised when an M4 model release configuration is invalid."""


@dataclass(frozen=True)
class ContextModelRelease:
    """
    Immutable identity and capability metadata for one
    configured M4 model release.

    This identifies the model release used for persisted
    inference. It does not contain model weights.
    """

    architecture_version: str
    model_revision: str
    model_artifact_hash: str
    state_capability: StateCapability
    confidence_calibrated: bool = False

    def __post_init__(self) -> None:
        _require_nonblank(
            self.architecture_version,
            field_name="architecture_version",
        )

        _require_nonblank(
            self.model_revision,
            field_name="model_revision",
        )

        _require_nonblank(
            self.model_artifact_hash,
            field_name="model_artifact_hash",
        )

        if not isinstance(
            self.state_capability,
            StateCapability,
        ):
            raise ContextModelReleaseError(
                "state_capability must be a StateCapability"
            )

        if not isinstance(
            self.confidence_calibrated,
            bool,
        ):
            raise ContextModelReleaseError(
                "confidence_calibrated must be boolean"
            )

        if (
            self.architecture_version
            != CONTEXT_ARCHITECTURE_VERSION
        ):
            raise ContextModelReleaseError(
                "Configured architecture version does not "
                "match the supported M4 architecture"
            )


def build_research_model_release(
    *,
    model_revision: str,
    model_artifact_hash: str,
) -> ContextModelRelease:
    """
    Explicit helper for research/test releases.

    Research capability does not permit publication of
    named state estimates.
    """

    return ContextModelRelease(
        architecture_version=(
            CONTEXT_ARCHITECTURE_VERSION
        ),
        model_revision=model_revision,
        model_artifact_hash=model_artifact_hash,
        state_capability=StateCapability.RESEARCH,
        confidence_calibrated=False,
    )


def _require_nonblank(
    value: str,
    *,
    field_name: str,
) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ContextModelReleaseError(
            f"{field_name} must be a non-blank string"
        )
        
def build_context_model(
    release: ContextModelRelease,
) -> ContextModel:
    """
    Construct a ContextModel whose runtime capability metadata
    exactly matches the configured release.
    """

    if not isinstance(
        release,
        ContextModelRelease,
    ):
        raise ContextModelReleaseError(
            "release must be a ContextModelRelease"
        )

    return ContextModel(
        state_capability=release.state_capability,
        confidence_calibrated=(
            release.confidence_calibrated
        ),
    )