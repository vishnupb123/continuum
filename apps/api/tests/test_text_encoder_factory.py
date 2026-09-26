from unittest.mock import patch

from app.services.features.text_encoder.factory import (
    _get_mpnet_encoder,
    get_text_encoder,
)


def test_blank_revision_is_normalized_to_none(monkeypatch):
    monkeypatch.setattr(
        "app.services.features.text_encoder.factory."
        "settings.text_encoder_provider",
        "mpnet",
    )

    monkeypatch.setattr(
        "app.services.features.text_encoder.factory."
        "settings.text_encoder_revision",
        "",
    )

    _get_mpnet_encoder.cache_clear()

    with patch(
        "app.services.features.text_encoder.factory."
        "_get_mpnet_encoder"
    ) as get_mpnet:
        get_text_encoder()

        args = get_mpnet.call_args.args

        assert args[1] is None