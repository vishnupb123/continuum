import uuid

import pytest

from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_is_hashed():
    password = "ContinuumTestPassword123!"

    hashed = hash_password(password)

    assert hashed != password
    assert password not in hashed


def test_correct_password_verifies():
    password = "ContinuumTestPassword123!"
    hashed = hash_password(password)

    assert verify_password(password, hashed) is True


def test_wrong_password_fails():
    hashed = hash_password("CorrectPassword123!")

    assert (
        verify_password(
            "WrongPassword123!",
            hashed,
        )
        is False
    )


def test_access_token_round_trip():
    user_id = uuid.uuid4()

    token = create_access_token(user_id)
    decoded_user_id = decode_access_token(token)

    assert decoded_user_id == user_id


def test_invalid_access_token_fails():
    with pytest.raises(ValueError):
        decode_access_token("this-is-not-a-valid-token")