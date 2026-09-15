from sqlalchemy.orm import Session

from app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from app.models.user import User
from app.repositories.users import get_user_by_email


class EmailAlreadyRegisteredError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


def register_user(
    db: Session,
    *,
    email: str,
    password: str,
    display_name: str | None,
) -> User:
    normalized_email = email.strip().lower()

    existing_user = get_user_by_email(
        db,
        normalized_email,
    )

    if existing_user is not None:
        raise EmailAlreadyRegisteredError

    user = User(
        email=normalized_email,
        password_hash=hash_password(password),
        display_name=display_name.strip() if display_name else None,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def authenticate_user(
    db: Session,
    *,
    email: str,
    password: str,
) -> tuple[User, str]:
    normalized_email = email.strip().lower()

    user = get_user_by_email(
        db,
        normalized_email,
    )

    if (
        user is None
        or not user.is_active
        or not verify_password(
            password,
            user.password_hash,
        )
    ):
        raise InvalidCredentialsError

    token = create_access_token(user.id)

    return user, token