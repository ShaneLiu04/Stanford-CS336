from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy import select

from cs336_scaling.config import settings_from_env
from cs336_scaling.db import get_session_factory
from cs336_scaling.db.tables import UserTable


@dataclass(frozen=True)
class AuthenticatedUser:
    sunet_id: str
    api_key: str


def get_current_user(
    request: Request,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> AuthenticatedUser:
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing X-API-Key",
        )

    with get_session_factory()() as session:
        user = session.scalar(select(UserTable).where(UserTable.api_key == api_key))

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid API key",
        )

    authenticated_user = AuthenticatedUser(sunet_id=user.sunet_id, api_key=user.api_key)
    request.state.user_sunet_id = authenticated_user.sunet_id
    return authenticated_user


CurrentUserDep = Annotated[AuthenticatedUser, Depends(get_current_user)]


def require_internal_key(
    internal_key: Annotated[str | None, Header(alias="X-Internal-Key")] = None,
) -> None:
    if internal_key != settings_from_env().internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid internal key",
        )
