import functools
from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from cs336_scaling.config import settings_from_env


class Base(DeclarativeBase):
    pass


@functools.cache
def get_engine():
    return create_engine(settings_from_env().database_url, pool_pre_ping=True)


@functools.cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_engine(),
        autoflush=False,
        expire_on_commit=False,
    )


def get_session() -> Generator[Session, None, None]:
    with get_session_factory()() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def init_db() -> None:
    import cs336_scaling.db.tables  # noqa: F401

    Base.metadata.create_all(bind=get_engine())
