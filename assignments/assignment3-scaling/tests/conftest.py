from collections.abc import Generator
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from cs336_scaling.app import create_app
from cs336_scaling.config import settings_from_env
from cs336_scaling.db import get_engine, get_session_factory, init_db


def _clear_app_caches() -> None:
    get_session_factory.cache_clear()
    get_engine.cache_clear()
    settings_from_env.cache_clear()


@pytest.fixture
def _test_database_url(monkeypatch: pytest.MonkeyPatch) -> Generator[str, None, None]:
    admin_engine = create_engine(
        "postgresql+psycopg:///postgres", isolation_level="AUTOCOMMIT"
    )

    test_database_name = f"cs336_scaling_test_{uuid4().hex}"
    test_database_url = f"postgresql+psycopg:///{test_database_name}"

    try:
        with admin_engine.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{test_database_name}"'))

        monkeypatch.setenv(
            "DATABASE_URL_DEV",
            test_database_url,
        )
        monkeypatch.setenv("DB_ENV", "dev")
        _clear_app_caches()
        init_db()

        yield test_database_url
    finally:
        get_engine().dispose()
        _clear_app_caches()

        with admin_engine.connect() as connection:
            connection.execute(
                text(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = :database_name
                      AND pid <> pg_backend_pid()
                    """
                ),
                {"database_name": test_database_name},
            )
            connection.execute(text(f'DROP DATABASE IF EXISTS "{test_database_name}"'))

        admin_engine.dispose()


@pytest.fixture
def db_session_factory(_test_database_url: str) -> sessionmaker[Session]:
    return get_session_factory()


@pytest.fixture
def client(_test_database_url: str) -> Generator[TestClient, None, None]:
    with TestClient(create_app()) as test_client:
        yield test_client
