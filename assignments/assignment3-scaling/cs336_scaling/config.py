import datetime as dt
import os
from dataclasses import dataclass
from functools import cache
from typing import Literal, cast

from dotenv import load_dotenv

LaunchMode = Literal["local", "modal"]


@dataclass(frozen=True)
class Settings:
    database_url: str
    internal_api_key: str
    api_base_url: str
    launch_mode: LaunchMode = "local"
    max_concurrent_workers: int = 100
    zero_running_user_headroom: int = 10
    dispatch_interval_seconds: float = 5
    reconcile_interval_seconds: float = 30
    total_budget_seconds: float = 12.0 * 3600.0
    modal_app_name: str = "a3-scaling"

    @property
    def soft_cap_for_users_with_running_jobs(self) -> int:
        return self.max_concurrent_workers - self.zero_running_user_headroom


@cache
def settings_from_env() -> Settings:
    load_dotenv()
    match os.environ.get("DB_ENV", "dev"):
        case "dev":
            database_url_env_var = "DATABASE_URL_DEV"
        case "prod":
            database_url_env_var = "DATABASE_URL_PROD"
        case db_env:
            raise ValueError(f"DB_ENV must be 'dev' or 'prod', got {db_env!r}")

    return Settings(
        database_url=os.getenv(database_url_env_var, ""),
        internal_api_key=os.getenv(
            "INTERNAL_API_KEY",
            os.environ["INTERNAL_API_KEY"],
        ),
        api_base_url=os.getenv(
            "API_BASE_URL",
            "http://127.0.0.1:8000",
        ),
        launch_mode=cast(LaunchMode, os.getenv("LAUNCH_MODE", "local")),
    )


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)
