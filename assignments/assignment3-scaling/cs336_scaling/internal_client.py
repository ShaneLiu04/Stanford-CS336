import logging
import os
from collections.abc import Mapping

import requests
from pydantic import BaseModel

from cs336_scaling.api.internal import FinishPayload
from cs336_scaling.config import settings_from_env
from cs336_scaling.log.internal_client_errors import InternalAPIError
from cs336_scaling.schemas.event import WorkerEvent

logger = logging.getLogger(__name__)


def post_internal_worker_event(
    experiment_id: int,
    event_type: WorkerEvent,
    log_extra: Mapping[str, object],
) -> None:
    settings = settings_from_env()
    api_base_url = os.getenv("TAILSCALE_URL", settings.api_base_url).rstrip("/")
    event_log_extra = {
        **log_extra,
        "experiment_id": experiment_id,
        "event_log_type": event_type.log_type,
    }
    logger.info("posting_internal_worker_event_started", extra=event_log_extra)
    response = _post_internal_worker(
        api_base_url=api_base_url,
        internal_api_key=settings.internal_api_key,
        experiment_id=experiment_id,
        endpoint="event",
        payload=event_type,
        log_name="posting_internal_worker_event",
        log_extra=event_log_extra,
    )
    logger.info(
        "posting_internal_worker_event_completed",
        extra=event_log_extra | {"status_code": response.status_code},
    )


def post_internal_finish(
    experiment_id: int,
    payload: FinishPayload,
    log_extra: Mapping[str, object],
) -> None:
    settings = settings_from_env()
    api_base_url = os.getenv("TAILSCALE_URL", settings.api_base_url).rstrip("/")
    finish_log_extra = {
        **log_extra,
        "experiment_id": experiment_id,
        "result_type": payload.result_type,
        "used_runtime_seconds": payload.used_runtime_seconds,
    }
    logger.info("posting_internal_finish_started", extra=finish_log_extra)
    response = _post_internal_worker(
        api_base_url=api_base_url,
        internal_api_key=settings.internal_api_key,
        experiment_id=experiment_id,
        endpoint="finish",
        payload=payload,
        log_name="posting_internal_finish",
        log_extra=finish_log_extra,
    )
    logger.info(
        "posting_internal_finish_completed",
        extra=finish_log_extra | {"status_code": response.status_code},
    )


def _post_internal_worker(
    *,
    api_base_url: str,
    internal_api_key: str,
    experiment_id: int,
    endpoint: str,
    payload: BaseModel,
    log_name: str,
    log_extra: Mapping[str, object],
) -> requests.Response:
    try:
        response = requests.post(
            f"{api_base_url}/internal/worker/{experiment_id}/{endpoint}",
            json=payload.model_dump(mode="json"),
            headers={"X-Internal-Key": internal_api_key},
            proxies={
                "http": "http://localhost:1080",
                "https": "http://localhost:1080",
            },
        )
        if not response.ok:
            raise InternalAPIError(response)
    except Exception as exc:
        logger.exception(
            f"{log_name}_failed",
            extra={
                **log_extra,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    return response
