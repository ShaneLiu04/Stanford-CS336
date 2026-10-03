import json
import logging
import re
import sys
import time
import traceback
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

DEFAULT_LOG_FILE_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_LOG_FILE_BACKUP_COUNT = 0
DEFAULT_REQUEST_BODY_LOG_TOKEN_LIMIT = 512
DEFAULT_REQUEST_BODY_LOG_CHAR_LIMIT = 8192
API_FILE_LOGGER_NAME = "cs336_scaling.api.requests"
REQUEST_ID_HEADER = "X-Request-ID"
_MANAGED_FILE_HANDLER = "_cs336_scaling_file_handler"
_BODY_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

_original_stdout: Any | None = None
_original_stderr: Any | None = None

_RESERVED_LOG_RECORD_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "taskName",
    "thread",
    "threadName",
}


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info is not None:
            payload["exception"] = "".join(traceback.format_exception(*record.exc_info))

        return json.dumps(payload, default=str, separators=(",", ":"))


class _NonDeletingRotatingFileHandler(RotatingFileHandler):
    def doRollover(self) -> None:
        if self.stream:
            self.stream.close()
            self.stream = None

        log_path = Path(self.baseFilename)
        if log_path.exists():
            log_path.rename(_next_rotated_log_path(log_path))

        if not self.delay:
            self.stream = self._open()


class _LogFileStream:
    def __init__(self, stream: Any, logger_name: str, level: int) -> None:
        self.stream = stream
        self.logger_name = logger_name
        self.level = level
        self._buffer = ""

    def write(self, text: str) -> int:
        written = self.stream.write(text)
        self.stream.flush()

        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                _write_to_file_handlers(self.logger_name, self.level, line)
        return written

    def flush(self) -> None:
        if self._buffer:
            _write_to_file_handlers(self.logger_name, self.level, self._buffer)
            self._buffer = ""
        self.stream.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()

    @property
    def encoding(self) -> str | None:
        return self.stream.encoding


def configure_logging(
    *,
    log_file: str | Path | None,
    app: Any | None = None,
    log_level: str | int = logging.INFO,
    log_format: str = "text",
    log_file_level: str | int = logging.DEBUG,
    log_file_max_bytes: int = DEFAULT_LOG_FILE_MAX_BYTES,
    log_file_backup_count: int = DEFAULT_LOG_FILE_BACKUP_COUNT,
) -> None:
    del log_file_backup_count

    console_level = _log_level(log_level)
    file_level = _log_level(log_file_level)
    resolved_log_file = _optional_path(log_file)
    file_enabled = resolved_log_file is not None
    console_formatter: logging.Formatter
    if log_format.lower() in {"json", "jsonl"}:
        console_formatter = JSONFormatter()
    else:
        console_formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(
        min(console_level, file_level) if file_enabled else console_level
    )

    _ensure_stream_handler(
        root_logger,
        console_formatter,
        console_level,
    )
    _ensure_file_handler(
        root_logger,
        JSONFormatter(),
        file_level,
        resolved_log_file,
        max_bytes=log_file_max_bytes,
    )

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(logger_name).setLevel(console_level)

    if app is not None:
        _add_api_observability(app)


def _add_api_observability(app: Any) -> None:
    request_logger = logging.getLogger(API_FILE_LOGGER_NAME)

    @app.middleware("http")
    async def log_request(request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER, uuid4().hex)
        request.state.request_id = request_id
        await _capture_request_body_for_logging(request)
        started_at = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            request_logger.exception(
                "http_request_failed",
                extra=_request_log_extra(request, request_id)
                | {"duration_ms": _elapsed_ms(started_at), "status_code": 500},
            )
            raise

        response.headers[REQUEST_ID_HEADER] = request_id
        request_logger.info(
            "http_request_completed",
            extra=_request_log_extra(request, request_id)
            | {
                "duration_ms": _elapsed_ms(started_at),
                "status_code": response.status_code,
            },
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def log_request_validation_exception(
        request: Request,
        exc: RequestValidationError,
    ):
        request_logger.warning(
            "request_validation_failed",
            extra=_request_log_extra(request) | {"validation_errors": exc.errors()},
        )
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(StarletteHTTPException)
    async def log_http_exception(request: Request, exc: StarletteHTTPException):
        request_logger.warning(
            "http_exception",
            extra=_request_log_extra(request)
            | {"status_code": exc.status_code, "detail": exc.detail},
        )
        return await http_exception_handler(request, exc)


def _ensure_stream_handler(
    logger: logging.Logger,
    formatter: logging.Formatter,
    level: int,
) -> None:
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler,
            RotatingFileHandler,
        ):
            handler.setFormatter(formatter)
            handler.setLevel(level)
            return

    stream = _original_stdout if isinstance(sys.stdout, _LogFileStream) else sys.stdout
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    logger.addHandler(handler)


def _request_log_extra(
    request: Request,
    request_id: str | None = None,
) -> dict[str, Any]:
    return {
        "request_id": request_id
        or getattr(request.state, "request_id", None)
        or request.headers.get(REQUEST_ID_HEADER),
        "method": request.method,
        "path": request.url.path,
        "client_host": request.client.host if request.client is not None else None,
        "user_agent": request.headers.get("user-agent"),
        "user_sunet_id": getattr(request.state, "user_sunet_id", None),
        "request_body": getattr(request.state, "request_body_log", None),
    }


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


async def _capture_request_body_for_logging(request: Request) -> None:
    body = await request.body()
    request.state.request_body_log = _request_body_for_log(
        body,
        content_type=request.headers.get("content-type"),
        token_limit=DEFAULT_REQUEST_BODY_LOG_TOKEN_LIMIT,
        char_limit=DEFAULT_REQUEST_BODY_LOG_CHAR_LIMIT,
    )

    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = receive


def _request_body_for_log(
    body: bytes,
    *,
    content_type: str | None,
    token_limit: int,
    char_limit: int,
) -> dict[str, Any] | None:
    if not body:
        return None

    text = body.decode("utf-8", errors="replace")
    preview = _bounded_text_preview(
        text,
        token_limit=token_limit,
        char_limit=char_limit,
    )
    body_log: dict[str, Any] = {
        "content_type": content_type,
        "bytes": len(body),
        "token_limit": token_limit,
        "char_limit": char_limit,
        "truncated": preview["truncated"],
    }

    if _is_json_content_type(content_type) and not preview["truncated"]:
        try:
            body_log["json"] = json.loads(text)
            return body_log
        except json.JSONDecodeError:
            pass

    body_log["preview"] = preview["text"]
    body_log["preview_tokens"] = preview["tokens"]
    return body_log


def _bounded_text_preview(
    text: str,
    *,
    token_limit: int,
    char_limit: int,
) -> dict[str, Any]:
    stop = min(len(text), char_limit)
    tokens = 0
    for match in _BODY_TOKEN_PATTERN.finditer(text):
        if match.start() >= stop:
            break
        tokens += 1
        if tokens >= token_limit:
            stop = min(stop, match.end())
            break

    return {
        "text": text[:stop],
        "tokens": tokens,
        "truncated": stop < len(text),
    }


def _is_json_content_type(content_type: str | None) -> bool:
    if content_type is None:
        return False
    media_type = content_type.split(";", maxsplit=1)[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")


def _ensure_file_handler(
    logger: logging.Logger,
    formatter: logging.Formatter,
    level: int,
    log_path: Path | None,
    *,
    max_bytes: int,
) -> None:
    if log_path is None:
        _remove_managed_file_handlers()
        _restore_print_streams()
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)

    for handler in logger.handlers:
        if not getattr(handler, _MANAGED_FILE_HANDLER, False):
            continue
        if isinstance(
            handler, _NonDeletingRotatingFileHandler
        ) and handler.baseFilename == str(log_path):
            handler.setFormatter(formatter)
            handler.setLevel(level)
            handler.maxBytes = max_bytes
            _capture_print_streams()
            return
        logger.removeHandler(handler)
        handler.close()

    handler = _NonDeletingRotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=0,
    )
    setattr(handler, _MANAGED_FILE_HANDLER, True)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    logger.addHandler(handler)
    _capture_print_streams()


def _next_rotated_log_path(log_path: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    candidate = log_path.with_name(f"{log_path.name}.{timestamp}")
    suffix = 1
    while candidate.exists():
        candidate = log_path.with_name(f"{log_path.name}.{timestamp}.{suffix}")
        suffix += 1
    return candidate


def _remove_managed_file_handlers() -> None:
    for logger_name in ("", API_FILE_LOGGER_NAME):
        logger = logging.getLogger(logger_name)
        for handler in list(logger.handlers):
            if getattr(handler, _MANAGED_FILE_HANDLER, False):
                logger.removeHandler(handler)
                handler.close()


def _capture_print_streams() -> None:
    global _original_stderr, _original_stdout

    if not isinstance(sys.stdout, _LogFileStream):
        _original_stdout = sys.stdout
        sys.stdout = _LogFileStream(sys.stdout, "stdout", logging.INFO)

    if not isinstance(sys.stderr, _LogFileStream):
        _original_stderr = sys.stderr
        sys.stderr = _LogFileStream(sys.stderr, "stderr", logging.ERROR)


def _restore_print_streams() -> None:
    global _original_stderr, _original_stdout

    if isinstance(sys.stdout, _LogFileStream) and _original_stdout is not None:
        sys.stdout = _original_stdout
        _original_stdout = None

    if isinstance(sys.stderr, _LogFileStream) and _original_stderr is not None:
        sys.stderr = _original_stderr
        _original_stderr = None


def _write_to_file_handlers(logger_name: str, level: int, message: str) -> None:
    record = logging.LogRecord(
        logger_name,
        level,
        pathname="<captured-stream>",
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    for handler in logging.getLogger().handlers:
        if getattr(handler, _MANAGED_FILE_HANDLER, False):
            if record.levelno >= handler.level:
                handler.handle(record)


def _log_level(value: str | int) -> int:
    if isinstance(value, int):
        return value

    return logging.getLevelNamesMapping().get(value.upper(), logging.INFO)


def _optional_path(value: str | Path | None) -> Path | None:
    if value is not None:
        return Path(value)
    return None
