import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class TqdmLogger:
    """File-like class redirecting tqdm progress bar to given logging logger."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def write(self, msg: str) -> None:
        msg = msg.strip("\r\n")
        if msg:
            self.logger.info(msg)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


@contextmanager
def tqdm_for_logger(logger: logging.Logger, **kwargs: Any) -> Iterator[Any]:
    from tqdm.auto import tqdm

    with tqdm(
        file=TqdmLogger(logger),
        dynamic_ncols=False,
        mininterval=10.0,
        maxinterval=60.0,
        **kwargs,
    ) as progress:
        yield progress


def format_params(n: int) -> str:
    for scale, suffix in [(10**12, "T"), (10**9, "B"), (10**6, "M"), (10**3, "K")]:
        if n >= scale:
            s = f"{n / scale:.1f}".rstrip("0").rstrip(".")
            return f"{s}{suffix}"
    return str(n)


def format_bytes(n: int) -> str:
    for scale, suffix in [
        (2**40, "TiB"),
        (2**30, "GiB"),
        (2**20, "MiB"),
        (2**10, "KiB"),
    ]:
        if n >= scale:
            s = f"{n / scale:.1f}".rstrip("0").rstrip(".")
            return f"{s}{suffix}"
    return str(n)


def require_int(num: float) -> int:
    if num != int(num):
        raise ValueError(f"expected an integer, got {num}")
    return int(num)
