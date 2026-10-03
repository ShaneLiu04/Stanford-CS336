from __future__ import annotations

import logging
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Thread

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int
from numpy.typing import NDArray

from cs336_scaling.tokenized_data import DCLMDataResult
from cs336_scaling.training.training_config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadedData:
    validation_filepath: Path
    train_filepaths: list[Path]
    n_train_tokens_per_chunk: int
    n_validation_tokens: int


def download_data(
    data: DCLMDataResult, training_config: TrainingConfig
) -> DownloadedData:
    def _n_train_chunks_for_tokens(data: DCLMDataResult, n_train_tokens: int) -> int:
        tokens_per_chunk = data.n_train_tokens // len(data.train_filepaths)
        chunks_with_sequence_starts = math.ceil(n_train_tokens / tokens_per_chunk)
        return min(chunks_with_sequence_starts + 1, len(data.train_filepaths))

    def copy_data_file(source: Path, destination: Path) -> None:
        tmp_destination = destination.with_suffix(f"{destination.suffix}.tmp")
        shutil.copy2(source, tmp_destination)
        tmp_destination.rename(destination)

    def missing_files(
        file_copies: list[tuple[Path, Path]],
    ) -> list[tuple[Path, Path]]:
        return [
            (source, destination)
            for source, destination in file_copies
            if not destination.exists()
        ]

    n_train_chunks = _n_train_chunks_for_tokens(
        data, training_config.total_train_tokens
    )
    required_chunks = _n_train_chunks_for_tokens(
        data, training_config.eval_every_tokens
    )

    local_data_dir = Path("/tmp/data")
    local_data_dir.mkdir(parents=True, exist_ok=True)

    validation_filepath = local_data_dir / data.validation_filepath.name
    tokens_per_chunk = data.n_train_tokens // len(data.train_filepaths)
    if not 1 <= required_chunks <= n_train_chunks:
        raise ValueError(
            f"required_chunks must be between 1 and {n_train_chunks}, "
            f"got {required_chunks=}"
        )

    train_filepaths = [
        local_data_dir / filepath.name
        for filepath in data.train_filepaths[:n_train_chunks]
    ]

    required_file_copies = [
        (data.validation_filepath, validation_filepath),
        *zip(
            data.train_filepaths[:required_chunks],
            train_filepaths[:required_chunks],
            strict=True,
        ),
    ]
    required_file_copies = missing_files(required_file_copies)
    if required_file_copies:
        required_copy_started_at = time.perf_counter()
        logger.info(
            "required_data_copy_started",
            extra={
                "file_count": len(required_file_copies),
                "required_train_chunks": required_chunks,
                "total_train_chunks": n_train_chunks,
                "local_data_dir": str(local_data_dir),
            },
        )
        with ThreadPoolExecutor(
            max_workers=min(8, len(required_file_copies))
        ) as executor:
            for _ in executor.map(
                lambda paths: copy_data_file(*paths), required_file_copies
            ):
                pass
        logger.info(
            "required_data_copy_completed",
            extra={
                "file_count": len(required_file_copies),
                "copy_seconds": round(
                    time.perf_counter() - required_copy_started_at, 3
                ),
            },
        )
    else:
        logger.info(
            "required_data_copy_skipped",
            extra={
                "required_train_chunks": required_chunks,
                "total_train_chunks": n_train_chunks,
                "local_data_dir": str(local_data_dir),
            },
        )

    remaining_train_file_copies = missing_files(
        list(
            zip(
                data.train_filepaths[required_chunks:n_train_chunks],
                train_filepaths[required_chunks:],
                strict=True,
            )
        )
    )

    def copy_remaining_train_files() -> None:
        remaining_copy_started_at = time.perf_counter()
        logger.info(
            "remaining_train_data_copy_started",
            extra={"file_count": len(remaining_train_file_copies)},
        )
        try:
            for source, destination in remaining_train_file_copies:
                copy_data_file(source, destination)
        except Exception:
            logger.exception(
                "remaining_train_data_copy_failed",
                extra={"file_count": len(remaining_train_file_copies)},
            )
            raise
        logger.info(
            "remaining_train_data_copy_completed",
            extra={
                "file_count": len(remaining_train_file_copies),
                "copy_seconds": round(
                    time.perf_counter() - remaining_copy_started_at,
                    3,
                ),
            },
        )

    if remaining_train_file_copies:
        Thread(
            target=copy_remaining_train_files,
            name="train-data-download",
        ).start()
        logger.info(
            "remaining_train_data_copy_scheduled",
            extra={"file_count": len(remaining_train_file_copies)},
        )
    else:
        logger.info("remaining_train_data_copy_skipped")

    return DownloadedData(
        validation_filepath=validation_filepath,
        train_filepaths=train_filepaths,
        n_train_tokens_per_chunk=tokens_per_chunk,
        n_validation_tokens=data.n_validation_tokens,
    )


class Batch[A: (Array, NDArray)](eqx.Module):
    input_ids: Int[A, " seq_len"]
    labels: Int[A, " seq_len"]

    @classmethod
    def train_data(
        cls,
        data: DownloadedData,
        *,
        batch_idx: int,
        training_config: TrainingConfig,
    ) -> Batch[NDArray]:
        seq_len = training_config.seq_len
        n_tokens = training_config.eval_every_tokens

        start = batch_idx * n_tokens
        source_stop = start + n_tokens + 1
        n_train_tokens = data.n_train_tokens_per_chunk * len(data.train_filepaths)
        if source_stop > n_train_tokens:
            raise ValueError(
                f"requested source token range [{start}, {source_stop}) exceeds "
                f"{n_train_tokens=}"
            )

        tokens_per_chunk = data.n_train_tokens_per_chunk
        if tokens_per_chunk % seq_len != 0:
            raise ValueError(
                f"tokens_per_chunk must be divisible by seq_len, got "
                f"{tokens_per_chunk=} and {seq_len=}"
            )

        n_sequences = n_tokens // seq_len
        sequences_per_chunk = tokens_per_chunk // seq_len
        global_sequence_start = batch_idx * n_sequences
        global_sequence_stop = global_sequence_start + n_sequences

        starts: list[NDArray] = []
        remaining_start = global_sequence_start
        while remaining_start < global_sequence_stop:
            chunk_idx = remaining_start // sequences_per_chunk
            chunk_sequence_start = remaining_start % sequences_per_chunk
            chunk_sequence_stop = min(
                global_sequence_stop - chunk_idx * sequences_per_chunk,
                sequences_per_chunk,
            )
            chunk_permutation = np.random.default_rng(chunk_idx).permutation(
                sequences_per_chunk
            )
            starts.append(
                chunk_idx * tokens_per_chunk
                + chunk_permutation[chunk_sequence_start:chunk_sequence_stop] * seq_len
            )
            remaining_start += chunk_sequence_stop - chunk_sequence_start

        sequence_starts = np.concatenate(starts)
        offsets = np.arange(seq_len)
        input_idxs = sequence_starts[:, None] + offsets[None, :]

        def read_tokens(indices: NDArray) -> NDArray:
            flat_indices = indices.reshape(-1)
            flat_tokens = np.empty_like(flat_indices)
            for chunk_idx in np.unique(flat_indices // tokens_per_chunk):
                mask = flat_indices // tokens_per_chunk == chunk_idx
                chunk_tokens = np.load(data.train_filepaths[chunk_idx], mmap_mode="r")
                flat_tokens[mask] = chunk_tokens[flat_indices[mask] % tokens_per_chunk]
            return flat_tokens.reshape(indices.shape)

        input_ids = read_tokens(input_idxs)
        labels = read_tokens(input_idxs + 1)
        return Batch(input_ids=input_ids, labels=labels)

    @classmethod
    def val_data(
        cls,
        data: DownloadedData,
        *,
        training_config: TrainingConfig,
    ) -> Batch[NDArray]:
        seq_len = training_config.seq_len
        n_tokens = training_config.n_val_tokens

        source_stop = n_tokens + 1
        if source_stop > data.n_validation_tokens:
            raise ValueError(
                f"requested source token range [0, {source_stop}) exceeds "
                f"{data.n_validation_tokens=}"
            )

        tokens = np.asarray(
            np.load(data.validation_filepath, mmap_mode="r")[:source_stop]
        )
        n_sequences = n_tokens // seq_len
        offsets = np.arange(seq_len)
        starts = np.arange(n_sequences) * seq_len
        input_idxs = starts[:, None] + offsets[None, :]
        input_ids = tokens[input_idxs]
        labels = tokens[input_idxs + 1]
        return Batch(input_ids=input_ids, labels=labels)

    def sequence_length(self) -> int:
        return self.input_ids.shape[-1]

    def n_sequences(self) -> int:
        match self.input_ids.shape:
            case (x, _):
                return x
            case (_,):
                return 1
            case _:
                raise RuntimeError("expected input_ids to have rank 1 or 2")

    def to_jax(self) -> Batch[Array]:
        return jax.tree.map(jnp.array, self)
