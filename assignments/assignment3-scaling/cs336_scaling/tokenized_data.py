import math
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from furu import Furu
from huggingface_hub import HfApi
from numpy.lib.format import open_memmap
from transformers import AutoTokenizer

from cs336_scaling.utils import tqdm_for_logger

_LOCAL_RAW_TOKEN_DIR = Path("/tmp/rawotokendata")
_LOCAL_FINAL_SHARD_DIR = Path("/tmp/final_dclm_shards")


def _close_memmap(arr: np.memmap | None) -> None:
    if arr is None:
        return

    mmap_obj = getattr(arr, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


@dataclass
class TokenizedDCLMChunkResult:
    tokens_path: Path
    lengths_path: Path
    n_sequences: int
    n_tokens: int


class ShuffledDCLMShardUrls(Furu[list[str]]):
    repo_path: str = "mlfoundations/dclm-baseline-1.0-parquet"
    seed: int = 67

    def _create(self) -> list[str]:
        files = HfApi().list_repo_files(self.repo_path, repo_type="dataset")
        sorted_files = sorted([file for file in files if file.endswith(".parquet")])
        rng = random.Random(self.seed)
        shuffled_list = rng.sample(sorted_files, len(sorted_files))
        return [f"hf://datasets/{self.repo_path}@main/{path}" for path in shuffled_list]


class TokenizedDCLMChunk(Furu[TokenizedDCLMChunkResult]):
    url: str
    tokenizer_name: str = "NousResearch/Llama-2-7b-hf"

    def _create(self) -> TokenizedDCLMChunkResult:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        assert self.tokenizer_name == "NousResearch/Llama-2-7b-hf", (
            "right now, we are assuming the vocab fits in uint16. make sure to either switch from uint16 or use tokenizer with less than 65k tokens"
        )

        tokens_path = self.data_dir / "tokens.npy"
        lengths_path = self.data_dir / "lengths.npy"

        from transformers import TokenizersBackend

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        assert isinstance(tokenizer, TokenizersBackend)

        eos_token_id = tokenizer.eos_token_id
        assert eos_token_id is not None, f"{self.tokenizer_name} has no eos_token_id"

        self.logger.info(f"Reading parquet shard: {self.url}")
        df = pl.read_parquet(self.url, columns=["text"])

        tokenized = tokenizer(
            df["text"].to_list(),
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        lengths = np.fromiter(
            (len(ids) + 1 for ids in tokenized), dtype=np.uint32, count=len(tokenized)
        )
        n_tokens = int(lengths.sum(dtype=np.uint64))

        tokens = open_memmap(
            tokens_path,
            mode="w+",
            dtype=np.uint16,
            shape=(n_tokens,),
        )
        try:
            write_pos = 0
            for ids, seq_len in zip(tokenized, lengths, strict=True):
                write_stop = write_pos + int(seq_len)
                tokens[write_pos : write_stop - 1] = ids
                tokens[write_stop - 1] = eos_token_id
                write_pos = write_stop

            assert write_pos == n_tokens
            tokens.flush()
        finally:
            _close_memmap(tokens)

        np.save(lengths_path, lengths)

        return TokenizedDCLMChunkResult(
            tokens_path=tokens_path,
            lengths_path=lengths_path,
            n_sequences=len(lengths),
            n_tokens=n_tokens,
        )


@dataclass(frozen=True)
class DCLMDataResult:
    validation_filepath: Path
    train_filepaths: list[Path]
    n_train_tokens: int
    n_validation_tokens: int
    n_source_chunks: int


@dataclass(frozen=True)
class _LocalTokenizedChunk:
    tokens_path: Path
    lengths_path: Path
    n_sequences: int
    n_tokens: int


@dataclass(frozen=True)
class _ShardJob:
    split: str
    shard_idx: int
    token_start: int
    token_stop: int


@dataclass(frozen=True)
class _MaterializedShard:
    split: str
    shard_idx: int
    local_path: Path
    final_path: Path
    n_tokens: int


@dataclass(frozen=True)
class _WrittenShard:
    split: str
    shard_idx: int
    final_path: Path
    n_tokens: int


@dataclass(frozen=True)
class _ShuffleRun:
    run_idx: int
    tokens_path: Path
    lengths_path: Path
    n_sequences: int
    n_tokens: int


class DCLMData(Furu[DCLMDataResult]):
    tokens_per_chunk: int = 2**29  # this is 1GiB since uint16 is 2 bytes
    total_tokens: int = 500_000_000_000
    shuffled_dclm_urls: ShuffledDCLMShardUrls = field(
        default_factory=ShuffledDCLMShardUrls
    )
    validation_tokens: int = 2**29
    seed = 67
    n_workers = 32
    shuffle_run_target_bytes: int = 128 * 1024**3
    merge_block_target_bytes: int = 1024**3

    def _create(self) -> DCLMDataResult:
        self.data_dir.mkdir(parents=True, exist_ok=True)

        total_tokens_padded = (
            math.ceil(self.total_tokens / self.tokens_per_chunk) * self.tokens_per_chunk
        )

        self.logger.info(f"making {total_tokens_padded} train tokens")

        requested_tokens = total_tokens_padded + self.validation_tokens

        self.logger.info(f"loading {requested_tokens} cached tokenized chunks")

        source_chunks = self._load_cached_tokenized_chunks(requested_tokens)

        local_chunks = self._copy_chunks_to_local_ssd(source_chunks)

        shutil.rmtree(self._local_final_shard_dir, ignore_errors=True)
        self._local_final_shard_dir.mkdir(parents=True, exist_ok=True)

        train_jobs = self._make_shard_jobs(
            split="train", token_start=0, n_tokens=total_tokens_padded
        )
        validation_jobs = self._make_shard_jobs(
            split="validation",
            token_start=total_tokens_padded,
            n_tokens=self.validation_tokens,
            max_shard_tokens=self.validation_tokens,
        )

        split_widths = {
            "train": len(str(max(0, len(train_jobs) - 1))),
            "validation": len(str(max(0, len(validation_jobs) - 1))),
        }

        materialized = self._write_shuffled_final_shards(
            local_chunks=local_chunks,
            jobs=[*train_jobs, *validation_jobs],
            split_widths=split_widths,
        )

        written = self._copy_materialized_shards_to_data_dir(materialized)

        train_filepaths = [
            shard.final_path
            for shard in sorted(written, key=lambda shard: shard.shard_idx)
            if shard.split == "train"
        ]
        validation_filepaths = [
            shard.final_path
            for shard in sorted(written, key=lambda shard: shard.shard_idx)
            if shard.split == "validation"
        ]
        assert len(validation_filepaths) == 1

        return DCLMDataResult(
            validation_filepath=validation_filepaths[0],
            train_filepaths=train_filepaths,
            n_train_tokens=sum(
                shard.n_tokens for shard in written if shard.split == "train"
            ),
            n_validation_tokens=sum(
                shard.n_tokens for shard in written if shard.split == "validation"
            ),
            n_source_chunks=len(source_chunks),
        )

    def _load_cached_tokenized_chunks(
        self,
        requested_tokens: int,
    ) -> list[TokenizedDCLMChunkResult]:
        chunks: list[TokenizedDCLMChunkResult] = []
        n_tokens = 0

        for i, url in enumerate(self.shuffled_dclm_urls.load_or_create()):
            chunk = TokenizedDCLMChunk(url=url).try_load()
            assert chunk is not None, f"missing cached tokenized DCLM chunk for {url}"

            chunks.append(chunk)
            n_tokens += int(chunk.n_tokens)

            if n_tokens >= requested_tokens:
                break

            if i % 500 == 0:
                i += 1
                self.logger.info(f"seen {i=} chunks and {n_tokens=:_}")

        assert n_tokens >= requested_tokens, (
            f"found {n_tokens:,} cached tokens, need {requested_tokens:,}"
        )
        return chunks

    def _copy_chunks_to_local_ssd(
        self,
        chunks: list[TokenizedDCLMChunkResult],
    ) -> list[_LocalTokenizedChunk]:
        self.logger.info(
            f"Copying {len(chunks) * 2:,} tokenized files from Modal volume "
            f"to {_LOCAL_RAW_TOKEN_DIR}"
        )

        shutil.rmtree(_LOCAL_RAW_TOKEN_DIR, ignore_errors=True)
        _LOCAL_RAW_TOKEN_DIR.mkdir(parents=True, exist_ok=True)

        local_chunks: list[_LocalTokenizedChunk | None] = [None] * len(chunks)

        with tqdm_for_logger(
            self.logger,
            desc="Copying tokenized files to local SSD",
            total=len(chunks) * 2,
            unit="file",
        ) as progress:
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(self._copy_one_chunk_to_local_ssd, idx, chunk): idx
                    for idx, chunk in enumerate(chunks)
                }

                for future in as_completed(futures):
                    local_chunks[futures[future]] = future.result()
                    progress.update(2)

        self.logger.info(f"Copied {len(chunks) * 2:,} tokenized files to local SSD")

        assert all(chunk is not None for chunk in local_chunks)
        return [chunk for chunk in local_chunks if chunk is not None]

    def _copy_one_chunk_to_local_ssd(
        self,
        idx: int,
        chunk: TokenizedDCLMChunkResult,
    ) -> _LocalTokenizedChunk:
        dst_dir = _LOCAL_RAW_TOKEN_DIR / f"chunk_{idx:06d}"
        dst_dir.mkdir(parents=True, exist_ok=True)

        tokens_path = dst_dir / "tokens.npy"
        lengths_path = dst_dir / "lengths.npy"

        shutil.copy2(chunk.tokens_path, tokens_path)
        shutil.copy2(chunk.lengths_path, lengths_path)

        return _LocalTokenizedChunk(
            tokens_path=tokens_path,
            lengths_path=lengths_path,
            n_sequences=chunk.n_sequences,
            n_tokens=chunk.n_tokens,
        )

    def _make_shard_jobs(
        self,
        split: str,
        token_start: int,
        n_tokens: int,
        max_shard_tokens: int | None = None,
    ) -> list[_ShardJob]:
        shard_tokens = max_shard_tokens or self.tokens_per_chunk
        n_shards = math.ceil(n_tokens / shard_tokens)

        return [
            _ShardJob(
                split=split,
                shard_idx=shard_idx,
                token_start=token_start + shard_idx * shard_tokens,
                token_stop=min(
                    token_start + (shard_idx + 1) * shard_tokens,
                    token_start + n_tokens,
                ),
            )
            for shard_idx in range(n_shards)
        ]

    def _write_shuffled_final_shards(
        self,
        local_chunks: list[_LocalTokenizedChunk],
        jobs: list[_ShardJob],
        split_widths: dict[str, int],
    ) -> list[_MaterializedShard]:
        self.logger.info(
            f"Writing {len(jobs):,} final DCLM shards to {self._local_final_shard_dir}"
        )

        runs_dir = self._local_final_shard_dir / "shuffle_runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        run_groups = self._make_shuffle_run_groups(local_chunks)
        self.logger.info(
            f"Building {len(run_groups):,} shuffled DCLM runs in {runs_dir} "
            f"with a target size of {self.shuffle_run_target_bytes / 1024**3:.1f} GiB"
        )

        runs: list[_ShuffleRun] = []
        try:
            with tqdm_for_logger(
                self.logger,
                desc="Building shuffled DCLM runs on local SSD",
                total=len(run_groups),
                unit="run",
            ) as progress:
                for run_idx, chunks in enumerate(run_groups):
                    runs.append(self._build_shuffle_run(run_idx, chunks, runs_dir))
                    progress.update()

            self.logger.info(
                f"Removing local raw tokenized chunks from {_LOCAL_RAW_TOKEN_DIR}"
            )
            shutil.rmtree(_LOCAL_RAW_TOKEN_DIR, ignore_errors=True)

            return self._merge_shuffle_runs_to_final_shards(
                runs=runs,
                jobs=jobs,
                split_widths=split_widths,
            )
        finally:
            shutil.rmtree(runs_dir, ignore_errors=True)

    def _make_shuffle_run_groups(
        self, chunks: list[_LocalTokenizedChunk]
    ) -> list[list[_LocalTokenizedChunk]]:
        target_tokens = max(
            1, self.shuffle_run_target_bytes // np.dtype(np.uint16).itemsize
        )
        groups: list[list[_LocalTokenizedChunk]] = []
        current: list[_LocalTokenizedChunk] = []
        current_tokens = 0

        for chunk in chunks:
            if current and current_tokens + chunk.n_tokens > target_tokens:
                groups.append(current)
                current = []
                current_tokens = 0

            current.append(chunk)
            current_tokens += chunk.n_tokens

        if current:
            groups.append(current)

        return groups

    def _build_shuffle_run(
        self,
        run_idx: int,
        chunks: list[_LocalTokenizedChunk],
        runs_dir: Path,
    ) -> _ShuffleRun:
        n_sequences = sum(chunk.n_sequences for chunk in chunks)
        n_tokens = sum(chunk.n_tokens for chunk in chunks)
        assert n_sequences > 0
        assert n_tokens > 0

        source_tokens = np.empty(n_tokens, dtype=np.uint16)
        source_lengths = np.empty(n_sequences, dtype=np.uint32)

        token_pos = 0
        seq_pos = 0
        for chunk in chunks:
            tokens = self._load_npy_memmap(chunk.tokens_path)
            lengths = self._load_npy_memmap(chunk.lengths_path)
            try:
                assert tokens.ndim == 1
                assert tokens.dtype == np.uint16
                assert lengths.ndim == 1
                assert lengths.dtype == np.uint32
                assert len(lengths) == chunk.n_sequences

                next_token_pos = token_pos + chunk.n_tokens
                next_seq_pos = seq_pos + chunk.n_sequences
                source_tokens[token_pos:next_token_pos] = tokens[:]
                source_lengths[seq_pos:next_seq_pos] = lengths[:]
                token_pos = next_token_pos
                seq_pos = next_seq_pos
            finally:
                self._close_memmap(tokens)
                self._close_memmap(lengths)

        assert token_pos == n_tokens
        assert seq_pos == n_sequences

        source_offsets = np.empty(n_sequences, dtype=np.uint64)
        source_offsets[0] = 0
        np.cumsum(source_lengths[:-1], dtype=np.uint64, out=source_offsets[1:])

        rng = np.random.default_rng(self.seed + run_idx + 1)
        shuffled_sequence_idxs = rng.permutation(n_sequences)

        tokens_path = runs_dir / f"run_{run_idx:05d}.tokens.npy"
        lengths_path = runs_dir / f"run_{run_idx:05d}.lengths.npy"
        run_tokens = open_memmap(
            tokens_path,
            mode="w+",
            dtype=np.uint16,
            shape=(n_tokens,),
        )
        run_lengths = open_memmap(
            lengths_path,
            mode="w+",
            dtype=np.uint32,
            shape=(n_sequences,),
        )

        write_pos = 0
        try:
            for out_seq_idx, source_seq_idx in enumerate(shuffled_sequence_idxs):
                seq_len = int(source_lengths[source_seq_idx])
                source_start = int(source_offsets[source_seq_idx])
                source_stop = source_start + seq_len
                write_stop = write_pos + seq_len

                run_tokens[write_pos:write_stop] = source_tokens[
                    source_start:source_stop
                ]
                run_lengths[out_seq_idx] = seq_len
                write_pos = write_stop

            assert write_pos == n_tokens
            run_tokens.flush()
            run_lengths.flush()
        finally:
            self._close_memmap(run_tokens)
            self._close_memmap(run_lengths)

        return _ShuffleRun(
            run_idx=run_idx,
            tokens_path=tokens_path,
            lengths_path=lengths_path,
            n_sequences=n_sequences,
            n_tokens=n_tokens,
        )

    def _merge_shuffle_runs_to_final_shards(
        self,
        runs: list[_ShuffleRun],
        jobs: list[_ShardJob],
        split_widths: dict[str, int],
    ) -> list[_MaterializedShard]:
        run_tokens = [self._load_npy_memmap(run.tokens_path) for run in runs]
        run_lengths = [self._load_npy_memmap(run.lengths_path) for run in runs]
        remaining_sequences = np.asarray(
            [run.n_sequences for run in runs], dtype=np.int64
        )
        remaining_tokens = np.asarray([run.n_tokens for run in runs], dtype=np.int64)
        sequence_cursors = np.zeros(len(runs), dtype=np.int64)
        sequence_token_cursors = np.zeros(len(runs), dtype=np.int64)
        token_cursors = np.zeros(len(runs), dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        materialized: list[_MaterializedShard] = []

        try:
            for tokens, lengths, run in zip(run_tokens, run_lengths, runs):
                assert tokens.ndim == 1
                assert tokens.dtype == np.uint16
                assert lengths.ndim == 1
                assert lengths.dtype == np.uint32
                assert len(lengths) == run.n_sequences

            with tqdm_for_logger(
                self.logger,
                desc="Writing final DCLM shards to local SSD",
                total=len(jobs),
                unit="shard",
            ) as progress:
                for job in jobs:
                    materialized.append(
                        self._write_one_shard_from_shuffle_runs(
                            job=job,
                            run_tokens=run_tokens,
                            run_lengths=run_lengths,
                            remaining_sequences=remaining_sequences,
                            remaining_tokens=remaining_tokens,
                            sequence_cursors=sequence_cursors,
                            sequence_token_cursors=sequence_token_cursors,
                            token_cursors=token_cursors,
                            rng=rng,
                            filename_width=split_widths[job.split],
                        )
                    )
                    progress.update()

            return materialized
        finally:
            for tokens in run_tokens:
                self._close_memmap(tokens)
            for lengths in run_lengths:
                self._close_memmap(lengths)

    def _write_one_shard_from_shuffle_runs(
        self,
        job: _ShardJob,
        run_tokens: list[np.memmap],
        run_lengths: list[np.memmap],
        remaining_sequences: np.ndarray,
        remaining_tokens: np.ndarray,
        sequence_cursors: np.ndarray,
        sequence_token_cursors: np.ndarray,
        token_cursors: np.ndarray,
        rng: np.random.Generator,
        filename_width: int,
    ) -> _MaterializedShard:
        final_path = (
            self.data_dir / job.split / f"{job.shard_idx:0{filename_width}d}.npy"
        )
        local_split_dir = self._local_final_shard_dir / job.split
        local_split_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_split_dir / final_path.name
        target_tokens = job.token_stop - job.token_start

        out = open_memmap(
            local_path,
            mode="w+",
            dtype=np.uint16,
            shape=(target_tokens,),
        )

        try:
            write_pos = 0

            while write_pos < target_tokens and int(remaining_sequences.sum()) > 0:
                for run_idx, sequence_count in self._sample_run_blocks(
                    rng=rng,
                    remaining_sequences=remaining_sequences,
                    remaining_tokens=remaining_tokens,
                ):
                    if write_pos >= target_tokens:
                        break

                    tokens_written = self._copy_sequence_block_from_shuffle_run(
                        out=out,
                        out_start=write_pos,
                        max_out_tokens=target_tokens - write_pos,
                        run_idx=run_idx,
                        max_sequences=sequence_count,
                        run_tokens=run_tokens,
                        run_lengths=run_lengths,
                        remaining_sequences=remaining_sequences,
                        remaining_tokens=remaining_tokens,
                        sequence_cursors=sequence_cursors,
                        sequence_token_cursors=sequence_token_cursors,
                        token_cursors=token_cursors,
                    )
                    write_pos += tokens_written

            if write_pos != target_tokens:
                raise RuntimeError(
                    f"wrote {write_pos:,} tokens for {job.split} shard {job.shard_idx}; "
                    f"expected {target_tokens:,}"
                )

            out.flush()
        finally:
            self._close_memmap(out)

        return _MaterializedShard(
            split=job.split,
            shard_idx=job.shard_idx,
            local_path=local_path,
            final_path=final_path,
            n_tokens=write_pos,
        )

    def _sample_run_blocks(
        self,
        rng: np.random.Generator,
        remaining_sequences: np.ndarray,
        remaining_tokens: np.ndarray,
    ) -> list[tuple[int, int]]:
        total_sequences = int(remaining_sequences.sum())
        assert total_sequences > 0

        total_tokens = int(remaining_tokens.sum())
        avg_tokens_per_sequence = max(1, math.ceil(total_tokens / total_sequences))
        block_tokens = max(
            1, self.merge_block_target_bytes // np.dtype(np.uint16).itemsize
        )
        block_sequences = min(
            total_sequences,
            max(1, block_tokens // avg_tokens_per_sequence),
        )

        active = np.flatnonzero(remaining_sequences > 0)
        counts = self._sample_counts_without_replacement(
            rng=rng,
            colors=remaining_sequences[active],
            n=block_sequences,
        )
        assert int(counts.sum()) == block_sequences

        block_idxs = np.flatnonzero(counts > 0)
        rng.shuffle(block_idxs)
        return [(int(active[idx]), int(counts[idx])) for idx in block_idxs]

    def _sample_counts_without_replacement(
        self,
        rng: np.random.Generator,
        colors: np.ndarray,
        n: int,
    ) -> np.ndarray:
        colors = np.asarray(colors, dtype=np.int64)
        out = np.zeros_like(colors)
        if len(colors) == 0:
            return out

        remaining_total = int(colors.sum())
        remaining_draw = int(n)

        for i in range(len(colors) - 1):
            if remaining_draw == 0:
                break

            ngood = int(colors[i])
            draw = int(
                rng.hypergeometric(
                    ngood=ngood,
                    nbad=remaining_total - ngood,
                    nsample=remaining_draw,
                )
            )
            out[i] = draw
            remaining_total -= ngood
            remaining_draw -= draw

        out[-1] = remaining_draw
        return out

    def _copy_sequence_block_from_shuffle_run(
        self,
        out: np.memmap,
        out_start: int,
        max_out_tokens: int,
        run_idx: int,
        max_sequences: int,
        run_tokens: list[np.memmap],
        run_lengths: list[np.memmap],
        remaining_sequences: np.ndarray,
        remaining_tokens: np.ndarray,
        sequence_cursors: np.ndarray,
        sequence_token_cursors: np.ndarray,
        token_cursors: np.ndarray,
    ) -> int:
        assert max_out_tokens > 0
        assert max_sequences > 0

        seq_cursor = int(sequence_cursors[run_idx])
        seq_token_cursor = int(sequence_token_cursors[run_idx])
        sequence_count = min(max_sequences, int(remaining_sequences[run_idx]))
        assert sequence_count > 0

        lengths = np.asarray(
            run_lengths[run_idx][seq_cursor : seq_cursor + sequence_count],
            dtype=np.int64,
        )
        assert len(lengths) == sequence_count
        assert np.all(lengths > 0)

        if seq_token_cursor:
            lengths[0] -= seq_token_cursor
            assert lengths[0] > 0

        cumulative_tokens = np.cumsum(lengths, dtype=np.int64)
        consumed_tokens = min(int(cumulative_tokens[-1]), max_out_tokens)
        assert consumed_tokens > 0

        token_start = int(token_cursors[run_idx])
        token_stop = token_start + consumed_tokens
        out[out_start : out_start + consumed_tokens] = run_tokens[run_idx][
            token_start:token_stop
        ]

        token_cursors[run_idx] += consumed_tokens
        remaining_tokens[run_idx] -= consumed_tokens

        completed_sequences = int(
            np.searchsorted(cumulative_tokens, consumed_tokens, side="right")
        )
        if completed_sequences == 0:
            sequence_token_cursors[run_idx] += consumed_tokens
        else:
            sequence_cursors[run_idx] += completed_sequences
            remaining_sequences[run_idx] -= completed_sequences
            completed_tokens = int(cumulative_tokens[completed_sequences - 1])
            sequence_token_cursors[run_idx] = consumed_tokens - completed_tokens

        assert remaining_tokens[run_idx] >= 0
        assert remaining_sequences[run_idx] >= 0
        return consumed_tokens

    def _copy_materialized_shards_to_data_dir(
        self, materialized: list[_MaterializedShard]
    ) -> list[_WrittenShard]:
        self.logger.info(
            f"Copying {len(materialized):,} final DCLM shards from "
            f"{self._local_final_shard_dir} to {self.data_dir}"
        )

        written: list[_WrittenShard] = []
        with tqdm_for_logger(
            self.logger,
            desc="Copying final DCLM shards",
            total=len(materialized),
            unit="shard",
        ) as progress:
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [
                    executor.submit(self._copy_one_materialized_shard, shard)
                    for shard in materialized
                ]

                for future in as_completed(futures):
                    written.append(future.result())
                    progress.update()

        return written

    def _copy_one_materialized_shard(self, shard: _MaterializedShard) -> _WrittenShard:
        shard.final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(shard.local_path, shard.final_path)

        return _WrittenShard(
            split=shard.split,
            shard_idx=shard.shard_idx,
            final_path=shard.final_path,
            n_tokens=shard.n_tokens,
        )

    @property
    def _local_final_shard_dir(self) -> Path:
        return _LOCAL_FINAL_SHARD_DIR / self.artifact_hash

    def _load_npy_memmap(self, path: Path) -> np.memmap:
        arr = np.load(path, mmap_mode="r")
        assert isinstance(arr, np.memmap)
        return arr

    def _close_memmap(self, arr: np.memmap | None) -> None:
        _close_memmap(arr)
