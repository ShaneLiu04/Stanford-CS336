import gzip
import shutil
import tempfile
import urllib.request
from functools import cached_property
from io import BytesIO
from pathlib import Path

import fasttext
import modal
import polars as pl
from warcio.archiveiterator import ArchiveIterator
from warcio.warcwriter import WARCWriter

from cs336_data.common import get_shared_assets_path
from cs336_data.modal_utils import VOLUME_MOUNTS, app, build_image
from furu import Furu

BASE_URL = "https://data.commoncrawl.org/"


class _EnglishWetFile(Furu[Path]):
    chunk_urls: tuple[str, ...]

    def _create(self) -> Path:
        output_path = self.data_dir / "data.warc.wet.gz"

        self.logger.info("Loading English language identifier")
        model = fasttext.load_model(str(get_shared_assets_path() / "classifiers" / "lid.176.bin"))

        def is_english(text: str) -> bool:
            labels, scores = model.predict(text.replace("\n", " "), k=1)
            return bool(labels and labels[0].removeprefix("__label__") == "en" and scores[0] >= 0.7)

        total_text = 0
        skipped_text = 0
        self.logger.info("Processing WET chunk (%d files)", len(self.chunk_urls))

        with tempfile.NamedTemporaryFile(
            delete=False,
            dir="/tmp",
            suffix=f".{output_path.name}",
        ) as temp_output_file:
            temp_output_path = Path(temp_output_file.name)

        with gzip.open(temp_output_path, "wb") as output_stream:
            writer = WARCWriter(output_stream, gzip=False)
            for wet_url in self.chunk_urls:
                local_wet_path = Path("/tmp") / wet_url.split("/")[-1]
                if not local_wet_path.exists():
                    self.logger.info("Downloading %s to %s", wet_url, local_wet_path)
                    urllib.request.urlretrieve(wet_url, local_wet_path)
                else:
                    self.logger.info("Using cached WET file %s", local_wet_path)
                with gzip.open(local_wet_path, "rb") as input_stream:
                    for rec in ArchiveIterator(input_stream):
                        if rec.rec_type != "conversion":
                            writer.write_record(rec)
                            continue
                        payload = rec.content_stream().read()
                        text = payload.decode("utf-8", errors="replace")
                        total_text += len(text)

                        if is_english(text):
                            rec.raw_stream = BytesIO(payload)
                            writer.write_record(rec)
                        else:
                            skipped_text += len(text)
        shutil.copy2(temp_output_path, output_path)
        temp_output_path.unlink(missing_ok=True)

        self.logger.info(
            "Finished WET chunk: wrote %s, kept %.2f%% of text",
            output_path,
            100 * (total_text - skipped_text) / total_text if total_text else 0,
        )
        return output_path

    @cached_property
    def storage_root(self) -> Path:
        return get_shared_assets_path() / "furu"


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, timeout=60 * 60 * 12, max_containers=128)
def make_wet_file_on_modal(wet_file: _EnglishWetFile) -> Path:
    return wet_file.load_or_create()


class EnglishWetFiles(Furu[list[Path]]):
    n_files: int = 2500
    group_size: int = 4
    shuffle_seed: int = 336
    crawl_id: str = "CC-MAIN-2026-17"

    def _create(self) -> list[Path]:
        assert self.n_files % self.group_size == 0
        wet_paths = f"{BASE_URL}crawl-data/{self.crawl_id}/wet.paths.gz"
        self.logger.info("Loading WET paths from %s", wet_paths)

        wet_urls = list(
            BASE_URL
            + pl.read_csv(
                wet_paths,
                has_header=False,
                new_columns=["wet_path"],
            ).sample(n=self.n_files, shuffle=True, seed=self.shuffle_seed, with_replacement=False)["wet_path"]
        )

        self.logger.info("Selected %d WET files for crawl %s", len(wet_urls), self.crawl_id)

        wet_files: list[_EnglishWetFile] = []
        for chunk_idx in range(0, len(wet_urls), self.group_size):
            chunk_urls = tuple(wet_urls[chunk_idx : chunk_idx + self.group_size])
            wet_files.append(_EnglishWetFile(chunk_urls=chunk_urls))

        self.logger.info("Making %d english wet files", len(wet_files))

        wet_data_paths: list[Path] = []
        if modal.is_local():
            self.logger.info("downloading wet files locally")
            for wet_file_idx, wet_file in enumerate(wet_files):
                wet_data_paths.append(wet_file.load_or_create())
                self.logger.info(
                    "Completed %d/%d WET chunks",
                    wet_file_idx,
                    len(wet_files),
                )
        else:
            self.logger.info("downloading wet files on remote")

            wet_data_paths = list(make_wet_file_on_modal.map(wet_files))
            self.logger.info("Completed %d remote WET chunks", len(wet_data_paths))

            repo_path = get_shared_assets_path() / "english-wet-data"
            repo_path.mkdir(exist_ok=False)
            self.logger.info("Linking remote WET outputs into %s", repo_path)

            source_link = repo_path / ".source"
            if source_link.exists() or source_link.is_symlink():
                self.logger.info("Replacing existing source link %s", source_link)
                source_link.unlink()
            source_link.symlink_to(self.data_dir)
            self.logger.info("Linked source data directory %s -> %s", source_link, self.data_dir)

            for wet_data_idx, wet_data_path in enumerate(wet_data_paths):
                link_path = repo_path / f"{wet_data_idx:05d}-{wet_data_path.name}"
                if link_path.exists() or link_path.is_symlink():
                    self.logger.info("Replacing existing WET chunk link %s", link_path)
                    link_path.unlink()
                link_path.symlink_to(wet_data_path)
                self.logger.info("Linked WET chunk %d: %s -> %s", wet_data_idx, link_path, wet_data_path)

        self.logger.info("Finished creating %d English WET files", len(wet_data_paths))
        return wet_data_paths

    @cached_property
    def storage_root(self) -> Path:
        return get_shared_assets_path() / "furu"
