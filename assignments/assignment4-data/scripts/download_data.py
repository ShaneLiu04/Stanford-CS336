import bz2
import gzip
import re
import shutil
import urllib.request
from pathlib import Path

import modal

from cs336_data.common import get_shared_assets_path
from cs336_data.modal_utils import VOLUME_MOUNTS, app, build_image
from cs336_data.wet_files import EnglishWetFiles


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, timeout=60 * 60 * 12, max_containers=128)
def extract_wiki_urls(shard: str) -> list[str]:
    dump_date = "20260501"
    tmp_dir = Path("/tmp/wiki")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dump = tmp_dir / shard
    url_re = re.compile(
        r"\b(?:https?|telnet|gopher|file|wais|ftp):[\w/#~:.?+=&%@!\-.:?\\-]+?(?=[.:?\-]*(?:[^\w/#~:.?+=&%@!\-.:?\-]|$))"
    )

    print(f"[wiki] downloading {shard}", flush=True)
    urllib.request.urlretrieve(f"https://dumps.wikimedia.org/enwiki/{dump_date}/{shard}", dump)
    urls = []
    with bz2.open(dump, "rt", errors="ignore") as f:
        for line in f:
            if refs := re.search("&lt;ref&gt(.*)&lt;/ref&gt;", line):
                urls.extend(url_re.findall(refs.group(0)))
    dump.unlink(missing_ok=True)
    return urls


def download_offline_files(*, root_path: Path) -> None:
    paloma_out = root_path / "tokenized_paloma_c4_100_domains_validation.bin"
    if not paloma_out.exists():
        print(f"[huggingface] downloading {paloma_out.name}", flush=True)
        urllib.request.urlretrieve(
            "https://huggingface.co/datasets/brunborg/cs336-a4/resolve/main/tokenized_paloma_c4_100_domains_validation.bin",
            paloma_out,
        )

    cc = root_path / "CC"
    cc.mkdir(parents=True, exist_ok=True)
    for kind, out_name in [("warc", "example.warc.gz"), ("wet", "example.warc.wet.gz")]:
        out = cc / out_name
        if not out.exists():
            print(f"[cc] downloading {out_name}", flush=True)
            with urllib.request.urlopen(
                f"https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-12/{kind}.paths.gz"
            ) as r:
                first_path = gzip.decompress(r.read()).decode().splitlines()[0]
            urllib.request.urlretrieve(f"https://data.commoncrawl.org/{first_path}", out)

    for rel_path, url in [
        (
            "classifiers/lid.176.bin",
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
        ),
        (
            "classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin",
            "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech/resolve/main/model.bin",
        ),
        (
            "classifiers/dolma_fasttext_nsfw_jigsaw_model.bin",
            "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw/resolve/main/model.bin",
        ),
    ]:
        out = root_path / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            print(f"[file] downloading {rel_path}", flush=True)
            urllib.request.urlretrieve(url, out)


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, timeout=60 * 60 * 12)
def main(offline_only: bool = False):
    root_path = get_shared_assets_path()
    download_offline_files(root_path=root_path)
    if offline_only:
        return

    dump_date = "20260501"
    base_url = f"https://dumps.wikimedia.org/enwiki/{dump_date}/"
    html = urllib.request.urlopen(base_url).read().decode()
    shards = sorted(
        set(re.findall(rf"enwiki-{dump_date}-pages-articles-multistream[0-9]+\.xml-p[0-9]+p[0-9]+\.bz2", html))
    )
    wiki_out = root_path / "wiki/enwiki-20260501-extracted_urls.txt.gz"
    if not wiki_out.exists():
        wiki_out.parent.mkdir(parents=True, exist_ok=True)
        tmp_out = Path("/tmp") / wiki_out.name
        tmp_out.unlink(missing_ok=True)
        print(f"[wiki] extracting {len(shards)} shards", flush=True)
        with gzip.open(tmp_out, "wt") as f:
            for urls in (
                [extract_wiki_urls.local(shard) for shard in shards]
                if modal.is_local()
                else extract_wiki_urls.map(shards)
            ):
                for url in urls:
                    f.write(url + "\n")
        shutil.copy2(tmp_out, wiki_out)
        tmp_out.unlink(missing_ok=True)
        print(f"[wiki] wrote {wiki_out}", flush=True)

    english_wet_files = EnglishWetFiles()
    wet_file_paths = english_wet_files.load_or_create()
    print(f"downloaded {len(wet_file_paths)} including {wet_file_paths[0]=}")


@app.local_entrypoint()
def modal_main(offline_only: bool = False):
    main.remote(offline_only=offline_only)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Only download files needed for running the assignment offline; skip full WET/wiki data creation.",
    )
    args = parser.parse_args()
    main.local(offline_only=args.offline_only)
