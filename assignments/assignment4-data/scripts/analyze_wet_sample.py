"""Profile real Common Crawl WET records with the complete filter stack."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from warcio.archiveiterator import ArchiveIterator

from cs336_data.extract_lang import identify_language
from cs336_data.quality import classify_quality, gopher_quality_filter
from cs336_data.safety import (
    classify_nsfw,
    classify_toxic_speech,
    mask_emails,
    mask_ips,
    mask_phone_numbers,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wet", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("report/results/raw/wet_sample_analysis.json"))
    args = parser.parse_args()
    records = []
    with gzip.open(args.wet, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type != "conversion":
                continue
            text = record.content_stream().read().decode("utf-8", errors="replace")
            url = record.rec_headers.get_header("WARC-Target-URI") or ""
            language, language_score = identify_language(text)
            quality, quality_score = classify_quality(text)
            nsfw, nsfw_score = classify_nsfw(text)
            toxic, toxic_score = classify_toxic_speech(text)
            masked, emails = mask_emails(text)
            masked, phones = mask_phone_numbers(masked)
            _, ips = mask_ips(masked)
            records.append(
                {
                    "url": url,
                    "domain": urlparse(url).netloc.lower(),
                    "characters": len(text),
                    "language": language,
                    "language_score": language_score,
                    "quality": quality,
                    "quality_score": quality_score,
                    "gopher": gopher_quality_filter(text),
                    "nsfw": nsfw,
                    "nsfw_score": nsfw_score,
                    "toxic": toxic,
                    "toxic_score": toxic_score,
                    "emails": emails,
                    "phones": phones,
                    "ips": ips,
                }
            )
            if len(records) >= args.limit:
                break
    summary = {
        "records": len(records),
        "languages": Counter(item["language"] for item in records),
        "gopher_pass": sum(item["gopher"] for item in records),
        "wiki_quality": sum(item["quality"] == "wiki" for item in records),
        "nsfw": sum(item["nsfw"] == "nsfw" for item in records),
        "toxic": sum(item["toxic"] == "toxic" for item in records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
