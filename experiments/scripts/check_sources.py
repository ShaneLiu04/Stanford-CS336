"""Validate the experiments source registry and local Markdown links."""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "id",
    "title",
    "url",
    "author",
    "accessed",
    "version",
    "topics",
    "type",
    "license",
    "mirror",
    "sha256",
    "confidence",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-http", action="store_true")
    args = parser.parse_args()
    sources = json.loads((ROOT / "SOURCES.yaml").read_text(encoding="utf-8"))
    errors: list[str] = []
    for index, source in enumerate(sources):
        missing = REQUIRED - set(source)
        if missing:
            errors.append(f"source[{index}] missing {sorted(missing)}")
        if source.get("mirror") and source.get("license") in {
            "UNKNOWN",
            "SEE_SOURCE",
            "ARXIV",
        }:
            errors.append(
                f"{source.get('id')}: mirrored without explicit redistribution license"
            )
        if source.get("mirror") and not source.get("sha256"):
            errors.append(f"{source.get('id')}: mirrored source missing sha256")
    for field in ("id", "url"):
        duplicates = [
            value
            for value, count in Counter(item[field] for item in sources).items()
            if count > 1
        ]
        if duplicates:
            errors.append(f"duplicate {field}: {duplicates}")

    broken_links = []
    for markdown in ROOT.rglob("*.md"):
        text = markdown.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path = (markdown.parent / target.split("#", 1)[0]).resolve()
            if not path.exists():
                broken_links.append(f"{markdown.relative_to(ROOT)} -> {target}")
    errors.extend(f"broken local link: {item}" for item in broken_links)

    http_results = {}
    if args.check_http:
        for source in sources:
            try:
                request = urllib.request.Request(
                    source["url"],
                    method="HEAD",
                    headers={"User-Agent": "cs336-study-index/1.0"},
                )
                with urllib.request.urlopen(request, timeout=15) as response:
                    http_results[source["id"]] = response.status
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                http_results[source["id"]] = type(exc).__name__

    report = [
        "# Source registry report",
        "",
        f"- Sources: {len(sources)}",
        f"- Official: {sum(item['type'] == 'official' for item in sources)}",
        f"- Papers: {sum(item['type'] == 'paper' for item in sources)}",
        f"- Community: {sum(item['type'] == 'community' for item in sources)}",
        f"- Mirrored third-party items: {sum(bool(item['mirror']) for item in sources)}",
        f"- Errors: {len(errors)}",
        "",
    ]
    if errors:
        report += ["## Errors", *[f"- {error}" for error in errors], ""]
    if http_results:
        report += [
            "## HTTP status",
            *[f"- `{key}`: {value}" for key, value in http_results.items()],
        ]
    (ROOT / "source-report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"validated {len(sources)} sources and local Markdown links")


if __name__ == "__main__":
    main()
