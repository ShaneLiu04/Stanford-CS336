"""Controlled offline filter ablations for the self-study report."""

from __future__ import annotations

import json
import time
from pathlib import Path

from cs336_data.pipeline import filter_documents
from cs336_data.quality import classify_quality, gopher_quality_filter
from cs336_data.safety import classify_nsfw, classify_toxic_speech


def main() -> None:
    fixtures = Path("tests/fixtures")
    wiki = (fixtures / "high_quality_wiki_reference.txt").read_text(encoding="utf-8")
    cc = (fixtures / "low_quality_cc.txt").read_text(encoding="utf-8")
    groups = {
        "wiki": [wiki + f"\nReference section {index}." for index in range(80)],
        "cc": [cc + f"\ntracking={index}" for index in range(80)],
        "short": [f"Short page {index}." for index in range(80)],
        "numeric": [("123 456 789 " * 80) + str(index) for index in range(80)],
        "ellipsis": [
            ("\n".join(["unfinished thought..." for _ in range(70)] + ["normal sentence." for _ in range(30)]))
            + str(index)
            for index in range(80)
        ],
    }
    records = []
    for group, documents in groups.items():
        for document in documents:
            label, quality_score = classify_quality(document)
            records.append(
                {
                    "group": group,
                    "characters": len(document),
                    "quality_label": label,
                    "quality_score": quality_score,
                    "gopher": gopher_quality_filter(document),
                    "nsfw_score": classify_nsfw(document)[1],
                    "toxic_score": classify_toxic_speech(document)[1],
                }
            )
    ablations = []
    all_documents = [document for documents in groups.values() for document in documents]
    for use_gopher in (False, True):
        for use_safety in (False, True):
            for threshold in (0.3, 0.5, 0.7):
                started = time.perf_counter()
                kept, stats = filter_documents(
                    all_documents,
                    use_gopher=use_gopher,
                    use_safety=use_safety,
                    quality_threshold=threshold,
                )
                ablations.append(
                    {
                        "gopher": use_gopher,
                        "safety": use_safety,
                        "threshold": threshold,
                        "kept": len(kept),
                        "keep_rate": len(kept) / len(all_documents),
                        "runtime_ms": (time.perf_counter() - started) * 1000,
                        **stats.__dict__,
                    }
                )
    output = Path("report/results/raw/offline_filter_experiments.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"records": records, "ablations": ablations}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
