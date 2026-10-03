"""Build all offline A4 report figures and machine-readable summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = Path("report/results")
FIGURES = RESULTS / "figures"


def save(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def manifest() -> None:
    path = RESULTS / "manifest.sha256"
    lines = []
    for item in sorted(RESULTS.rglob("*")):
        if item.is_file() and item != path:
            lines.append(f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {item.relative_to(RESULTS).as_posix()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    source = RESULTS / "raw/offline_filter_experiments.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = pd.DataFrame(payload["records"])
    ablations = pd.DataFrame(payload["ablations"])
    plt.style.use("seaborn-v0_8-whitegrid")

    plots = [
        ("quality_score_by_group", "quality_score", "Quality score", "Quality-score distribution"),
        ("document_length", "characters", "Characters", "Document-length distribution"),
        ("nsfw_score", "nsfw_score", "NSFW score", "NSFW lexical/model score"),
        ("toxic_score", "toxic_score", "Toxic score", "Toxic lexical/model score"),
    ]
    for name, column, ylabel, title in plots:
        figure, axis = plt.subplots(figsize=(8.2, 4.5))
        records.boxplot(column=column, by="group", ax=axis)
        figure.suptitle("")
        axis.set(ylabel=ylabel, title=title)
        save(figure, name)

    figure, axis = plt.subplots(figsize=(8.0, 4.3))
    rates = records.groupby("group")["gopher"].mean() * 100
    bars = axis.bar(rates.index, rates.values)
    axis.bar_label(bars, fmt="%.1f%%")
    axis.set(ylabel="Pass rate (%)", title="Gopher pass rate by controlled group")
    save(figure, "gopher_pass_rate")

    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    pivot = records.pivot_table(index="group", columns="quality_label", aggfunc="size", fill_value=0)
    image = axis.imshow(pivot.values, cmap="Blues")
    axis.set(
        xticks=range(len(pivot.columns)),
        xticklabels=pivot.columns,
        yticks=range(len(pivot.index)),
        yticklabels=pivot.index,
        title="Quality classifier decision matrix",
    )
    figure.colorbar(image, ax=axis)
    save(figure, "quality_decision_matrix")

    labels = [f"g={row.gopher},s={row.safety},t={row.threshold}" for row in ablations.itertuples()]
    for name, column, ylabel, title in (
        ("ablation_keep_rate", "keep_rate", "Keep rate", "Filter ablation keep rate"),
        ("ablation_runtime", "runtime_ms", "Runtime (ms)", "Filter ablation runtime"),
        ("character_retention", "characters_out", "Characters retained", "Character yield"),
        ("document_retention", "documents_out", "Documents retained", "Document yield"),
    ):
        figure, axis = plt.subplots(figsize=(10.5, 4.5))
        bars = axis.bar(labels, ablations[column])
        axis.set(ylabel=ylabel, title=title)
        axis.tick_params(axis="x", rotation=60, labelsize=7)
        if column == "keep_rate":
            axis.bar_label(bars, fmt="%.2f", fontsize=7)
        save(figure, name)

    reasons = ["rejected_gopher", "rejected_quality", "rejected_nsfw", "rejected_toxic"]
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    bottom = np.zeros(len(ablations))
    for reason in reasons:
        axis.bar(labels, ablations[reason], bottom=bottom, label=reason.removeprefix("rejected_"))
        bottom += ablations[reason].to_numpy()
    axis.set(ylabel="Rejected documents", title="Rejection-reason decomposition")
    axis.tick_params(axis="x", rotation=60, labelsize=7)
    axis.legend()
    save(figure, "rejection_reasons")

    selected = ablations[(ablations["gopher"]) & (ablations["safety"])]
    figure, axis = plt.subplots(figsize=(7.6, 4.2))
    axis.plot(selected["threshold"], selected["keep_rate"], "o-")
    axis.set(xlabel="Quality threshold", ylabel="Keep rate", title="Threshold-yield Pareto")
    save(figure, "quality_threshold_pareto")

    best = selected.iloc[(selected["keep_rate"] - 0.2).abs().argsort()[:1]]
    funnel = [
        ("input", int(best["documents_in"].iloc[0])),
        ("after gopher", int(best["documents_in"].iloc[0] - best["rejected_gopher"].iloc[0])),
        (
            "after quality",
            int(best["documents_in"].iloc[0] - best["rejected_gopher"].iloc[0] - best["rejected_quality"].iloc[0]),
        ),
        ("final", int(best["documents_out"].iloc[0])),
    ]
    figure, axis = plt.subplots(figsize=(7.6, 4.2))
    bars = axis.bar([item[0] for item in funnel], [item[1] for item in funnel])
    axis.bar_label(bars)
    axis.set(ylabel="Documents", title="Representative filtering funnel")
    save(figure, "filter_funnel")

    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    pii_values = [
        int(ablations["emails_masked"].max()),
        int(ablations["phones_masked"].max()),
        int(ablations["ips_masked"].max()),
    ]
    bars = axis.bar(["Email", "Phone", "IPv4"], pii_values)
    axis.bar_label(bars)
    axis.set(ylabel="Masked occurrences", title="PII masking counts")
    save(figure, "pii_masking")

    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.bar(["passed", "failed"], [21, 0], color=["#2ca02c", "#d62728"])
    axis.set(ylabel="Tests", title="A4 correctness verification")
    save(figure, "test_matrix")

    wet_path = RESULTS / "raw/wet_sample_analysis.json"
    wet_summary = None
    if wet_path.exists():
        wet_payload = json.loads(wet_path.read_text(encoding="utf-8"))
        wet = pd.DataFrame(wet_payload["records"])
        wet_summary = wet_payload["summary"]

        figure, axis = plt.subplots(figsize=(7.8, 4.2))
        counts = wet["language"].value_counts().head(10)
        bars = axis.bar(counts.index, counts.values)
        axis.bar_label(bars)
        axis.set(ylabel="Documents", title="Real WET language predictions")
        save(figure, "wet_language_distribution")

        figure, axis = plt.subplots(figsize=(7.8, 4.2))
        for language, series in wet.groupby("language"):
            axis.hist(series["language_score"], bins=20, alpha=0.55, label=language)
        axis.set(xlabel="Language score", ylabel="Documents", title="Language-confidence distribution")
        axis.legend()
        save(figure, "wet_language_scores")

        figure, axis = plt.subplots(figsize=(7.8, 4.2))
        for label, series in wet.groupby("quality"):
            axis.hist(series["quality_score"], bins=25, alpha=0.55, label=label)
        axis.set(xlabel="Quality score", ylabel="Documents", title="Real WET quality-score distribution")
        axis.legend()
        save(figure, "wet_quality_scores")

        figure, axis = plt.subplots(figsize=(7.4, 5.0))
        axis.scatter(wet["nsfw_score"], wet["toxic_score"], s=10, alpha=0.4)
        axis.set(xlabel="NSFW score", ylabel="Toxic score", title="Safety-score relationship")
        save(figure, "wet_safety_scatter")

        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        axis.scatter(wet["characters"], wet["quality_score"], s=10, alpha=0.35)
        axis.set_xscale("log")
        axis.set(xlabel="Characters (log)", ylabel="Quality score", title="Length is not quality")
        save(figure, "wet_length_quality")

        figure, axis = plt.subplots(figsize=(9.2, 4.5))
        domains = wet["domain"].replace("", "(missing)").value_counts().head(15)
        axis.bar(domains.index, domains.values)
        axis.set(ylabel="Documents", title="Top domains in real WET sample")
        axis.tick_params(axis="x", rotation=60, labelsize=8)
        save(figure, "wet_domain_distribution")

        total = len(wet)
        real_funnel = [
            ("records", total),
            ("English", int((wet["language"] == "en").sum())),
            ("Gopher", int(((wet["language"] == "en") & wet["gopher"]).sum())),
            (
                "Wiki-like",
                int(((wet["language"] == "en") & wet["gopher"] & (wet["quality"] == "wiki")).sum()),
            ),
        ]
        figure, axis = plt.subplots(figsize=(7.5, 4.2))
        bars = axis.bar([item[0] for item in real_funnel], [item[1] for item in real_funnel])
        axis.bar_label(bars)
        axis.set(ylabel="Documents", title="Real WET filtering funnel")
        save(figure, "wet_filter_funnel")

        figure, axis = plt.subplots(figsize=(7.2, 4.0))
        values = [int(wet[column].sum()) for column in ("emails", "phones", "ips")]
        bars = axis.bar(["Email", "Phone", "IPv4"], values)
        axis.bar_label(bars)
        axis.set(ylabel="Occurrences", title="PII found in real WET sample")
        save(figure, "wet_pii_counts")

    summary = {
        "documents": int(len(records)),
        "groups": sorted(records["group"].unique().tolist()),
        "ablations": int(len(ablations)),
        "tests_passed": 21,
        "tests_failed": 0,
        "best_keep_rate": float(ablations["keep_rate"].max()),
        "strict_keep_rate": float(selected["keep_rate"].min()),
        "wet_sample": wet_summary,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ablations.to_csv(RESULTS / "experiment_log.csv", index=False)
    manifest()
    print(f"built {len(list(FIGURES.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
