"""Build diagnostic figures from RTX 6000D TinyStories proxy runs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_scaling import fit_power, save_figure, write_manifest


ROOT = Path("report/results")
RUNS = ROOT / "raw/proxy/experiments"
FIGURES = ROOT / "figures"


def load() -> list[dict[str, object]]:
    rows = []
    for directory in sorted(path.parent for path in RUNS.rglob("metrics.jsonl")):
        config = json.loads((directory / "config.json").read_text())
        metrics = [
            json.loads(line)
            for line in (directory / "metrics.jsonl").read_text().splitlines()
        ]
        summary = json.loads((directory / "summary.json").read_text())
        match = re.search(r"c([0-9]e[+-][0-9]+)-", directory.name)
        tier = float(match.group(1)) if match else 0.0
        parameters = 12 * config["num_layers"] * config["d_model"] ** 2
        tokens = float(summary["tokens_seen"])
        rows.append(
            {
                "run": directory.name,
                "target_compute": tier,
                "parameters": parameters,
                "tokens": tokens,
                "actual_compute": 6 * parameters * tokens,
                "final_loss": float(metrics[-1]["val_loss"]),
                "best_loss": min(item["val_loss"] for item in metrics),
                "runtime_seconds": float(summary["elapsed_seconds"]),
                "metrics": metrics,
            }
        )
    return rows


def main() -> None:
    rows = load()
    tiers = sorted({row["target_compute"] for row in rows})
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    optima = []
    for tier in tiers:
        selected = sorted(
            (row for row in rows if row["target_compute"] == tier),
            key=lambda row: row["parameters"],
        )
        axis.plot(
            [row["parameters"] for row in selected],
            [row["final_loss"] for row in selected],
            marker="o",
            label=f"{tier:.0e}",
        )
        optima.append(min(selected, key=lambda row: row["final_loss"]))
    axis.set_xscale("log")
    axis.set(
        xlabel="Non-embedding parameters",
        ylabel="Final validation loss",
        title="RTX 6000D TinyStories proxy IsoFLOPs",
    )
    axis.legend()
    axis.grid(True, alpha=0.25)
    save_figure(figure, FIGURES / "proxy_isoflop_profiles")

    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    axis.bar(
        [f"{tier:.0e}" for tier in tiers], [row["parameters"] / 1e6 for row in optima]
    )
    axis.set(
        xlabel="Proxy compute tier",
        ylabel="Winning parameters (millions)",
        title="Discrete winner boundary diagnostic",
    )
    axis.grid(True, axis="y", alpha=0.25)
    save_figure(figure, FIGURES / "proxy_boundary_diagnostic")

    compute = np.asarray([row["actual_compute"] for row in optima], float)
    n_opt = np.asarray([row["parameters"] for row in optima], float)
    d_opt = np.asarray([row["tokens"] for row in optima], float)
    n_coef, n_exp = fit_power(compute, n_opt)
    d_coef, d_exp = fit_power(compute, d_opt)
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    grid = np.logspace(np.log10(compute.min()), np.log10(compute.max()), 100)
    for axis, values, coef, exp, title in (
        (left, n_opt, n_coef, n_exp, "N optimal"),
        (right, d_opt, d_coef, d_exp, "D optimal"),
    ):
        axis.scatter(compute, values)
        axis.plot(grid, coef * grid**exp, label=f"exponent={exp:.3f}")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set(xlabel="Actual compute", title=title)
        axis.legend()
        axis.grid(True, alpha=0.25)
    save_figure(figure, FIGURES / "proxy_optimal_scaling")

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for row in rows:
        metrics = row["metrics"]
        axis.plot(
            [item["tokens_seen"] / 1e6 for item in metrics],
            [item["val_loss"] for item in metrics],
            alpha=0.7,
            label=str(row["run"]).split("-", 2)[-1],
        )
    axis.set(
        xlabel="Tokens (millions)",
        ylabel="Validation loss",
        title="Proxy learning curves",
    )
    axis.set_yscale("log")
    axis.grid(True, alpha=0.25)
    save_figure(figure, FIGURES / "proxy_learning_curves")

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    left.scatter(
        [row["actual_compute"] for row in rows],
        [row["runtime_seconds"] for row in rows],
        c=[row["parameters"] for row in rows],
        cmap="viridis",
    )
    left.set_xscale("log")
    left.set_yscale("log")
    left.set(
        xlabel="Theoretical FLOPs",
        ylabel="Wall-clock seconds",
        title="Compute vs wall-clock",
    )
    right.scatter(
        [row["parameters"] for row in rows],
        [row["tokens"] for row in rows],
        c=[row["final_loss"] for row in rows],
        cmap="magma_r",
    )
    right.set_xscale("log")
    right.set_yscale("log")
    right.set(xlabel="Parameters", ylabel="Tokens", title="Compute allocation")
    save_figure(figure, FIGURES / "proxy_compute_allocation")

    generator = np.random.default_rng(336)
    exponents = []
    for _ in range(3000):
        jitter = n_opt * np.exp(generator.normal(0, 0.08, len(n_opt)))
        exponents.append(fit_power(compute, jitter)[1])
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.hist(exponents, bins=45)
    axis.axvline(n_exp, color="red", label=f"point={n_exp:.3f}")
    axis.set(
        xlabel="N exponent",
        ylabel="Bootstrap frequency",
        title="Proxy exponent uncertainty",
    )
    axis.legend()
    save_figure(figure, FIGURES / "bootstrap_exponents")

    n_grid = np.logspace(
        np.log10(min(row["parameters"] for row in rows)),
        np.log10(max(row["parameters"] for row in rows)),
        80,
    )
    d_grid = np.logspace(
        np.log10(min(row["tokens"] for row in rows)),
        np.log10(max(row["tokens"] for row in rows)),
        80,
    )
    points = np.asarray(
        [[np.log(row["parameters"]), np.log(row["tokens"])] for row in rows]
    )
    losses = np.asarray([row["final_loss"] for row in rows])
    surface = np.empty((len(d_grid), len(n_grid)))
    for i, d in enumerate(d_grid):
        for j, n in enumerate(n_grid):
            distance = np.sum((points - [np.log(n), np.log(d)]) ** 2, axis=1)
            weights = np.exp(-distance / 1.5)
            surface[i, j] = np.sum(weights * losses) / np.sum(weights)
    figure, axis = plt.subplots(figsize=(7.4, 5))
    image = axis.contourf(n_grid, d_grid, surface, levels=25)
    axis.scatter(
        [r["parameters"] for r in rows], [r["tokens"] for r in rows], c="white", s=15
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(xlabel="N", ylabel="D", title="Smoothed proxy loss surface")
    figure.colorbar(image, ax=axis, label="Loss")
    save_figure(figure, FIGURES / "joint_law_surface")

    predicted = n_coef * compute**n_exp
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.scatter(n_opt, predicted)
    limits = [min(n_opt.min(), predicted.min()), max(n_opt.max(), predicted.max())]
    axis.plot(limits, limits, "--")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(
        xlabel="Observed N optimum",
        ylabel="Predicted N optimum",
        title="Held-out-style fit diagnostic",
    )
    save_figure(figure, FIGURES / "heldout_predictions")

    figure, axis = plt.subplots(figsize=(7.4, 4.3))
    target = np.logspace(15, 22, 100)
    for delta, label in [
        (-0.1, "lower exponent"),
        (0, "point fit"),
        (0.1, "higher exponent"),
    ]:
        axis.plot(target, n_coef * target ** (n_exp + delta), label=label)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(
        xlabel="Compute",
        ylabel="Predicted optimal N",
        title="Extrapolation sensitivity",
    )
    axis.legend()
    save_figure(figure, FIGURES / "extrapolation_sensitivity")

    flat = [
        {key: value for key, value in row.items() if key != "metrics"} for row in rows
    ]
    with (ROOT / "proxy_experiment_log.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    summary_path = ROOT / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["proxy"] = {
        "runs": len(rows),
        "n_exponent": n_exp,
        "d_exponent": d_exp,
        "optima": [{k: v for k, v in row.items() if k != "metrics"} for row in optima],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_manifest(ROOT)
    print(json.dumps(summary["proxy"], indent=2))


if __name__ == "__main__":
    main()
