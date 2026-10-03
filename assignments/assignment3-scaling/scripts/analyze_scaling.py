"""Fit IsoFLOP power laws and build reproducible report assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def fit_power(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    exponent, intercept = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(intercept)), float(exponent)


def select_optima(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    budgets = sorted({float(row["compute_budget"]) for row in rows})
    result = []
    for compute in budgets:
        candidates = [row for row in rows if float(row["compute_budget"]) == compute]
        best = min(candidates, key=lambda row: float(row["final_loss"]))
        parameters = float(best["parameters"])
        result.append(
            {
                "compute_budget": compute,
                "parameters": parameters,
                "tokens": compute / (6 * parameters),
                "final_loss": float(best["final_loss"]),
            }
        )
    return result


def bootstrap_power(
    x: np.ndarray, y: np.ndarray, targets: np.ndarray, repetitions: int = 5000
) -> np.ndarray:
    generator = np.random.default_rng(336)
    predictions = []
    for _ in range(repetitions):
        indices = generator.integers(0, len(x), len(x))
        if len(np.unique(indices)) < 2:
            continue
        coefficient, exponent = fit_power(x[indices], y[indices])
        predictions.append(coefficient * targets**exponent)
    return np.asarray(predictions)


def fit_joint(rows: list[dict[str, float]]) -> dict[str, float]:
    n = np.asarray([row["parameters"] for row in rows], dtype=float)
    c = np.asarray([row["compute_budget"] for row in rows], dtype=float)
    d = c / (6 * n)
    loss = np.asarray([row["final_loss"] for row in rows], dtype=float)

    best: tuple[float, float, float, np.ndarray] | None = None
    for alpha in np.linspace(0.05, 1.0, 80):
        for beta in np.linspace(0.05, 1.0, 80):
            design = np.column_stack([np.ones_like(n), n**-alpha, d**-beta])
            coefficients, *_ = np.linalg.lstsq(design, loss, rcond=None)
            prediction = design @ coefficients
            rmse = float(np.sqrt(np.mean((prediction - loss) ** 2)))
            if best is None or rmse < best[0]:
                best = (rmse, float(alpha), float(beta), coefficients)
    assert best is not None
    rmse, alpha, beta, coefficients = best
    floor, a_value, b_value = coefficients
    return {
        "floor": float(floor),
        "A": float(a_value),
        "alpha": float(alpha),
        "B": float(b_value),
        "beta": float(beta),
        "rmse": rmse,
    }


def save_figure(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_profiles(rows: list[dict[str, float]], figures: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    for compute in sorted({row["compute_budget"] for row in rows}):
        selected = sorted(
            (row for row in rows if row["compute_budget"] == compute),
            key=lambda row: row["parameters"],
        )
        axis.plot(
            [row["parameters"] for row in selected],
            [row["final_loss"] for row in selected],
            marker="o",
            label=f"{compute:.0e} FLOPs",
        )
    axis.set_xscale("log")
    axis.set(
        xlabel="Parameters N", ylabel="Final loss", title="Official IsoFLOP profiles"
    )
    axis.legend(ncol=2, fontsize=8)
    axis.grid(True, alpha=0.25)
    save_figure(figure, figures / "official_isoflop_profiles")


def plot_power(
    x: np.ndarray,
    y: np.ndarray,
    coefficient: float,
    exponent: float,
    bootstrap: np.ndarray,
    targets: np.ndarray,
    ylabel: str,
    output: Path,
) -> None:
    grid = np.logspace(np.log10(x.min()), 24, 300)
    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    axis.scatter(x, y, s=55, label="Discrete minima")
    axis.plot(grid, coefficient * grid**exponent, label=f"fit exponent={exponent:.3f}")
    target_predictions = coefficient * targets**exponent
    low, high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    axis.errorbar(
        targets,
        target_predictions,
        yerr=[target_predictions - low, high - target_predictions],
        fmt="s",
        label="extrapolation ±95% bootstrap",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(xlabel="Compute C (FLOPs)", ylabel=ylabel)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    save_figure(figure, output)


def write_manifest(results: Path) -> None:
    manifest = results / "manifest.sha256"
    lines = []
    for path in sorted(results.rglob("*")):
        if path.is_file() and path != manifest:
            lines.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(results).as_posix()}"
            )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/isoflops_curves.json"))
    parser.add_argument("--results", type=Path, default=Path("report/results"))
    args = parser.parse_args()
    rows = json.loads(args.data.read_text(encoding="utf-8"))
    optima = select_optima(rows)
    compute = np.asarray([row["compute_budget"] for row in optima])
    parameters = np.asarray([row["parameters"] for row in optima])
    tokens = np.asarray([row["tokens"] for row in optima])
    targets = np.asarray([1e23, 1e24])
    n_coefficient, n_exponent = fit_power(compute, parameters)
    d_coefficient, d_exponent = fit_power(compute, tokens)
    n_bootstrap = bootstrap_power(compute, parameters, targets)
    d_bootstrap = bootstrap_power(compute, tokens, targets)
    figures = args.results / "figures"
    plot_profiles(rows, figures)
    plot_power(
        compute,
        parameters,
        n_coefficient,
        n_exponent,
        n_bootstrap,
        targets,
        "Compute-optimal parameters",
        figures / "official_n_opt",
    )
    plot_power(
        compute,
        tokens,
        d_coefficient,
        d_exponent,
        d_bootstrap,
        targets,
        "Compute-optimal training tokens",
        figures / "official_d_opt",
    )

    n_prediction = n_coefficient * targets**n_exponent
    d_prediction = d_coefficient * targets**d_exponent
    fitted = n_coefficient * compute**n_exponent
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    axis.axhline(0, color="black", linewidth=1)
    axis.scatter(compute, np.log(parameters) - np.log(fitted))
    axis.set_xscale("log")
    axis.set(xlabel="Compute C", ylabel="Log residual", title="Power-law residuals")
    axis.grid(True, alpha=0.25)
    save_figure(figure, figures / "official_power_residuals")

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    left.plot(compute, [row["final_loss"] for row in optima], marker="o")
    left.set_xscale("log")
    left.set(xlabel="Compute", ylabel="Minimum loss", title="Best loss by compute tier")
    ratio = tokens / parameters
    right.plot(compute, ratio, marker="o")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set(xlabel="Compute", ylabel="Tokens / parameter", title="Optimal data ratio")
    save_figure(figure, figures / "official_compute_progress")

    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    tier_counts = [
        sum(row["compute_budget"] == value for row in rows) for value in compute
    ]
    axis.bar([f"{value:.0e}" for value in compute], tier_counts)
    axis.set(xlabel="Compute tier", ylabel="Runs", title="Official experiment coverage")
    axis.tick_params(axis="x", rotation=30)
    save_figure(figure, figures / "official_experiment_map")

    summary = {
        "source": "https://raw.githubusercontent.com/stanford-cs336/assignment3-scaling/main/data/isoflops_curves.json",
        "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "records": len(rows),
        "tiers": len(optima),
        "n_power": {"coefficient": n_coefficient, "exponent": n_exponent},
        "d_power": {"coefficient": d_coefficient, "exponent": d_exponent},
        "predictions": {
            f"{target:.0e}": {"parameters": float(n), "tokens": float(d)}
            for target, n, d in zip(targets, n_prediction, d_prediction, strict=True)
        },
        "joint_law": fit_joint(rows),
        "optima": optima,
    }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.results / "experiment_log.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(optima[0]))
        writer.writeheader()
        writer.writerows(optima)
    write_manifest(args.results)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
