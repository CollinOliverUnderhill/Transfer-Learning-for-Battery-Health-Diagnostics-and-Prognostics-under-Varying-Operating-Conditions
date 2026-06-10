#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


ROOT = Path(r"E:\Datasets\IVAS")
OUT_DIR = ROOT / "Figure" / "Figure_later" / "42"
BASE = ROOT / "week_based" / "SOHest" / "results"

EXPERIMENTS = [
    ("Source-only", "scheme2_37train_12test"),
    ("Fine-tuned", "scheme2_37train_8ft_12test"),
]

MODEL_DIRS = {
    "Ridge": BASE / "ridge_cross_cell" / "domain_shift",
    "MLP": BASE / "mlp_cross_cell",
}

MODEL_COLORS = {
    "Ridge": "#1f4e79",
    "MLP": "#d96c9f",
}

CM = 1 / 2.54
FIGSIZE = (12.74 * CM, 4.00 * CM)


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 0.75,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.labelsize": 8,
        "legend.fontsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    ax.tick_params(axis="both", which="major", width=0.6, length=2.2, pad=1.5)


def load_experiment(model: str, exp_dir_name: str) -> dict[str, object]:
    path = MODEL_DIRS[model] / exp_dir_name
    overall = read_csv(path / "test_overall_metrics.csv")[0]
    cell_rows = read_csv(path / "test_cell_metrics.csv")
    cell_mae = np.asarray([float(r["mae"]) for r in cell_rows], dtype=float)
    return {
        "path": path,
        "overall": overall,
        "cell_mae": cell_mae,
        "mean": float(np.mean(cell_mae)),
        "std": float(np.std(cell_mae, ddof=1)),
        "median": float(np.median(cell_mae)),
        "min": float(np.min(cell_mae)),
        "max": float(np.max(cell_mae)),
    }


def plot(models_data: dict[str, list[dict[str, object]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    rng = np.random.default_rng(2026)

    for ax, model, panel in zip(axes, ["Ridge", "MLP"], ["a", "b"]):
        color = MODEL_COLORS[model]
        data = models_data[model]
        xs = np.arange(len(EXPERIMENTS), dtype=float)
        means = np.asarray([d["mean"] for d in data], dtype=float)
        stds = np.asarray([d["std"] for d in data], dtype=float)
        overall_mae = np.asarray([float(d["overall"]["mae"]) for d in data], dtype=float)  # type: ignore[index]

        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.12, linewidth=0, zorder=1)
        ax.plot(xs, means, color=color, lw=0.9, zorder=4, label="Cell mean")
        ax.scatter(xs, means, s=22, facecolor=color, edgecolor="white", linewidth=0.4, zorder=5)
        ax.plot(xs, overall_mae, color="#111111", lw=0.9, linestyle="--", zorder=5, label="Overall")

        for i, d in enumerate(data):
            cell_mae = d["cell_mae"]  # type: ignore[assignment]
            jitter = rng.normal(0.0, 0.035, size=len(cell_mae))
            ax.scatter(
                np.full(len(cell_mae), i) + jitter,
                cell_mae,
                s=13,
                facecolor="white",
                edgecolor=color,
                linewidth=0.45,
                alpha=0.36,
                zorder=3,
            )

        ax.set_xlim(-0.32, len(EXPERIMENTS) - 0.68)
        ax.set_xticks(xs)
        ax.set_xticklabels([label for label, _ in EXPERIMENTS], rotation=16, ha="right")
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.grid(axis="y", color="#e5e5e5", lw=0.5, zorder=0)
        ax.text(0.04, 0.94, panel, transform=ax.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")
        ax.text(0.96, 0.08, model, transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
        style_axes(ax)
        if ax is axes[0]:
            ax.set_ylabel("MAE (SOH)")
        else:
            ax.tick_params(axis="y", labelleft=False)

    fig.text(0.52, 0.055, "Experiment group", ha="center", va="center", fontsize=8)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.30, top=0.97, wspace=0.045)
    fig.savefig(OUT_DIR / "cross_cell_experiment_groups_mae_ridge_mlp.pdf")
    fig.savefig(OUT_DIR / "cross_cell_experiment_groups_mae_ridge_mlp.png", dpi=600)
    plt.close(fig)


def write_summary(models_data: dict[str, list[dict[str, object]]]) -> None:
    with (OUT_DIR / "cross_cell_experiment_groups_mae_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "experiment_group",
                "n_rows",
                "n_cells",
                "n_groups",
                "overall_mae",
                "overall_rmse",
                "overall_r2",
                "overall_mape_percent",
                "cell_mae_mean",
                "cell_mae_std",
                "cell_mae_median",
                "cell_mae_min",
                "cell_mae_max",
            ]
        )
        for model, data in models_data.items():
            for (label, _), d in zip(EXPERIMENTS, data):
                overall = d["overall"]  # type: ignore[assignment]
                writer.writerow(
                    [
                        model,
                        label,
                        overall["n_rows"],
                        overall["n_cells"],
                        overall["n_groups"],
                        overall["mae"],
                        overall["rmse"],
                        overall["r2"],
                        overall["mape_percent"],
                        d["mean"],
                        d["std"],
                        d["median"],
                        d["min"],
                        d["max"],
                    ]
                )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models_data: dict[str, list[dict[str, object]]] = {}
    for model in ["Ridge", "MLP"]:
        models_data[model] = [load_experiment(model, dirname) for _, dirname in EXPERIMENTS]
    plot(models_data)
    write_summary(models_data)
    for model, data in models_data.items():
        for (label, _), d in zip(EXPERIMENTS, data):
            overall = d["overall"]  # type: ignore[assignment]
            print(
                f"{model} {label}: n_rows={overall['n_rows']}, n_cells={overall['n_cells']}, "
                f"overall_MAE={float(overall['mae']):.6f}, cell_mean={float(d['mean']):.6f}, "
                f"cell_std={float(d['std']):.6f}"
            )


if __name__ == "__main__":
    main()
