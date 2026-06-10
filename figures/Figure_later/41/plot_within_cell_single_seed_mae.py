#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


ROOT = Path(r"E:\Datasets\IVAS")
OUT_DIR = ROOT / "Figure" / "Figure_later" / "41"
METRICS_CSV = (
    ROOT
    / "week_based"
    / "SOHest"
    / "results"
    / "ridge_single_cell"
    / "seed_sweep_by_cell"
    / "_seed_summary"
    / "all_seed_metrics.csv"
)

SEED = 2
CM = 1 / 2.54
FIGSIZE = (6.37 * CM, 4.00 * CM)


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 0.75,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.labelsize": 8,
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_csv(METRICS_CSV) if int(float(r["seed"])) == SEED]
    if not rows:
        raise RuntimeError(f"No rows found for seed={SEED}.")

    values = np.asarray([float(r["test_mae"]) for r in rows], dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    color = "#1f4e79"
    x0 = 0
    width = 0.46
    rng = np.random.default_rng(2026)

    ax.bar(x0, mean, width=width, color=color, alpha=0.55, edgecolor="none", zorder=1)
    ax.errorbar(x0, mean, yerr=std, color="#4d4d4d", lw=0.75, capsize=3.0, capthick=0.75, zorder=4)
    ax.boxplot(
        [values],
        positions=[x0],
        widths=width * 0.52,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "none", "edgecolor": "#333333", "linewidth": 0.75, "zorder": 5},
        medianprops={"color": "#333333", "linewidth": 0.9, "zorder": 6},
        whiskerprops={"color": "#333333", "linewidth": 0.75, "zorder": 5},
        capprops={"color": "#333333", "linewidth": 0.75, "zorder": 5},
    )
    jitter = rng.normal(0.0, 0.042, size=len(values))
    ax.scatter(
        np.full(len(values), x0) + jitter,
        values,
        s=18,
        facecolor="white",
        edgecolor=color,
        linewidth=0.55,
        alpha=0.52,
        zorder=7,
    )
    ax.plot([x0 - width * 0.40, x0 + width * 0.40], [mean, mean], color="#111111", lw=1.05, zorder=8)

    ax.set_xlim(-0.55, 0.55)
    ax.set_xticks([x0])
    ax.set_xticklabels([f"Seed {SEED}"])
    ax.set_xlabel("Random seed")
    ax.set_ylabel("MAE (SOH)")
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(axis="y", color="#e5e5e5", lw=0.5, zorder=0)
    style_axes(ax)
    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.28, top=0.97)

    fig.savefig(OUT_DIR / "within_cell_single_seed_mae_distribution.pdf")
    fig.savefig(OUT_DIR / "within_cell_single_seed_mae_distribution.png", dpi=600)
    plt.close(fig)

    out_csv = OUT_DIR / "within_cell_single_seed_mae_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "n_cells", "mae_min", "mae_max", "mae_mean", "mae_std", "mae_median"])
        writer.writerow([SEED, len(values), np.min(values), np.max(values), mean, std, np.median(values)])

    print(
        f"seed={SEED}, n_cells={len(values)}, min={np.min(values):.6f}, "
        f"max={np.max(values):.6f}, mean={mean:.6f}, std={std:.6f}, median={np.median(values):.6f}"
    )


if __name__ == "__main__":
    main()
