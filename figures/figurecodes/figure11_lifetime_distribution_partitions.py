#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 11: lifetime distribution across data partitions."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


ROOT = Path(r"E:\Datasets\IVAS")
STAGE_DIR = (
    ROOT
    / "week_based"
    / "Final"
    / "EOL70"
    / "3step"
    / "outputs_400"
    / "BasicModel"
    / "stage3_final_rerun_400"
)
OUT_DIR = ROOT / "Figure" / "figure11"

CM = 1 / 2.54
FIG_SIZE = (6.0 * CM, 3.6 * CM)


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 0.75,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: object) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")


def load_partition(filename: str) -> np.ndarray:
    rows = read_csv(STAGE_DIR / filename)
    values = np.asarray(
        [as_float(r.get("lifetime_weeks_EOL70", r.get("lifetime_week"))) for r in rows],
        dtype=float,
    )
    return values[np.isfinite(values)]


def style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    ax.tick_params(axis="both", which="major", width=0.6, length=2.2, pad=1.5)
    ax.tick_params(axis="both", which="minor", width=0.5, length=1.4)


def draw() -> None:
    partitions = [
        ("Source\ntrain", "source_train_samples.csv", "#2f6fbb"),
        ("Target\nfine-tune", "target_finetune_samples.csv", "#d9802e"),
        ("Target\ntest", "target_test_samples.csv", "#2f9d6a"),
    ]
    data = [(label, load_partition(filename), color) for label, filename, color in partitions]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.arange(len(data), dtype=float)
    bar_width = 0.46
    rng = np.random.default_rng(2026)

    for i, (label, values, color) in enumerate(data):
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0

        ax.bar(i, mean, width=bar_width, color=color, alpha=0.52, edgecolor="none", zorder=1)
        ax.boxplot(
            [values],
            positions=[i],
            widths=bar_width * 0.54,
            patch_artist=True,
            showfliers=False,
            boxprops={
                "facecolor": "none",
                "edgecolor": "#333333",
                "linewidth": 0.72,
                "zorder": 5,
            },
            medianprops={"color": "#333333", "linewidth": 0.85, "zorder": 6},
            whiskerprops={"color": "#333333", "linewidth": 0.72, "zorder": 5},
            capprops={"color": "#333333", "linewidth": 0.72, "zorder": 5},
        )
        jitter_scale = 0.045 if values.size < 60 else 0.060
        jitter = rng.normal(0.0, jitter_scale, size=values.size)
        ax.scatter(
            np.full(values.size, i) + jitter,
            values,
            s=9 if values.size > 60 else 13,
            facecolor="white",
            edgecolor=color,
            linewidth=0.42,
            alpha=0.48 if values.size > 60 else 0.62,
            zorder=7,
        )
        ax.text(
            i,
            58.2,
            f"n={values.size}",
            ha="center",
            va="top",
            fontsize=5.6,
            color="#4d4d4d",
        )

    ax.set_xlim(-0.55, len(data) - 0.45)
    ax.set_ylim(0, 60)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _, _ in data])
    ax.set_xlabel("Domain")
    ax.set_ylabel("Lifetime (weeks)")
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.grid(axis="y", color="#e5e5e5", lw=0.50, zorder=0)
    style_axes(ax)

    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.32, top=0.95)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "figure11.pdf")
    fig.savefig(OUT_DIR / "figure11.png", dpi=600)
    plt.close(fig)

    for label, values, _ in data:
        print(
            f"{label.replace(chr(10), ' ')}: n={values.size}, "
            f"mean={np.mean(values):.3f}, std={np.std(values, ddof=1):.3f}"
        )
    print(f"Saved {OUT_DIR / 'figure11.png'}")
    print(f"Saved {OUT_DIR / 'figure11.pdf'}")


if __name__ == "__main__":
    draw()
