#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator


ROOT = Path(r"E:\Datasets\IVAS")
IN_CSV = (
    ROOT
    / "week_based"
    / "SOHest"
    / "results"
    / "ridge_single_cell"
    / "seed_sweep_by_cell"
    / "_cond_explain"
    / "group_summary_with_conditions.csv"
)
OUT_DIR = ROOT / "PPT_Figure" / "P10_figure"
OUT_STEM = "P10_depth_of_discharge_mae_boxplot"

FIG_W_CM = 8.4
FIG_H_CM = 4.7
DPI = 600


def cm_to_in(value: float) -> float:
    return value / 2.54


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.75,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.unicode_minus": False,
        }
    )


def interval_label(interval) -> str:
    return f"{interval.left:.1f}-{interval.right:.1f}"


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV)
    df = df.dropna(subset=["dod_pct", "group_median_cell_mae"]).copy()
    df["depth_of_discharge_bin"] = pd.qcut(df["dod_pct"], q=4, duplicates="drop")
    df["depth_of_discharge_label"] = df["depth_of_discharge_bin"].apply(interval_label)
    df["state_of_health_mae_percent"] = df["group_median_cell_mae"] * 100.0

    bins = sorted(df["depth_of_discharge_bin"].dropna().unique(), key=lambda item: item.left)
    labels = [interval_label(item) for item in bins]
    data = [
        df.loc[df["depth_of_discharge_bin"] == item, "state_of_health_mae_percent"].to_numpy()
        for item in bins
    ]

    fig, ax = plt.subplots(figsize=(cm_to_in(FIG_W_CM), cm_to_in(FIG_H_CM)))
    fig.subplots_adjust(left=0.205, right=0.985, bottom=0.285, top=0.965)

    box = ax.boxplot(
        data,
        widths=0.52,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#C43B3B", "linewidth": 0.85},
        boxprops={"edgecolor": "#0A2F42", "linewidth": 0.75},
        whiskerprops={"color": "#0A2F42", "linewidth": 0.75},
        capprops={"color": "#0A2F42", "linewidth": 0.75},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "#0A2F42",
            "markeredgewidth": 0.65,
            "markersize": 3.0,
        },
    )

    fill_colors = ["#C9D7EE", "#A9C8D8", "#8BB8C2", "#6D9FA9"]
    for patch, color in zip(box["boxes"], fill_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)

    ax.set_xlabel("Depth of discharge quantile range (%)")
    ax.set_ylabel("Median Absolute Error")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")

    ax.set_ylim(0.5, 4.6)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.55)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#0A2F42")
    ax.tick_params(axis="both", which="major", length=2.4, width=0.65, pad=1.8)
    ax.tick_params(axis="both", which="minor", length=1.6, width=0.55)

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{suffix}", dpi=DPI if suffix == "png" else None, facecolor="white")

    out_csv = OUT_DIR / f"{OUT_STEM}_source_values.csv"
    df[
        [
            "cond_group",
            "group_num",
            "dod_pct",
            "depth_of_discharge_label",
            "group_median_cell_mae",
            "state_of_health_mae_percent",
        ]
    ].to_csv(out_csv, index=False, encoding="utf-8-sig")
    plt.close(fig)


if __name__ == "__main__":
    main()
