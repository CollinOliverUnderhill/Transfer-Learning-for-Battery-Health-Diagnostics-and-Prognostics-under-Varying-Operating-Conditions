#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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
OUT_STEM = "P10_depth_of_discharge_lifetime_eol70"

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
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.unicode_minus": False,
        }
    )


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV)
    df = df.dropna(subset=["dod_pct", "lifetime_wk"]).copy()
    df = df[np.isfinite(df["dod_pct"]) & np.isfinite(df["lifetime_wk"])].copy()
    df = df.sort_values("dod_pct")

    x = df["dod_pct"].to_numpy(dtype=float)
    y = df["lifetime_wk"].to_numpy(dtype=float)
    pearson_r = pearson(x, y)
    spearman_r = spearman(x, y)

    slope, intercept = np.polyfit(x, y, deg=1)
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = slope * x_fit + intercept

    fig, ax = plt.subplots(figsize=(cm_to_in(FIG_W_CM), cm_to_in(FIG_H_CM)))
    fig.subplots_adjust(left=0.245, right=0.985, bottom=0.23, top=0.965)

    ax.scatter(
        x,
        y,
        s=20,
        color="#7EAEB7",
        edgecolor="#0A2F42",
        linewidth=0.45,
        alpha=0.92,
        zorder=3,
    )
    ax.plot(x_fit, y_fit, color="#C43B3B", linewidth=1.05, zorder=4)

    ax.text(
        0.98,
        0.94,
        f"Pearson r = {pearson_r:.2f}\nSpearman rho = {spearman_r:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#333333",
    )

    ax.set_xlabel("Depth of discharge (%)")
    ax.set_ylabel("Lifetime")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, max(55, np.ceil(y.max() / 10.0) * 10.0))
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(axis="both", color="#E6E6E6", linewidth=0.55)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#0A2F42")
    ax.tick_params(axis="both", which="major", length=2.4, width=0.65, pad=1.8)
    ax.tick_params(axis="both", which="minor", length=1.6, width=0.55)

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{suffix}", dpi=DPI if suffix == "png" else None, facecolor="white")

    out_csv = OUT_DIR / f"{OUT_STEM}_source_values.csv"
    df[["cond_group", "group_num", "dod_pct", "lifetime_wk"]].to_csv(
        out_csv, index=False, encoding="utf-8-sig"
    )
    (OUT_DIR / f"{OUT_STEM}_correlation.txt").write_text(
        f"Pearson r = {pearson_r:.6f}\n"
        f"Spearman rho = {spearman_r:.6f}\n"
        f"n = {len(df)}\n",
        encoding="utf-8",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
