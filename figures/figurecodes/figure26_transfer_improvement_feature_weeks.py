#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 26 in the PulseBat-combo plotting style.

This panel uses the Delta MAE% row from the week-sensitivity table:
w5 from the main rerun_400 run and w6-w10 from the legacy400 week protocol.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Figure" / "figure26"
MAIN_STAGE = (
    ROOT / "week_based" / "Final" / "EOL70" / "3step" / "outputs_400"
    / "BasicModel" / "stage3_final_rerun_400"
)
WEEK_STAGE = (
    ROOT / "week_based" / "Final" / "EOL70" / "3step" / "outputs_400"
    / "protocol_w6_10_from_stage3_final_rerun_400_legacy400"
)
WEEK_SUMMARY = (
    WEEK_STAGE
    / "protocol_w6_10_from_stage3_final_rerun_400_legacy400_benchmark_transfer_summary_all_splits.csv"
)

CM = 1 / 2.54
DPI = 600
FIG_W_CM = 8.72
FIG_H_CM = 4.35
LINE_COLOR = "#2F9D6A"
NEG_COLOR = "#B54D4D"


def read_csv(path: Path) -> List[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def metric(path: Path, key: str) -> float:
    return float(read_csv(path)[0][key])


def load_delta_mae_rows() -> List[Tuple[str, float]]:
    bm5 = metric(MAIN_STAGE / "benchmark" / "test_overall_metrics.csv", "mae")
    tl5 = metric(MAIN_STAGE / "transfer_model" / "test_overall_metrics.csv", "mae")
    rows: List[Tuple[str, float]] = [("w5", (bm5 - tl5) / bm5 * 100.0)]
    for row in read_csv(WEEK_SUMMARY):
        week = row["stage3_dir"].split("/")[0].replace("week", "w")
        rows.append((week, float(row["test_transfer_vs_bench_mae_improve_percent"])))
    rows.sort(key=lambda item: int(item[0].replace("w", "")))
    return rows


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.75,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.unicode_minus": False,
    })

def zero_trim(value: float, _pos: int | None = None) -> str:
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.0f}"


def plot() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    delta_rows = load_delta_mae_rows()
    labels = [row[0] for row in delta_rows]
    x = np.arange(len(labels), dtype=float)
    improvement = np.asarray([row[1] for row in delta_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(FIG_W_CM * CM, FIG_H_CM * CM), dpi=DPI)
    fig.subplots_adjust(left=0.155, right=0.965, bottom=0.205, top=0.950)

    ax.axhline(0.0, color="#B8B8B8", lw=0.75, ls=(0, (3.0, 2.0)), zorder=1)
    colors = [LINE_COLOR if val >= 0 else NEG_COLOR for val in improvement]
    ax.bar(x, improvement, width=0.58, color=colors, edgecolor="#202020",
           linewidth=0.45, zorder=3)
    for xpos, val in zip(x, improvement):
        va = "bottom" if val >= 0 else "top"
        offset = 0.55 if val >= 0 else -0.55
        ax.text(xpos, val + offset, f"{val:+.1f}", ha="center", va=va,
                fontsize=6.5, color="#202020")

    ymin = np.floor((float(np.min(improvement)) - 3.0) / 5.0) * 5.0
    ymax = np.ceil((float(np.max(improvement)) + 3.0) / 5.0) * 5.0
    ax.set_xlim(float(min(x)) - 0.55, float(max(x)) + 0.55)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(ymin, ymax + 0.1, 10.0))
    ax.yaxis.set_major_formatter(FuncFormatter(zero_trim))
    ax.set_xlabel("Feature week", labelpad=2)
    ax.set_ylabel("Delta MAE (%)", labelpad=2)
    ax.grid(True, axis="y", color="#E8E8E8", lw=0.35, zorder=0)
    ax.tick_params(length=2.0, width=0.6, pad=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)

    fig.savefig(OUT_DIR / "figure26.png", dpi=DPI)
    fig.savefig(OUT_DIR / "figure26.pdf")
    plt.close(fig)
    print(f"[OK] Saved: {OUT_DIR / 'figure26.png'}")
    print(f"[OK] Saved: {OUT_DIR / 'figure26.pdf'}")


def main() -> None:
    plot()


if __name__ == "__main__":
    main()
