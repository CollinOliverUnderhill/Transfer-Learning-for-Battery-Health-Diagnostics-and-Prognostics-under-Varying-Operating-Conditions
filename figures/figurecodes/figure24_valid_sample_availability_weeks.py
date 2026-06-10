#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 24 in the PulseBat-combo plotting style."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
WEEK = ROOT / "week_based"
OUT_DIR = ROOT / "Figure" / "figure24"
SOURCE_CSV = WEEK / "Final" / "EOL70" / "features" / "week_availability_summary_EOL70.csv"
FALLBACK_CSV = WEEK / "Final" / "EOL70" / "features" / "feature_table_all_cells_multiweek_EOL70.csv"

CM = 1 / 2.54
DPI = 600
FIG_W_CM = 8.72
FIG_H_CM = 4.20
BAR_COLOR = "#6BB7B2"
BAR_EDGE = "#111111"


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


def read_csv(path: Path) -> List[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_availability() -> Tuple[List[str], np.ndarray]:
    rows = read_csv(SOURCE_CSV)
    labels: List[str] = []
    values: List[float] = []
    for row in rows:
        week = row.get("week", row.get("feature_week", "")).strip()
        value_text = (
            row.get("status_ok_cells")
            or row.get("valid_cells")
            or row.get("usable_non_nan_cells")
            or row.get("n_cells")
            or row.get("sample_count")
            or row.get("valid_cell_count")
        )
        if not week or value_text in {None, ""}:
            continue
        if week.replace("w", "") == "15":
            continue
        labels.append(f"w{week.replace('w', '')}")
        values.append(float(value_text))

    if labels:
        return labels, np.asarray(values, dtype=float)

    fallback = read_csv(FALLBACK_CSV)
    weeks = [3, 5, 6, 7, 8, 9, 10]
    labels = [f"w{week}" for week in weeks]
    values = [
        sum(1 for row in fallback if str(row.get(f"feature_status_w{week}", "")).lower() == "ok")
        for week in weeks
    ]
    return labels, np.asarray(values, dtype=float)


def plot() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels, values = load_availability()
    x = np.arange(len(labels), dtype=float)

    fig, ax = plt.subplots(figsize=(FIG_W_CM * CM, FIG_H_CM * CM), dpi=DPI)
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.210, top=0.920)

    ax.bar(x, values, width=0.62, color=BAR_COLOR, edgecolor=BAR_EDGE, linewidth=0.45, zorder=2)
    for xi, yi in zip(x, values):
        ax.text(xi, yi + max(values) * 0.020, f"{yi:.0f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Feature week", labelpad=2)
    ax.set_ylabel("Valid cells", labelpad=2)
    ax.set_ylim(0, max(values) * 1.14)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(True, axis="y", color="#E8E8E8", lw=0.35, zorder=1)
    ax.tick_params(length=2.0, width=0.6, pad=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)

    fig.savefig(OUT_DIR / "figure24.png", dpi=DPI)
    fig.savefig(OUT_DIR / "figure24.pdf")
    plt.close(fig)
    print(f"[OK] Saved: {OUT_DIR / 'figure24.png'}")
    print(f"[OK] Saved: {OUT_DIR / 'figure24.pdf'}")


def main() -> None:
    plot()


if __name__ == "__main__":
    main()
