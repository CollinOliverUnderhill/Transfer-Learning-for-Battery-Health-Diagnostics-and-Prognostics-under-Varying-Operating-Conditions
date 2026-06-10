#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 25 in the PulseBat-combo plotting style."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Figure" / "figure25"
MAIN_STAGE = (
    ROOT / "week_based" / "Final" / "EOL70" / "3step" / "outputs_400"
    / "BasicModel" / "stage3_final_rerun_400"
)
WEEK_STAGE = (
    ROOT / "week_based" / "Final" / "EOL70" / "3step" / "outputs_400"
    / "protocol_w6_10_from_stage3_final_rerun_400_legacy400"
)

CM = 1 / 2.54
DPI = 600
FIG_W_CM = 8.72
FIG_H_CM = 4.60

SERIES: Sequence[Tuple[str, str, str, str]] = (
    ("Benchmark", "benchmark_test_mae", "#7F7F7F", "o"),
    ("Source-only", "source_only_test_mae", "#B54D4D", "s"),
    ("Fine-tuned", "transfer_test_mae", "#2F9D6A", "D"),
)


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


def metric(path: Path, key: str) -> float:
    return float(read_csv(path)[0][key])


def load_rows() -> Tuple[np.ndarray, List[dict[str, str]]]:
    rows: List[dict[str, str]] = []
    rows.append({
        "week": "w5",
        "benchmark_test_mae": str(metric(MAIN_STAGE / "benchmark" / "test_overall_metrics.csv", "mae")),
        "source_only_test_mae": str(metric(MAIN_STAGE / "transfer_model" / "test_overall_metrics_source_only.csv", "mae")),
        "transfer_test_mae": str(metric(MAIN_STAGE / "transfer_model" / "test_overall_metrics.csv", "mae")),
    })
    for week in range(6, 11):
        stage = WEEK_STAGE / f"week{week}" / "stage3_final"
        rows.append({
            "week": f"w{week}",
            "benchmark_test_mae": str(metric(stage / "benchmark" / "test_overall_metrics.csv", "mae")),
            "source_only_test_mae": str(metric(stage / "transfer_model" / "test_overall_metrics_source_only.csv", "mae")),
            "transfer_test_mae": str(metric(stage / "transfer_model" / "test_overall_metrics.csv", "mae")),
        })
    weeks = np.asarray([float(row["week"].replace("w", "")) for row in rows], dtype=float)
    return weeks, rows


def zero_trim(value: float, _pos: int | None = None) -> str:
    if abs(value) < 1e-12:
        return "0"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}"


def plot() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weeks, rows = load_rows()

    fig, ax = plt.subplots(figsize=(FIG_W_CM * CM, FIG_H_CM * CM), dpi=DPI)
    fig.subplots_adjust(left=0.135, right=0.735, bottom=0.190, top=0.950)

    all_values: List[float] = []
    for label, key, color, marker in SERIES:
        y = np.asarray([float(row[key]) for row in rows], dtype=float)
        all_values.extend(y.tolist())
        ax.plot(weeks, y, color=color, lw=0.95, marker=marker, markersize=4.2,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.45,
                label=label, zorder=3)

    ymin = np.floor((min(all_values) - 0.15) * 2.0) / 2.0
    ymax = np.ceil((max(all_values) + 0.15) * 2.0) / 2.0
    ax.set_xlim(float(min(weeks)) - 0.25, float(max(weeks)) + 0.25)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(weeks)
    ax.set_xticklabels([f"w{int(w)}" for w in weeks])
    ax.set_yticks([ymin, 0.5 * (ymin + ymax), ymax])
    ax.yaxis.set_major_formatter(FuncFormatter(zero_trim))
    ax.set_xlabel("Feature week", labelpad=2)
    ax.set_ylabel("Target-test MAE (weeks)", labelpad=2)
    ax.grid(True, axis="y", color="#E8E8E8", lw=0.35, zorder=1)
    ax.tick_params(length=2.0, width=0.6, pad=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)

    ax.legend(loc="center left", bbox_to_anchor=(1.035, 0.55), frameon=False,
              handlelength=1.2, handletextpad=0.45, labelspacing=0.35, borderaxespad=0)

    fig.savefig(OUT_DIR / "figure25.png", dpi=DPI)
    fig.savefig(OUT_DIR / "figure25.pdf")
    plt.close(fig)
    print(f"[OK] Saved: {OUT_DIR / 'figure25.png'}")
    print(f"[OK] Saved: {OUT_DIR / 'figure25.pdf'}")


def main() -> None:
    plot()


if __name__ == "__main__":
    main()
