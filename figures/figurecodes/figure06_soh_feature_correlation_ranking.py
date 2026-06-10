#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 6 as a one-panel ranked bar chart."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "Data" / "Processing_Data" / "SOHest" / "hetero10_feature_soh_correlation_summary.csv"
OUT_DIR = ROOT / "Figure" / "figure6"
OUT_PNG = OUT_DIR / "figure6.png"
OUT_PDF = OUT_DIR / "figure6.pdf"


def read_rows() -> list[dict[str, str]]:
    with DATA_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def feature_short(name: str) -> str:
    return name.split("_", 1)[0]


def main() -> None:
    rows = read_rows()
    ranked = sorted(rows, key=lambda row: abs(float(row["r"])), reverse=True)

    labels = [feature_short(row["feature"]) for row in ranked]
    signed_r = [float(row["r"]) for row in ranked]
    values = [abs(v) for v in signed_r]

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(4.9, 2.05))
    x = range(len(labels))
    bars = ax.bar(
        x,
        values,
        width=0.62,
        color="#69AEEB",
        edgecolor="#1E88F5",
        linewidth=1.0,
        alpha=0.86,
    )

    for bar, value in zip(bars, signed_r):
        label = f"{value:.2f}"
        y = min(abs(value) + 0.025, 1.12)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="left",
            va="bottom",
            rotation=35,
            rotation_mode="anchor",
            fontsize=9.0,
            color="#111111",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylim(0, 1.30)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel("Features", fontsize=16, labelpad=6)
    ax.set_ylabel("Correlation", fontsize=16, labelpad=8)
    ax.tick_params(axis="y", labelsize=13, width=1.7, length=5, pad=5)
    ax.tick_params(axis="x", width=1.7, length=5, pad=4)

    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
        spine.set_color("black")

    ax.grid(False)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.330, top=0.965)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    plt.close(fig)

    print(f"[INFO] Fig. 6 -> {OUT_PNG}")
    print("       Order: " + ", ".join(f"{label} ({r:.3f})" for label, r in zip(labels, signed_r)))


if __name__ == "__main__":
    main()
