#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 8 as a compact week-5 feature-correlation heatmap."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "week_based" / "Final" / "EOL70" / "features" / "correlation_matrix_w5_EOL70.csv"
OUT_DIR = ROOT / "Figure" / "figure8"
OUT_PNG = OUT_DIR / "figure8.png"
OUT_PDF = OUT_DIR / "figure8.pdf"

FIG_W_CM = 7.4
FIG_H_CM = 6.6


def read_matrix() -> tuple[list[str], list[list[float]]]:
    with DATA_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    # Include both f*_w5 features AND lifetime_week
    features = [key for key in rows[0].keys() if key.startswith("f") and key.endswith("_w5")]
    features.append("lifetime_week")
    matrix: list[list[float]] = []
    for feature in features:
        row = next(item for item in rows if item["feature"] == feature)
        matrix.append([float(row[col]) for col in features])
    labels = [feature.replace("_w5", "").replace("lifetime_week", "lifetime") for feature in features]
    return labels, matrix


def main() -> None:
    labels, matrix = read_matrix()

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    cmap = LinearSegmentedColormap.from_list(
        "ivas_corr",
        ["#2F65B0", "#F4F1E8", "#B84A4A"],
        N=256,
    )

    fig = plt.figure(figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54))
    ax = fig.add_axes([0.165, 0.205, 0.640, 0.730])
    cax = fig.add_axes([0.855, 0.205, 0.040, 0.730])

    im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest", aspect="equal")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        va="top",
        rotation_mode="anchor",
        fontsize=6.6,
    )
    ax.set_yticklabels(labels, fontsize=6.6)
    ax.set_xlabel("Features", fontsize=7.2, labelpad=3)
    ax.set_ylabel("Features", fontsize=7.2, labelpad=3)
    ax.tick_params(axis="both", length=0, pad=1.5)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    # Thin white cell separators, with an explicit black outer frame.
    ax.set_xticks([i - 0.5 for i in range(1, len(labels))], minor=True)
    ax.set_yticks([i - 0.5 for i in range(1, len(labels))], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.add_patch(Rectangle((-0.5, -0.5), len(labels), len(labels), fill=False, ec="black", lw=1.0))

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            text_color = "white" if abs(value) >= 0.55 else "black"
            label_value = 0.0 if abs(value) < 0.05 else value
            ax.text(j, i, f"{label_value:.1f}", ha="center", va="center", fontsize=4.8, color=text_color)

    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["-1", "0", "+1"])
    cbar.ax.tick_params(labelsize=6.6, width=0.8, length=2.5, pad=2)
    cbar.outline.set_linewidth(1.0)
    cbar.outline.set_edgecolor("black")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    plt.close(fig)

    print(f"[INFO] Fig. 8 -> {OUT_PNG}")


if __name__ == "__main__":
    main()
