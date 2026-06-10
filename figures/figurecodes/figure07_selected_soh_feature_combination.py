#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 7 as the SOH-estimation input configuration table."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Figure" / "figure7"
OUT_PNG = OUT_DIR / "figure7.png"
OUT_PDF = OUT_DIR / "figure7.pdf"


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    columns = ["SOH analysis", "Model", "Input feature", "Feature count"]
    rows = [
        ["Single-cell estimation", "Ridge", "feature_mean_ic", "1"],
        ["Cross-cell/domain estimation", "Ridge / MLP", "feature_mean_ic", "1"],
    ]

    fig, ax = plt.subplots(figsize=(5.65, 1.55))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.33, 0.20, 0.31, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1.0, 1.48)

    header_color = "#E8EEF3"
    row_colors = ["#FFFFFF", "#F6F8FA"]
    edge_color = "#1F1F1F"

    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.65)
        if row_idx == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", color="#111111")
        else:
            cell.set_facecolor(row_colors[(row_idx - 1) % 2])
            cell.set_text_props(color="#111111")

    ax.text(
        0.5,
        0.985,
        "SOH Estimation Uses a Single IC-Mean Feature",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        weight="bold",
        color="#111111",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.060, top=0.860)
    fig.savefig(OUT_PNG, dpi=600, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    plt.close(fig)

    print(f"[INFO] Fig. 7 -> {OUT_PNG}")


if __name__ == "__main__":
    main()
