#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 29 in the PulseBat-combo plotting style."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from pulsebat_iv_common import (
    CM,
    COLORS,
    DPI,
    FIG_DIR,
    WEEK,
    read_csv,
    save_figure,
    setup_style,
    style_axes,
    to_float,
)


STAGE = WEEK / "Final" / "EOL70" / "3step" / "outputs_lowcapacity" / "lowcapacity_grid" / "stage3_final"
OUT_DIR = FIG_DIR / "figure29"
MODELS = (
    ("Benchmark", STAGE / "benchmark" / "test_overall_metrics.csv", COLORS["benchmark"]),
    ("Fine-tuned", STAGE / "transfer_model" / "test_overall_metrics.csv", COLORS["transfer"]),
)
METRICS = (
    ("MAE", "mae"),
    ("RMSE", "rmse"),
    ("MAPE (%)", "mape_percent"),
    ("R$^2$", "r2"),
)


def zero_trim(value: float, _pos: int | None = None) -> str:
    if abs(value) < 1e-12:
        return "0"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}"


def load_values() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, path, color in MODELS:
        data = read_csv(path)
        if data:
            rows.append({"model": label, "color": color, **{key: to_float(data[0].get(key)) for _, key in METRICS}})
    return rows


def main() -> None:
    setup_style()
    values = load_values()

    fig, axes = plt.subplots(1, len(METRICS), figsize=(13.25 * CM, 3.95 * CM), dpi=DPI)
    fig.subplots_adjust(left=0.070, right=0.760, bottom=0.225, top=0.880, wspace=0.40)

    for ax, (title, key) in zip(axes, METRICS):
        x = np.arange(len(values), dtype=float)
        y = np.asarray([float(row[key]) for row in values], dtype=float)
        colors = [str(row["color"]) for row in values]
        ax.bar(x, y, width=0.58, color=colors, edgecolor="white", linewidth=0.45, zorder=2)
        for xi, yi in zip(x, y):
            ax.text(xi, yi + (max(abs(y)) * 0.035 if max(abs(y)) else 0.05),
                    f"{yi:.2f}" if key != "r2" else f"{yi:.3f}",
                    ha="center", va="bottom", fontsize=5.8)

        ymin = min(0.0, float(np.nanmin(y)) * 1.15)
        ymax = max(0.1, float(np.nanmax(y)) * 1.18)
        if key == "r2":
            ymin = min(-0.2, float(np.nanmin(y)) * 1.20)
            ymax = max(0.2, float(np.nanmax(y)) * 1.18)
            ax.axhline(0.0, color=COLORS["mid_grey"], lw=0.70, ls=(0, (3.0, 2.0)), zorder=1)
        ax.set_ylim(ymin, ymax)
        ax.set_title(title, fontsize=7, pad=2.0)
        ax.set_xticks([])
        ax.yaxis.set_major_formatter(FuncFormatter(zero_trim))
        ax.grid(True, axis="y", color=COLORS["grid"], lw=0.35, zorder=1)
        style_axes(ax)

    axes[0].set_ylabel("Target-test metric", labelpad=2)
    handles = [
        plt.Line2D([], [], marker="s", linestyle="None", markersize=5.5,
                   markerfacecolor=str(row["color"]), markeredgecolor="none", label=str(row["model"]))
        for row in values
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.785, 0.55),
               frameon=False, handletextpad=0.45, labelspacing=0.30, borderaxespad=0)

    save_figure(fig, OUT_DIR, "figure29")
    plt.close(fig)
    print(f"[OK] Saved: {OUT_DIR / 'figure29.png'}")
    print(f"[OK] Saved: {OUT_DIR / 'figure29.pdf'}")


if __name__ == "__main__":
    main()
