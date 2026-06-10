#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 30 in the PulseBat-combo plotting style."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

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
OUT_DIR = FIG_DIR / "figure30"
DATASETS = (
    ("Benchmark", STAGE / "benchmark" / "predictions_test.csv", STAGE / "benchmark" / "test_overall_metrics.csv", COLORS["benchmark"]),
    ("Fine-tuned", STAGE / "transfer_model" / "predictions_test.csv", STAGE / "transfer_model" / "test_overall_metrics.csv", COLORS["transfer"]),
)


def load_prediction(path):
    rows = read_csv(path)
    y_true = np.asarray([to_float(row.get("y_true")) for row in rows], dtype=float)
    y_pred = np.asarray([to_float(row.get("y_pred")) for row in rows], dtype=float)
    return y_true, y_pred


def draw_panel(ax, label: str, panel: str, pred_path, metric_path, color: str, lo: float, hi: float, show_ylabel: bool) -> None:
    y_true, y_pred = load_prediction(pred_path)
    metrics = read_csv(metric_path)[0]
    ax.plot([lo, hi], [lo, hi], color="#404040", lw=0.75, zorder=1)
    ax.scatter(y_true, y_pred, s=9.0, facecolor=color, edgecolor="white",
               linewidth=0.25, alpha=0.84, rasterized=True, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ticks = np.arange(lo, hi + 1e-9, 15.0)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Predicted RUL (weeks)")
    else:
        ax.tick_params(labelleft=False)
        ax.set_xticklabels([""] + [f"{int(t)}" for t in ticks[1:]])
    ax.text(0.05, 0.95, panel, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, fontweight="bold")
    ax.text(0.14, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.text(0.05, 0.82,
            f"MAE={to_float(metrics.get('mae')):.2f}\n"
            f"RMSE={to_float(metrics.get('rmse')):.2f}\n"
            f"MAPE={to_float(metrics.get('mape_percent')):.2f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=6, linespacing=1.15)
    style_axes(ax)


def main() -> None:
    setup_style()
    all_vals = []
    for _, pred_path, _, _ in DATASETS:
        y_true, y_pred = load_prediction(pred_path)
        all_vals.extend(y_true.tolist())
        all_vals.extend(y_pred.tolist())
    lo = 0.0
    hi = float(np.ceil(max(all_vals) * 1.08 / 15.0) * 15.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.74 * CM, 4.00 * CM), dpi=DPI)
    for idx, (ax, (label, pred_path, metric_path, color), panel) in enumerate(zip(axes, DATASETS, ("a", "b"))):
        draw_panel(ax, label, panel, pred_path, metric_path, color, lo, hi, show_ylabel=idx == 0)

    fig_w_cm, fig_h_cm = 12.74, 4.00
    ax_h = 0.76
    ax_w = ax_h * (fig_h_cm / fig_w_cm)
    ax_bottom = 0.20
    ax_left = 0.19
    ax_gap = 0.012
    axes[0].set_position([ax_left, ax_bottom, ax_w, ax_h])
    axes[1].set_position([ax_left + ax_w + ax_gap, ax_bottom, ax_w, ax_h])
    fig.text(ax_left + ax_w + ax_gap / 2, 0.055, "True RUL (weeks)",
             ha="center", va="center", fontsize=8)

    save_figure(fig, OUT_DIR, "figure30")
    plt.close(fig)
    print(f"[OK] Saved: {OUT_DIR / 'figure30.png'}")
    print(f"[OK] Saved: {OUT_DIR / 'figure30.pdf'}")


if __name__ == "__main__":
    main()
