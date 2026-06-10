#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 31 in the PulseBat-combo plotting style."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pulsebat_iv_common import (
    CM,
    COLORS,
    DPI,
    FIG_DIR,
    WEEK,
    finite,
    read_csv,
    save_figure,
    setup_style,
    style_axes,
    to_float,
)


STAGE = WEEK / "Final" / "EOL70" / "3step" / "outputs_lowcapacity" / "lowcapacity_grid" / "stage3_final"
OUT_DIR = FIG_DIR / "figure31"
SERIES = (
    ("Benchmark", STAGE / "benchmark" / "test_group_metrics.csv", COLORS["benchmark"]),
    ("Source-only", STAGE / "transfer_model" / "test_group_metrics_source_only.csv", COLORS["source_only"]),
    ("Fine-tuned", STAGE / "transfer_model" / "test_group_metrics.csv", COLORS["transfer"]),
)


def load_group_rows() -> tuple[list[int], list[tuple[str, dict[int, float], str]]]:
    loaded: list[tuple[str, dict[int, float], str]] = []
    groups: set[int] = set()
    for label, path, color in SERIES:
        rows = read_csv(path)
        if not rows:
            continue
        values: dict[int, float] = {}
        for row in rows:
            group = to_float(row.get("group_num"))
            value = to_float(row.get("cell_mae_mean", row.get("mae")))
            if finite(group) and finite(value):
                group_num = int(group)
                values[group_num] = value
                groups.add(group_num)
        loaded.append((label, values, color))
    return sorted(groups), loaded


def main() -> None:
    setup_style()
    groups, loaded = load_group_rows()
    x = np.arange(len(groups), dtype=float)
    width = min(0.22, 0.70 / max(1, len(loaded)))
    offsets = np.linspace(-width, width, len(loaded)) if len(loaded) > 1 else np.asarray([0.0])

    fig, ax = plt.subplots(figsize=(13.25 * CM, 4.45 * CM), dpi=DPI)
    fig.subplots_adjust(left=0.085, right=0.805, bottom=0.220, top=0.950)

    ymax = 0.0
    for offset, (label, values, color) in zip(offsets, loaded):
        y = np.asarray([values.get(group, np.nan) for group in groups], dtype=float)
        ymax = max(ymax, float(np.nanmax(y)))
        ax.bar(x + offset, y, width=width * 0.92, color=color, edgecolor="white",
               linewidth=0.35, label=label, zorder=2)

    ax.set_xlabel("Test group", labelpad=2)
    ax.set_ylabel("Group MAE (weeks)", labelpad=2)
    ax.set_xlim(-0.65, len(groups) - 0.35)
    ax.set_ylim(0.0, ymax * 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{group}" for group in groups], rotation=35, ha="right", fontsize=6.2)
    ax.grid(True, axis="y", color=COLORS["grid"], lw=0.35, zorder=1)
    ax.legend(loc="center left", bbox_to_anchor=(1.025, 0.55), frameon=False,
              handlelength=1.0, handletextpad=0.45, labelspacing=0.30, borderaxespad=0)
    style_axes(ax)

    save_figure(fig, OUT_DIR, "figure31")
    plt.close(fig)
    print(f"[OK] Saved: {OUT_DIR / 'figure31.png'}")
    print(f"[OK] Saved: {OUT_DIR / 'figure31.pdf'}")


if __name__ == "__main__":
    main()
