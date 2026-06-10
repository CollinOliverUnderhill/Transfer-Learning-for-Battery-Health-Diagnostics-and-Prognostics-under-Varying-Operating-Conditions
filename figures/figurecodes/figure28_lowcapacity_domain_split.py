#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 28 in the PulseBat-combo plotting style."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pulsebat_iv_common import (
    CM,
    COLORS,
    DPI,
    FIG_DIR,
    ROOT,
    WEEK,
    finite,
    group_num_from_cell,
    read_csv,
    save_figure,
    setup_style,
    style_axes,
    to_float,
)


STAGE = WEEK / "Final" / "EOL70" / "3step" / "outputs_lowcapacity" / "lowcapacity_grid" / "stage3_final"
OUT_DIR = FIG_DIR / "figure28"
SPLITS = (
    ("Source train", "source_train_samples.csv", COLORS["source"]),
    ("Target fine-tune", "target_finetune_samples.csv", COLORS["fine_tune"]),
    ("Target test", "target_test_samples.csv", COLORS["test"]),
)


def split_map() -> dict[int, str]:
    out: dict[int, str] = {}
    for label, filename, _ in SPLITS:
        for row in read_csv(STAGE / filename):
            group = to_float(row.get("group_num"))
            group_num = int(group) if finite(group) else group_num_from_cell(row.get("cell", ""))
            if group_num is not None:
                out[group_num] = label
    return out


def main() -> None:
    setup_style()
    split_by_group = split_map()
    color_by_split = {label: color for label, _, color in SPLITS}
    cond_rows = read_csv(ROOT / "Groupcondi.csv")
    groups = []
    lifetimes = []
    colors = []
    for row in cond_rows:
        group = int(to_float(row.get("Group")))
        groups.append(group)
        lifetimes.append(to_float(row.get("Mean Lifetime [weeks]")))
        colors.append(color_by_split.get(split_by_group.get(group, ""), COLORS["missing"]))

    x = np.arange(len(groups), dtype=float)
    fig, ax = plt.subplots(figsize=(13.25 * CM, 4.45 * CM), dpi=DPI)
    fig.subplots_adjust(left=0.075, right=0.805, bottom=0.205, top=0.955)
    ax.bar(x, lifetimes, width=0.72, color=colors, edgecolor="white", linewidth=0.25, zorder=2)

    ax.set_xlabel("Group", labelpad=2)
    ax.set_ylabel("Mean lifetime (weeks)", labelpad=2)
    ax.set_xlim(-0.75, len(groups) - 0.25)
    ax.set_ylim(0, max(lifetimes) * 1.10)
    tick_idx = np.arange(0, len(groups), 4)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"G{groups[i]}" for i in tick_idx], rotation=0)
    ax.grid(True, axis="y", color=COLORS["grid"], lw=0.35, zorder=1)
    style_axes(ax)

    handles = [
        mpl.patches.Patch(facecolor=color, edgecolor="none", label=label)
        for label, _, color in SPLITS
    ]
    handles.append(mpl.patches.Patch(facecolor=COLORS["missing"], edgecolor="none", label="Unused"))
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.025, 0.55),
        frameon=False,
        handlelength=1.0,
        handletextpad=0.45,
        labelspacing=0.30,
        borderaxespad=0,
    )

    save_figure(fig, OUT_DIR, "figure28")
    plt.close(fig)
    print(f"[OK] Saved: {OUT_DIR / 'figure28.png'}")
    print(f"[OK] Saved: {OUT_DIR / 'figure28.pdf'}")


if __name__ == "__main__":
    main()
