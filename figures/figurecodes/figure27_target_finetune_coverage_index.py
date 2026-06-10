#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 27 in the PulseBat-combo plotting style."""

from __future__ import annotations

import math
from statistics import mean

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
    ROOT,
    WEEK,
    finite,
    read_csv,
    save_figure,
    setup_style,
    style_axes,
    to_float,
)


BASE = WEEK / "Final" / "EOL70" / "3step" / "outputs_400" / "random_w5_EOL70_10seeds_legacy400"
OUT_DIR = FIG_DIR / "figure27"


def load_coverage_rows() -> list[tuple[str, float, float]]:
    cond_rows = read_csv(ROOT / "Groupcondi.csv")
    cond = {
        int(to_float(row.get("Group"))): (
            to_float(row.get("Charging C-rate")),
            to_float(row.get("Discharging C-rate")),
            to_float(row.get("Mean DoD")) / 100.0,
        )
        for row in cond_rows
        if finite(to_float(row.get("Group")))
    }

    rows: list[tuple[str, float, float]] = []
    for seed_dir in sorted(BASE.glob("seed*")):
        stage = seed_dir / "stage3_final"
        bench = read_csv(stage / "benchmark" / "test_overall_metrics.csv")
        ft = read_csv(stage / "transfer_model" / "test_overall_metrics.csv")
        if not bench or not ft:
            continue
        ft_groups = sorted({
            int(to_float(row.get("group_num")))
            for row in read_csv(stage / "target_finetune_samples.csv")
            if finite(to_float(row.get("group_num")))
        })
        test_groups = sorted({
            int(to_float(row.get("group_num")))
            for row in read_csv(stage / "target_test_samples.csv")
            if finite(to_float(row.get("group_num")))
        })
        dists: list[float] = []
        for test_group in test_groups:
            if test_group not in cond or not ft_groups:
                continue
            tx, ty, tz = cond[test_group]
            best = min(
                math.sqrt((tx - cond[g][0]) ** 2 + (ty - cond[g][1]) ** 2 + (tz - cond[g][2]) ** 2)
                for g in ft_groups
                if g in cond
            )
            dists.append(best)
        coverage = 1.0 / (1.0 + mean(dists)) if dists else float("nan")
        bm_mae = to_float(bench[0].get("mae"))
        ft_mae = to_float(ft[0].get("mae"))
        improvement = (bm_mae - ft_mae) / bm_mae * 100.0 if bm_mae else float("nan")
        rows.append((seed_dir.name.replace("seed", ""), coverage, improvement))
    return rows


def zero_trim(value: float, _pos: int | None = None) -> str:
    if abs(value) < 1e-12:
        return "0"
    if abs(value) < 1:
        return f"{value:.2f}"
    return f"{value:.0f}"


def main() -> None:
    setup_style()
    rows = load_coverage_rows()
    x = np.asarray([row[1] for row in rows], dtype=float)
    y = np.asarray([row[2] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.72 * CM, 4.60 * CM), dpi=DPI)
    fig.subplots_adjust(left=0.145, right=0.965, bottom=0.205, top=0.955)

    ax.axhline(0.0, color=COLORS["mid_grey"], lw=0.75, ls=(0, (3.0, 2.0)), zorder=1)
    ax.scatter(
        x,
        y,
        s=22,
        color=COLORS["transfer"],
        edgecolors="white",
        linewidths=0.55,
        zorder=3,
    )
    ax.set_xlabel("Coverage index", labelpad=2)
    ax.set_ylabel("MAE improvement (%)", labelpad=2)
    ax.set_xlim(float(np.nanmin(x)) - 0.015, float(np.nanmax(x)) + 0.025)
    y_abs = max(abs(float(np.nanmin(y))), abs(float(np.nanmax(y))), 5.0)
    ax.set_ylim(-1.12 * y_abs, 1.12 * y_abs)
    ax.yaxis.set_major_formatter(FuncFormatter(zero_trim))
    ax.grid(True, axis="y", color=COLORS["grid"], lw=0.35, zorder=0)
    style_axes(ax)

    save_figure(fig, OUT_DIR, "figure27")
    plt.close(fig)
    print(f"[OK] Saved: {OUT_DIR / 'figure27.png'}")
    print(f"[OK] Saved: {OUT_DIR / 'figure27.pdf'}")


if __name__ == "__main__":
    main()
