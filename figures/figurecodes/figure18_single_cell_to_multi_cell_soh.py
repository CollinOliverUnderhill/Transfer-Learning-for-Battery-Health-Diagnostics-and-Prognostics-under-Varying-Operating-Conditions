#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 18 only.

This file keeps the figure-specific drawing logic local so the figure can be
modified without touching the batch script. Shared SVG/data helpers are imported
from plot_all_thesis_figures.py.
"""

from plot_all_thesis_figures import *  # noqa: F401,F403


def fig18(paths: List[Tuple[int, str, str]]) -> None:
    p = ROOT / "Originaltrails" / "chunqiu_codes" / "Dropped" / "SOHestimation_results" / "Ridge_Results" / "Ridge_Results_SingleCell_1fTrain_MultiCellTest" / "G1C1" / "test_cell_metrics.csv"
    rows = read_csv(p)
    data = []
    for r in rows[:70]:
        label = r.get("cell", r.get("test_cell", "cell"))
        val = to_float(r.get("mae_mean", r.get("mae")))
        if finite(val):
            data.append([label, val, PALETTE["source"]])
    draw_simple_bar_values(18, "Single-Cell-to-Multi-Cell SOH Estimation Results", data[:32], paths, "single_cell_to_multi_cell_soh", "G1C1-trained Ridge tested on other cells.")


def main() -> None:
    paths = []
    fig18(paths)
    for _, title, filename in paths:
        print(f"[INFO] {title} -> {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
