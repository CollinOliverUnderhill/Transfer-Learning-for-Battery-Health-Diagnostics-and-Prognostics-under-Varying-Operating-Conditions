#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 15 only.

This file keeps the figure-specific drawing logic local so the figure can be
modified without touching the batch script. Shared SVG/data helpers are imported
from plot_all_thesis_figures.py.
"""

from plot_all_thesis_figures import *  # noqa: F401,F403


def fig15(paths: List[Tuple[int, str, str]]) -> None:
    p = ROOT / "Originaltrails" / "chunqiu_codes" / "Dropped" / "SOHestimation_results" / "Ridge_Results" / "Ridge_Results_SingleCell" / "_seed_summary_mape" / "cell_seed_summary.csv"
    rows = read_csv(p)
    data = [
        ("Cell MAE", [to_float(r.get("test_mae_mean")) for r in rows], PALETTE["source"]),
        ("Seed std", [to_float(r.get("test_mae_seed_std")) for r in rows], PALETTE["fine_tune"]),
        ("MAPE (%)", [to_float(r.get("test_mape_percent_mean")) for r in rows], PALETTE["transfer"]),
    ]
    draw_box_like(15, "Within-Cell SOH Estimation Summary Across Cells", data, paths, "within_cell_soh_summary", "Ridge within-cell SOH estimation, summarized across cells.")


def main() -> None:
    paths = []
    fig15(paths)
    for _, title, filename in paths:
        print(f"[INFO] {title} -> {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
