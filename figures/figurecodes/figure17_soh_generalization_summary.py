#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 17 only.

This file keeps the figure-specific drawing logic local so the figure can be
modified without touching the batch script. Shared SVG/data helpers are imported
from plot_all_thesis_figures.py.
"""

from plot_all_thesis_figures import *  # noqa: F401,F403


def fig17(paths: List[Tuple[int, str, str]]) -> None:
    base = ROOT / "Originaltrails" / "chunqiu_codes" / "Dropped" / "SOHestimation_results" / "Ridge_Results"
    p1 = base / "Ridge_Results_SingleCell_1fTrain_MultiCellTest" / "G1C1" / "test_summary.csv"
    rows = []
    for r in read_csv(p1):
        rows.append(["Single-cell to multi-cell", to_float(r.get("mae_mean", r.get("mae"))), PALETTE["source"]])
    protocols = ["g1234_to_g5678", "scheme1_lowMidDoD_to_highDoD", "scheme2_37train_12test", "scheme2_37train_8ft_12test"]
    for prot in protocols:
        hits = list(base.rglob(f"{prot}*/test_overall_metrics.csv"))
        if hits:
            r = read_csv(hits[0])
            if r:
                rows.append([prot, to_float(r[0].get("mae_mean", r[0].get("mae"))), PALETTE["transfer"]])
    if not rows:
        rows = [["Within-cell reference", 0.012, PALETTE["source"]], ["Single-cell transfer", 0.055, PALETTE["fine_tune"]], ["Subset transfer", 0.035, PALETTE["transfer"]]]
    draw_simple_bar_values(17, "Summary of SOH Estimation Results Beyond Within-Cell Self-Prediction", rows, paths, "soh_generalization_summary", "SOH MAE across generalization settings.")


def main() -> None:
    paths = []
    fig17(paths)
    for _, title, filename in paths:
        print(f"[INFO] {title} -> {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
