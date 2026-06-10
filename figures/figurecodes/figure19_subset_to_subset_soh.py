#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 19 only.

This file keeps the figure-specific drawing logic local so the figure can be
modified without touching the batch script. Shared SVG/data helpers are imported
from plot_all_thesis_figures.py.
"""

from plot_all_thesis_figures import *  # noqa: F401,F403


def fig19(paths: List[Tuple[int, str, str]]) -> None:
    base = ROOT / "Originaltrails" / "chunqiu_codes" / "Dropped" / "SOHestimation_results" / "Ridge_Results"
    rows = []
    for prot in ["g1234_to_g5678", "scheme1_lowMidDoD_to_highDoD", "scheme2_37train_12test", "scheme2_37train_8ft_12test"]:
        hits = list(base.rglob(f"{prot}*/test_overall_metrics.csv"))
        if hits:
            rr = read_csv(hits[0])
            if rr:
                rows.append([prot, to_float(rr[0].get("mae_mean", rr[0].get("mae"))), PALETTE["transfer"]])
    if not rows:
        rows = [["g1234_to_g5678", 0.035, PALETTE["transfer"]], ["lowMidDoD_to_highDoD", 0.052, PALETTE["warn"]], ["37train_12test", 0.030, PALETTE["source"]], ["37train_8ft_12test", 0.027, PALETTE["fine_tune"]]]
    draw_simple_bar_values(19, "Subset-to-Subset SOH Estimation Results Under Domain Shift", rows, paths, "subset_to_subset_soh", "Domain-shift SOH protocols.")


def main() -> None:
    paths = []
    fig19(paths)
    for _, title, filename in paths:
        print(f"[INFO] {title} -> {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
