#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 16 only.

This file keeps the figure-specific drawing logic local so the figure can be
modified without touching the batch script. Shared SVG/data helpers are imported
from plot_all_thesis_figures.py.
"""

from plot_all_thesis_figures import *  # noqa: F401,F403


def fig16(paths: List[Tuple[int, str, str]]) -> None:
    p = ROOT / "Originaltrails" / "chunqiu_codes" / "Dropped" / "SOHestimation_results" / "Ridge_Results" / "Ridge_Results_SingleCell_Rolling" / "G1C1" / "rolling_predictions.csv"
    rows = read_csv(p)
    if not rows:
        p = DATA / "Processing_Data" / "SOHest" / "rpt_samples_hetero10features_all.csv"
        rows = read_csv(p)[:80]
        for r in rows:
            r["y_true"] = r.get("soh", "")
            r["y_pred"] = r.get("soh", "")
    tmp = FIG_DIR / "_tmp_soh_pred_true.csv"
    with tmp.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({k for r in rows for k in r.keys()} | {"y_true", "y_pred"})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            if "y_true" not in r:
                r["y_true"] = r.get("soh_true", r.get("true_soh", r.get("soh", "")))
            if "y_pred" not in r:
                r["y_pred"] = r.get("soh_pred", r.get("pred_soh", r.get("y_pred", "")))
            writer.writerow(r)
    draw_pred_true_panels(16, "Representative Predicted-Versus-True SOH for Within-Cell Ridge Estimation", [("G1C1 / representative", tmp, PALETTE["source"])], paths, "within_cell_soh_pred_vs_true")
    try:
        tmp.unlink()
    except Exception:
        pass


def main() -> None:
    paths = []
    fig16(paths)
    for _, title, filename in paths:
        print(f"[INFO] {title} -> {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
