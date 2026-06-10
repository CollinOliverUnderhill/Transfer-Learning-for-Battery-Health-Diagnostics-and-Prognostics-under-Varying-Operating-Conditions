#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 10 only.

This file keeps the figure-specific drawing logic local so the figure can be
modified without touching the batch script. Shared SVG/data helpers are imported
from plot_all_thesis_figures.py.
"""

from plot_all_thesis_figures import *  # noqa: F401,F403


def fig10(paths: List[Tuple[int, str, str]]) -> None:
    cond = read_csv(ROOT / "Groupcondi.csv")
    stage = WEEK / "Final" / "EOL70" / "3step" / "outputs_400" / "BasicModel" / "stage3_final_rerun_400"
    split_by_group: Dict[int, str] = {}
    for label, cell, g in split_rows(stage):
        gv = int(to_float(g)) if finite(to_float(g)) else group_num_from_cell(cell)
        if gv is not None:
            split_by_group[gv] = label
    colors = {"Source train": PALETTE["source"], "Target fine-tune": PALETTE["fine_tune"], "Target test": PALETTE["test"]}
    w, h = 1000, 680
    body = title_block(10, "EOL70 Source, Target Fine-Tuning, and Target Test Split", "The same operating-condition space, colored by modelling partition.")
    x0, y0, pw, ph = 140, 120, 640, 410
    body.append(rect(x0, y0, pw, ph, "#fbfcfd", PALETTE["grid"], rx=3))
    xs = [to_float(r.get("Charging C-rate")) for r in cond]; ys = [to_float(r.get("Discharging C-rate")) for r in cond]; zs = [to_float(r.get("Mean DoD")) for r in cond]
    lx, hx = nice_range(xs); ly, hy = nice_range(ys); lz, hz = nice_range(zs)
    for r in cond:
        g = int(to_float(r.get("Group")))
        label = split_by_group.get(g, "unused")
        col = colors.get(label, "#cfd6df")
        x = scale(to_float(r.get("Charging C-rate")), lx, hx, x0 + 20, x0 + pw - 35) + scale(to_float(r.get("Discharging C-rate")), ly, hy, 0, 70)
        y = scale(to_float(r.get("Mean DoD")), lz, hz, y0 + ph - 20, y0 + 30) - scale(to_float(r.get("Discharging C-rate")), ly, hy, 0, 58)
        body.append(circle(x, y, 7.5, col, "#ffffff", 1.0, 0.92))
    for i, (label, col) in enumerate(colors.items()):
        body.append(rect(815, 170 + i * 36, 18, 18, col, rx=2))
        body.append(text(842, 184 + i * 36, label, 13))
    save_svg(10, "eol70_domain_split", "EOL70 Source, Target Fine-Tuning, and Target Test Split", w, h, body, paths)


def main() -> None:
    paths = []
    fig10(paths)
    for _, title, filename in paths:
        print(f"[INFO] {title} -> {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
