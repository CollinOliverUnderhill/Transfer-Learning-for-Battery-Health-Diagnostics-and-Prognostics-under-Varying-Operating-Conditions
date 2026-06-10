#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 3 only.

This file keeps the figure-specific drawing logic local so the figure can be
modified without touching the batch script. Shared SVG/data helpers are imported
from plot_all_thesis_figures.py.
"""

from plot_all_thesis_figures import *  # noqa: F401,F403


def fig3(paths: List[Tuple[int, str, str]]) -> None:
    cap_path = DATA / "capacity_fade" / "Release 1.0" / "G1C1.csv"
    rows = read_csv(cap_path)
    pts = [(to_float(r.get("Time")), to_float(r.get("Capacity"))) for r in rows]
    pts = [(x, y) for x, y in pts if finite(x) and finite(y)]
    if not pts:
        pts = [(i, 1.0 - 0.012 * i) for i in range(45)]
    q0 = pts[0][1]
    soh = [(x, y / q0) for x, y in pts]
    obs_week = 5.0
    eol = next((x for x, s in soh if s <= 0.70), soh[-1][0])
    w, h = 920, 560
    body = title_block(3, "EOL70 RUL Label Definition", "RUL is measured from an early observation week to the 70% SOH threshold.")
    x0, y0, pw, ph = 110, 110, 650, 340
    xmin, xmax = 0, max(eol * 1.08, max(x for x, _ in soh))
    body.append(rect(x0, y0, pw, ph, "#ffffff", PALETTE["grid"], rx=2))
    for val in [0.7, 0.8, 0.9, 1.0]:
        yy = scale(val, 0.65, 1.02, y0 + ph, y0)
        body.append(line(x0, yy, x0 + pw, yy, PALETTE["grid"], 0.7))
        body.append(text(x0 - 10, yy + 4, f"{val:.1f}", 11, anchor="end", fill=PALETTE["muted"]))
    curve = [(scale(x, xmin, xmax, x0, x0 + pw), scale(s, 0.65, 1.02, y0 + ph, y0)) for x, s in soh]
    curve = clip_series_max_y(curve, y0 + ph)
    curve = [(x, min(y0 + ph, max(y0, y))) for x, y in curve]
    body.append(polyline(curve, PALETTE["source"], 2.5))
    y70 = scale(0.7, 0.65, 1.02, y0 + ph, y0)
    xobs = scale(obs_week, xmin, xmax, x0, x0 + pw)
    xeol = scale(eol, xmin, xmax, x0, x0 + pw)
    body.append(line(x0, y70, x0 + pw, y70, PALETTE["warn"], 1.8, "6,4"))
    body.append(line(xobs, y0, xobs, y0 + ph, PALETTE["gold"], 1.6, "5,4"))
    body.append(line(xeol, y0, xeol, y0 + ph, PALETTE["warn"], 1.6, "5,4"))
    body.extend(arrow(xobs, y0 + ph + 38, xeol, y0 + ph + 38, PALETTE["transfer"]))
    body.append(text((xobs + xeol) / 2, y0 + ph + 70, f"EOL70 RUL = {max(eol - obs_week, 0):.1f} weeks", 15, "700", "middle", PALETTE["transfer"]))
    body.append(text(xobs, y0 - 10, "week 5 observation", 12, anchor="middle", fill=PALETTE["gold"]))
    body.append(text(xeol, y0 - 10, "70% EOL", 12, anchor="middle", fill=PALETTE["warn"]))
    body.append(text(x0 + pw / 2, y0 + ph + 104, "Time (weeks)", 13, anchor="middle"))
    body.append(text(42, y0 + ph / 2, "SOH", 13, anchor="middle"))
    save_svg(3, "eol70_rul_label", "EOL70 RUL Label Definition", w, h, body, paths)


def main() -> None:
    paths = []
    fig3(paths)
    for _, title, filename in paths:
        print(f"[INFO] {title} -> {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
