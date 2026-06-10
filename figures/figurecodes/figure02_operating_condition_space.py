#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 2 only.

This file keeps the figure-specific drawing logic local so the figure can be
modified without touching the batch script. Shared SVG/data helpers are imported
from plot_all_thesis_figures.py.
"""

from plot_all_thesis_figures import *  # noqa: F401,F403


def fig2(paths: List[Tuple[int, str, str]]) -> None:
    rows = read_csv(ROOT / "Groupcondi.csv")
    xs, ys, zs, life = [], [], [], []
    for r in rows:
        xs.append(to_float(r.get("Charging C-rate")))
        ys.append(to_float(r.get("Discharging C-rate")))
        zs.append(to_float(r.get("Mean DoD")))
        life.append(to_float(r.get("Mean Lifetime [weeks]")))
    lx, hx = nice_range(xs); ly, hy = nice_range(ys); lz, hz = nice_range(zs)
    ll, hl = nice_range(life)
    w, h = 1000, 720
    body = title_block(2, "Operating-Condition Space of All Cell Groups", "Point color and size encode mean lifetime.")
    x0, y0, pw, ph = 150, 130, 640, 420
    body.append(rect(x0, y0, pw, ph, "#fbfcfd", PALETTE["grid"], rx=3))
    body.append(line(x0, y0 + ph, x0 + pw, y0 + ph, PALETTE["ink"], 1.6))
    body.append(line(x0, y0 + ph, x0, y0, PALETTE["ink"], 1.6))
    body.append(line(x0, y0 + ph, x0 + 90, y0 + ph - 80, PALETTE["ink"], 1.6))
    body.append(text(x0 + pw / 2, y0 + ph + 48, "Charging C-rate", 14, anchor="middle"))
    body.append(text(x0 - 58, y0 + ph / 2, "Mean DoD (%)", 14, anchor="middle"))
    body.append(text(x0 + 108, y0 + ph - 92, "Discharging C-rate", 13, fill=PALETTE["muted"]))
    for r in rows:
        x = to_float(r.get("Charging C-rate"))
        y = to_float(r.get("Discharging C-rate"))
        z = to_float(r.get("Mean DoD"))
        lf = to_float(r.get("Mean Lifetime [weeks]"))
        gx = int(to_float(r.get("Group")))
        px = scale(x, lx, hx, x0 + 20, x0 + pw - 30) + scale(y, ly, hy, 0, 70)
        py = scale(z, lz, hz, y0 + ph - 20, y0 + 28) - scale(y, ly, hy, 0, 58)
        t = (lf - ll) / (hl - ll) if hl != ll and finite(lf) else 0.5
        col = color_ramp(1 - t)
        rr = 4 + 7 * max(0, min(1, t))
        body.append(circle(px, py, rr, col, "#ffffff", 1.0, 0.9))
        if gx in {1, 8, 16, 24, 32, 40, 48, 56, 64}:
            body.append(text(px + 8, py - 8, f"G{gx}", 10, fill=PALETTE["muted"]))
    body.append(text(830, 170, "Mean lifetime", 14, "700"))
    for i in range(6):
        val = ll + (hl - ll) * i / 5
        col = color_ramp(1 - i / 5)
        body.append(rect(835, 195 + i * 34, 36, 20, col, rx=2))
        body.append(text(880, 210 + i * 34, f"{val:.1f} w", 12, fill=PALETTE["muted"]))
    save_svg(2, "operating_condition_space", "Operating-Condition Space of All Cell Groups", w, h, body, paths)


def main() -> None:
    paths = []
    fig2(paths)
    for _, title, filename in paths:
        print(f"[INFO] {title} -> {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
