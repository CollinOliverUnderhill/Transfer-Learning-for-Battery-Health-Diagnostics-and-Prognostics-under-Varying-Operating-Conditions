#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 5 only.

This file keeps the figure-specific drawing logic local so the figure can be
modified without touching the batch script. Shared SVG/data helpers are imported
from plot_all_thesis_figures.py.
"""

from plot_all_thesis_figures import *  # noqa: F401,F403


def fig5(paths: List[Tuple[int, str, str]]) -> None:
    w, h = 980, 560
    body = title_block(5, "Week-Based Early-Feature Sample Construction", "Each cell contributes one early feature vector at the selected observation week.")
    x0, y0 = 110, 150
    body.append(line(x0, y0 + 180, x0 + 680, y0 + 180, PALETTE["ink"], 2))
    for i, wk in enumerate([0, 3, 5, 6, 10, 15]):
        x = x0 + i * 118
        body.append(line(x, y0 + 170, x, y0 + 190, PALETTE["ink"], 1.5))
        body.append(text(x, y0 + 214, f"w{wk}", 13, anchor="middle"))
        if wk in {5, 6, 10}:
            body.append(circle(x, y0 + 148, 10, PALETTE["gold"], "#ffffff"))
            body.append(text(x, y0 + 130, "feature", 11, anchor="middle", fill=PALETTE["gold"]))
    body.append(rect(190, 270, 250, 72, "#f8fafc", PALETTE["grid"], rx=5))
    body.append(text(315, 300, "early feature vector", 15, "700", "middle"))
    body.append(text(315, 324, "f1_w5, f6_w5, ...", 12, anchor="middle", fill=PALETTE["muted"]))
    body.append(rect(560, 270, 245, 72, "#f8fafc", PALETTE["grid"], rx=5))
    body.append(text(682, 300, "EOL70 RUL label", 15, "700", "middle"))
    body.append(text(682, 324, "lifetime - observation week", 12, anchor="middle", fill=PALETTE["muted"]))
    body.extend(arrow(445, 306, 555, 306, PALETTE["transfer"]))
    save_svg(5, "week_based_sample_construction", "Week-Based Early-Feature Sample Construction", w, h, body, paths)


def main() -> None:
    paths = []
    fig5(paths)
    for _, title, filename in paths:
        print(f"[INFO] {title} -> {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
