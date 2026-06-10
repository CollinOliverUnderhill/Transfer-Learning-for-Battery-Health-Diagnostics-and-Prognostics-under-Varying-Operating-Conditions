#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 20 only.

This file keeps the figure-specific drawing logic local so the figure can be
modified without touching the batch script. Shared SVG/data helpers are imported
from plot_all_thesis_figures.py.
"""

from plot_all_thesis_figures import *  # noqa: F401,F403


def fig20(paths: List[Tuple[int, str, str]]) -> None:
    stage = WEEK / "Final" / "EOL70" / "3step" / "outputs_400" / "BasicModel" / "stage3_final_rerun_400"
    draw_metric_bars(20, "Main rerun400 Target-Test Metrics", grouped_metrics(stage), paths, "main_rerun400_target_test_metrics")


def main() -> None:
    paths = []
    fig20(paths)
    for _, title, filename in paths:
        print(f"[INFO] {title} -> {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
