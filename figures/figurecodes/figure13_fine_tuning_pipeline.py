#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 13 only.

This file keeps the figure-specific drawing logic local so the figure can be
modified without touching the batch script. Shared SVG/data helpers are imported
from plot_all_thesis_figures.py.
"""

from plot_all_thesis_figures import *  # noqa: F401,F403


def fig13(paths: List[Tuple[int, str, str]]) -> None:
    draw_flow(
        13,
        "Fine-Tuning-Based Transfer-Learning Pipeline",
        [
            ("Source pretraining", "learn source representation"),
            ("Checkpoint", "select source model"),
            ("Freeze layers", "partial parameter reuse"),
            ("Target fine-tune", "limited target cells"),
            ("Target test", "held-out groups"),
        ],
        paths,
        "fine_tuning_pipeline",
    )


def main() -> None:
    paths = []
    fig13(paths)
    for _, title, filename in paths:
        print(f"[INFO] {title} -> {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
