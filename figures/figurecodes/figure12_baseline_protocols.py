#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 12 only.

This file keeps the figure-specific drawing logic local so the figure can be
modified without touching the batch script. Shared SVG/data helpers are imported
from plot_all_thesis_figures.py.
"""

from plot_all_thesis_figures import *  # noqa: F401,F403


def fig12(paths: List[Tuple[int, str, str]]) -> None:
    draw_flow(
        12,
        "Source-Only and Benchmark Baseline Protocols",
        [
            ("Source-only", "source train"),
            ("Direct test", "target test"),
            ("Benchmark", "non-transfer train"),
            ("Compare", "same target test"),
        ],
        paths,
        "baseline_protocols",
        "Baselines define whether fine-tuning actually improves cross-domain RUL prediction.",
    )


def main() -> None:
    paths = []
    fig12(paths)
    for _, title, filename in paths:
        print(f"[INFO] {title} -> {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
