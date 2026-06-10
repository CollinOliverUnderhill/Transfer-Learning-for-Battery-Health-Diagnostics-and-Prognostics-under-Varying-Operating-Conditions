#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 14 only.

This file keeps the figure-specific drawing logic local so the figure can be
modified without touching the batch script. Shared SVG/data helpers are imported
from plot_all_thesis_figures.py.
"""

from plot_all_thesis_figures import *  # noqa: F401,F403


def fig14(paths: List[Tuple[int, str, str]]) -> None:
    draw_flow(
        14,
        "Staged Optuna/TPE Hyperparameter-Search Workflow",
        [
            ("Stage 1", "source pretraining search"),
            ("Stage 2", "fine-tuning search"),
            ("Stage 3", "rerun selected config"),
            ("Evaluation", "target-test metrics"),
        ],
        paths,
        "staged_hpo_workflow",
        "Staged search separates source representation learning from target-domain adaptation.",
    )


def main() -> None:
    paths = []
    fig14(paths)
    for _, title, filename in paths:
        print(f"[INFO] {title} -> {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
