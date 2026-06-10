#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 21 only.

This file keeps the figure-specific drawing logic local so the figure can be
modified without touching the batch script. Shared SVG/data helpers are imported
from plot_all_thesis_figures.py.
"""

from plot_all_thesis_figures import *  # noqa: F401,F403


def fig21(paths: List[Tuple[int, str, str]]) -> None:
    stage = WEEK / "Final" / "EOL70" / "3step" / "outputs_400" / "BasicModel" / "stage3_final_rerun_400"
    draw_pred_true_panels(
        21,
        "Predicted Versus True RUL for the Main rerun400 EOL70 Run",
        [
            ("Benchmark", stage / "benchmark" / "predictions_test.csv", PALETTE["benchmark"]),
            ("Source-only", stage / "transfer_model" / "predictions_test_source_only.csv", PALETTE["source_only"]),
            ("Fine-tuned", stage / "transfer_model" / "predictions_test.csv", PALETTE["transfer"]),
        ],
        paths,
        "main_rerun400_pred_vs_true",
    )


def main() -> None:
    paths = []
    fig21(paths)
    for _, title, filename in paths:
        print(f"[INFO] {title} -> {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
