#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 9 only.

This file keeps the figure-specific drawing logic local so the figure can be
modified without touching the batch script. Shared SVG/data helpers are imported
from plot_all_thesis_figures.py.
"""

from plot_all_thesis_figures import *  # noqa: F401,F403


def fig9(paths: List[Tuple[int, str, str]]) -> None:
    config = read_json(WEEK / "Final" / "EOL70" / "3step" / "outputs_400" / "BasicModel" / "stage3_final_rerun_400" / "transfer_model" / "config.json")
    rows = []
    for f in config.get("x_cols", []):
        rows.append([str(f), feature_label(str(f)), f"week {config.get('feature_week', 5)}", "main rerun400 input"])
    rows.append(["hidden dims", ",".join(map(str, config.get("hidden_dims", []))), "MLP", "transfer model"])
    draw_table(9, "Final Input Feature Combination Used in the Main rerun400 Experiment", ["Item", "Meaning / value", "Scope", "Role"], rows, paths, "main_rerun400_input_features")


def main() -> None:
    paths = []
    fig9(paths)
    for _, title, filename in paths:
        print(f"[INFO] {title} -> {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
