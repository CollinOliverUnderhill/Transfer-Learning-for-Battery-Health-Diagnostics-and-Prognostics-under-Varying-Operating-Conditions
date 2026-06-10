#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 4 only.

This file keeps the figure-specific drawing logic local so the figure can be
modified without touching the batch script. Shared SVG/data helpers are imported
from plot_all_thesis_figures.py.
"""

from plot_all_thesis_figures import *  # noqa: F401,F403


def fig4(paths: List[Tuple[int, str, str]]) -> None:
    draw_flow(
        4,
        "Dataset and Sample-Construction Pipeline",
        [
            ("Raw RPT / cycling", "capacity and Q(V)"),
            ("Cleaning", "valid cells and RPTs"),
            ("SOH samples", "RPT-level health labels"),
            ("RUL samples", "week-based early features"),
            ("Domain split", "source / fine-tune / test"),
            ("Models", "Ridge and transfer MLP"),
        ],
        paths,
        "dataset_sample_pipeline",
        "The SOH and RUL tasks use the same degradation data in different sample forms.",
    )


def main() -> None:
    paths = []
    fig4(paths)
    for _, title, filename in paths:
        print(f"[INFO] {title} -> {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
