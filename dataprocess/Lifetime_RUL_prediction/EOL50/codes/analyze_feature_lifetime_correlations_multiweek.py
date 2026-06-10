#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze lifetime-feature correlations for multi-week engineered features.

Outputs are written into the existing Processing_Data_dd_exclude/EOLxx folders with
explicit week suffixes, for example:
  - feature_lifetime_correlations_w5_EOL60.csv
  - feature_lifetime_correlations_w5_EOL60.md
  - feature_lifetime_correlations_w5_EOL60_ranked.svg
  - feature_lifetime_correlations_w5_EOL60_index.html
  - feature_plots_w5_EOL60/
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import analyze_feature_lifetime_correlations as base


DATA_CSV = Path(r"E:\Datasets\IVAS\Processing_Data\Lifetime_prediction\ivas_lifetime_10features_multiweek_per_cell.csv")
OUT_DIR = Path(r"E:\Datasets\IVAS\Processing_Data_dd_exclude")
WEEKS: Tuple[int, ...] = (3, 5, 10, 15)

STEP_LABELS: Dict[str, str] = {str(item["step"]): str(item["label"]) for item in base.FEATURES}


def feature_defs_for_week(week: int) -> List[Dict[str, str]]:
    features: List[Dict[str, str]] = []
    for step in range(1, 11):
        step_str = str(step)
        features.append(
            {
                "step": step_str,
                "key": f"f{step}_w{week}",
                "label": f"{STEP_LABELS[step_str]} [w{week}]",
            }
        )
    return features


def write_multiweek_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    base.write_csv(
        path,
        rows,
        ["week", "step", "feature_key", "feature_short", "target_key", "valid_n", "pearson_r", "spearman_r", "plot_file"],
    )


def main() -> None:
    rows = base.load_rows(DATA_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for target_col in base.TARGET_COLS:
        target_suffix = target_col.replace("lifetime_weeks_", "")
        target_dir = OUT_DIR / target_suffix
        target_dir.mkdir(parents=True, exist_ok=True)

        combined_rows: List[Dict[str, object]] = []

        for week in WEEKS:
            feat_defs = feature_defs_for_week(week)
            plots_dir = target_dir / f"feature_plots_w{week}_{target_suffix}"
            plots_dir.mkdir(parents=True, exist_ok=True)

            results: List[Dict[str, object]] = []
            for feat in feat_defs:
                x, y, groups = base.finite_xy_group(rows, feat["key"], target_col)
                pearson_r = base.pearson_corr(x, y)
                spearman_r = base.spearman_corr(x, y)
                plot_file = f"step{feat['step'].zfill(2)}_{feat['key']}.svg"
                base.scatter_svg(
                    x=x,
                    y=y,
                    groups=groups,
                    title=f"{feat['label']} vs {target_col}",
                    x_label=feat["label"],
                    y_label=target_col,
                    pearson_r=pearson_r,
                    spearman_r=spearman_r,
                    out_path=plots_dir / plot_file,
                )
                result = {
                    "week": week,
                    "step": feat["step"],
                    "feature_key": feat["key"],
                    "feature_short": feat["label"],
                    "target_key": target_col,
                    "valid_n": len(x),
                    "pearson_r": pearson_r,
                    "spearman_r": spearman_r,
                    "plot_file": plot_file,
                }
                results.append(result)
                combined_rows.append(result)

            summary_svg = target_dir / f"feature_lifetime_correlations_w{week}_{target_suffix}_ranked.svg"
            summary_csv = target_dir / f"feature_lifetime_correlations_w{week}_{target_suffix}.csv"
            summary_md = target_dir / f"feature_lifetime_correlations_w{week}_{target_suffix}.md"
            index_html = target_dir / f"feature_lifetime_correlations_w{week}_{target_suffix}_index.html"

            base.summary_bar_svg(results, f"{target_col} [w{week}]", summary_svg)
            base.write_csv(
                summary_csv,
                results,
                ["week", "step", "feature_key", "feature_short", "target_key", "valid_n", "pearson_r", "spearman_r", "plot_file"],
            )
            base.write_markdown(summary_md, f"{target_col} [w{week}]", results)
            base.write_html_index(index_html, f"{target_col} [w{week}]", results, plots_dir.name, summary_svg)

            print(f"[INFO] target/week    : {target_col} / w{week}")
            print(f"[INFO] plots dir      : {plots_dir}")
            print(f"[INFO] summary csv    : {summary_csv}")
            print(f"[INFO] summary md     : {summary_md}")
            print(f"[INFO] summary svg    : {summary_svg}")
            print(f"[INFO] index html     : {index_html}")

        combined_path = target_dir / f"feature_lifetime_correlations_multiweek_{target_suffix}.csv"
        write_multiweek_summary(combined_path, combined_rows)
        print(f"[INFO] multiweek csv   : {combined_path}")


if __name__ == "__main__":
    main()
