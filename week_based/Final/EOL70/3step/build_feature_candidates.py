#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORR_CSV = WORKSPACE_ROOT / "feature_engineering" / "feature_lifetime_correlations_w5_EOL70.csv"
DEFAULT_DATA_CSV = WORKSPACE_ROOT / "features" / "feature_table_all_cells_multiweek_EOL70.csv"
DEFAULT_OUT_CSV = WORKSPACE_ROOT / "features" / "informed_feature_candidates_w5_EOL70.csv"
DEFAULT_TARGET_COL = "lifetime_weeks_EOL70"
DEFAULT_MATRIX_CSV = WORKSPACE_ROOT / "features" / "correlation_matrix_w5_EOL70.csv"
DEFAULT_HEATMAP_PNG = WORKSPACE_ROOT / "features" / "correlation_heatmap_w5_EOL70.png"


def parse_float(raw: str) -> float:
    text = str(raw).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def load_relevance(corr_csv: Path) -> Dict[str, float]:
    relevance: Dict[str, float] = {}
    with corr_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            feature = str(row.get("feature_key", "")).strip()
            pearson_r = parse_float(row.get("pearson_r", ""))
            if feature:
                relevance[feature] = abs(float(pearson_r)) if math.isfinite(pearson_r) else float("nan")
    if not relevance:
        raise ValueError(f"No feature relevance found in {corr_csv}")
    return relevance


def load_columns(data_csv: Path, needed_cols: Sequence[str]) -> Dict[str, np.ndarray]:
    columns: Dict[str, List[float]] = {col: [] for col in needed_cols}
    with data_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [col for col in needed_cols if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"{data_csv} is missing required columns: {missing}")
        for row in reader:
            for col in needed_cols:
                columns[col].append(parse_float(row.get(col, "")))
    return {col: np.asarray(values, dtype=float) for col, values in columns.items()}


def pairwise_abs_pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < 3:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    if float(np.std(xv)) <= 0.0 or float(np.std(yv)) <= 0.0:
        return float("nan")
    return abs(float(np.corrcoef(xv, yv)[0, 1]))


def pairwise_pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < 3:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    if float(np.std(xv)) <= 0.0 or float(np.std(yv)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def build_rows(feature_cols: Sequence[str], relevance: Dict[str, float], columns: Dict[str, np.ndarray], max_combo_size: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    max_size = max(1, min(int(max_combo_size), len(feature_cols)))
    for combo_size in range(1, max_size + 1):
        for combo in combinations(feature_cols, combo_size):
            rel_vals = [float(relevance[col]) for col in combo if math.isfinite(float(relevance[col]))]
            if not rel_vals:
                continue
            rel_mean = float(np.mean(np.asarray(rel_vals, dtype=float)))

            redundancy_vals: List[float] = []
            if combo_size >= 2:
                for left, right in combinations(combo, 2):
                    corr = pairwise_abs_pearson(columns[left], columns[right])
                    if math.isfinite(corr):
                        redundancy_vals.append(float(corr))
            redundancy_mean = float(np.mean(np.asarray(redundancy_vals, dtype=float))) if redundancy_vals else 0.0
            redundancy_max = float(np.max(np.asarray(redundancy_vals, dtype=float))) if redundancy_vals else 0.0
            score = float(rel_mean - 0.5 * redundancy_mean)
            rows.append(
                {
                    "combo_size": combo_size,
                    "features": ",".join(combo),
                    "relevance_mean_abs_pearson": rel_mean,
                    "redundancy_mean_abs_pearson": redundancy_mean,
                    "redundancy_max_abs_pearson": redundancy_max,
                    "score": score,
                    "selector": "relevance_minus_redundancy",
                    "mrmr_last_added": "",
                    "mrmr_score": "",
                }
            )
    rows.sort(
        key=lambda row: (
            -float(row["score"]),
            -float(row["relevance_mean_abs_pearson"]),
            float(row["redundancy_mean_abs_pearson"]),
            int(row["combo_size"]),
            str(row["features"]),
        )
    )
    return rows


def write_rows(out_csv: Path, rows: Sequence[Dict[str, object]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "combo_size",
        "features",
        "relevance_mean_abs_pearson",
        "redundancy_mean_abs_pearson",
        "redundancy_max_abs_pearson",
        "score",
        "selector",
        "mrmr_last_added",
        "mrmr_score",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_correlation_matrix(columns: Dict[str, np.ndarray], ordered_cols: Sequence[str]) -> np.ndarray:
    n_cols = len(ordered_cols)
    matrix = np.full((n_cols, n_cols), np.nan, dtype=float)
    for i, left in enumerate(ordered_cols):
        for j, right in enumerate(ordered_cols):
            if i == j:
                matrix[i, j] = 1.0
            elif j < i:
                matrix[i, j] = matrix[j, i]
            else:
                matrix[i, j] = pairwise_pearson(columns[left], columns[right])
    return matrix


def short_label(col: str, target_col: str) -> str:
    if col == target_col:
        return "lifetime_week"
    return col


def feature_order_key(col: str, target_col: str) -> Tuple[int, str]:
    if col == target_col:
        return (999, col)
    head = str(col).split("_", 1)[0].lower()
    if head.startswith("f"):
        try:
            return (int(head[1:]), col)
        except ValueError:
            return (998, col)
    return (998, col)


def write_matrix_csv(out_csv: Path, ordered_cols: Sequence[str], matrix: np.ndarray, target_col: str) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    labels = [short_label(col, target_col) for col in ordered_cols]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["feature", *labels])
        for idx, row in enumerate(matrix):
            writer.writerow([labels[idx], *[f"{float(val):.6f}" if math.isfinite(float(val)) else "" for val in row]])


def plot_heatmap(out_png: Path, ordered_cols: Sequence[str], matrix: np.ndarray, target_col: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[WARN] matplotlib is not installed. Skip correlation heatmap generation.")
        return

    labels = [short_label(col, target_col) for col in ordered_cols]
    fig_w = max(8.5, 0.85 * len(labels))
    fig_h = max(7.5, 0.75 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            if not math.isfinite(float(val)):
                text = ""
            else:
                text = f"{float(val):.2f}"
            color = "white" if math.isfinite(float(val)) and abs(float(val)) >= 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation", rotation=90)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build informed week5 feature candidates for EOL70 three-step tuning.")
    ap.add_argument("--corr_csv", type=str, default=str(DEFAULT_CORR_CSV))
    ap.add_argument("--data_csv", type=str, default=str(DEFAULT_DATA_CSV))
    ap.add_argument("--out_csv", type=str, default=str(DEFAULT_OUT_CSV))
    ap.add_argument("--target_col", type=str, default=DEFAULT_TARGET_COL)
    ap.add_argument("--matrix_csv", type=str, default=str(DEFAULT_MATRIX_CSV))
    ap.add_argument("--heatmap_png", type=str, default=str(DEFAULT_HEATMAP_PNG))
    ap.add_argument("--max_combo_size", type=int, default=10)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    corr_csv = Path(args.corr_csv)
    data_csv = Path(args.data_csv)
    out_csv = Path(args.out_csv)
    matrix_csv = Path(args.matrix_csv)
    heatmap_png = Path(args.heatmap_png)

    relevance = load_relevance(corr_csv)
    feature_cols = sorted(relevance.keys(), key=lambda col: (-float(relevance[col]), col))
    needed_cols = [args.target_col, *feature_cols]
    columns = load_columns(data_csv, needed_cols)
    rows = build_rows(feature_cols, relevance, columns, max_combo_size=int(args.max_combo_size))
    write_rows(out_csv, rows)
    ordered_cols = sorted(feature_cols, key=lambda col: feature_order_key(col, args.target_col)) + [args.target_col]
    matrix = build_correlation_matrix(columns, ordered_cols)
    write_matrix_csv(matrix_csv, ordered_cols, matrix, args.target_col)
    plot_heatmap(
        heatmap_png,
        ordered_cols,
        matrix,
        args.target_col,
        title="EOL70 week5 Feature/Target Pearson Correlation Matrix",
    )
    print(f"Wrote {len(rows)} feature candidates to {out_csv}")
    print(f"Wrote correlation matrix csv to {matrix_csv}")
    print(f"Wrote correlation heatmap to {heatmap_png}")


if __name__ == "__main__":
    main()
