#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze correlation between the 10 engineered features and lifetime.

Default target:
  lifetime_weeks_EOL80

Outputs under:
  E:/Datasets/IVAS/Processing_data_fe/
    - feature_lifetime_correlations_EOL80.csv
    - feature_lifetime_correlations_EOL80.md
    - feature_lifetime_correlations_EOL80_ranked.svg
    - feature_lifetime_correlations_EOL80_index.html
    - feature_plots_EOL80/
        - step01_*.svg ... step10_*.svg

This script intentionally avoids pandas/matplotlib and writes SVG directly.
"""

from __future__ import annotations

import csv
import html
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


DATA_CSV = Path(r"E:\Datasets\IVAS\Processing_Data\Lifetime_prediction\ivas_lifetime_10features_per_cell.csv")
OUT_DIR = Path(r"E:\Datasets\IVAS\Processing_data_fe")
TARGET_COLS = [
    "lifetime_weeks_EOL80",
    "lifetime_weeks_EOL75",
    "lifetime_weeks_EOL70",
    "lifetime_weeks_EOL65",
    "lifetime_weeks_EOL60",
    "lifetime_weeks_EOL55",
    "lifetime_weeks_EOL50",
]

FEATURES: List[Dict[str, str]] = [
    {"step": "1", "key": "step1_log_abs_mean_delta_dQdV_w3_w0_3p6_3p9", "label": "Step 1: log|mean dQ/dV delta 3.6-3.9V|"},
    {"step": "2", "key": "step2_log_abs_delta_CV_time_w3_w0", "label": "Step 2: log|delta CV time|"},
    {"step": "3", "key": "step3_DoD", "label": "Step 3: DoD"},
    {"step": "4", "key": "step4_delta_Q1_DVA_w3_w0", "label": "Step 4: delta Q1 DVA"},
    {"step": "5", "key": "step5_sqrt_Cchg_sqrt_DoD", "label": "Step 5: sqrt(Cchg)*sqrt(DoD)"},
    {"step": "6", "key": "step6_Cchg", "label": "Step 6: Cchg"},
    {"step": "7", "key": "step7_log_abs_var_delta_dQdV_w3_w0_3p0_3p6", "label": "Step 7: log|var dQ/dV delta 3.0-3.6V|"},
    {"step": "8", "key": "step8_delta_Q3_DVA_w3_w0", "label": "Step 8: delta Q3 DVA"},
    {"step": "9", "key": "step9_log_abs_mean_delta_dQdV_w3_w0_3p0_3p6", "label": "Step 9: log|mean dQ/dV delta 3.0-3.6V|"},
    {"step": "10", "key": "step10_log_abs_CV_time_w0", "label": "Step 10: log|CV time w0|"},
]

GROUP_MIN = 1
GROUP_MAX = 64
GROUP_COLOR_STOPS: List[Tuple[float, str]] = [
    (0.00, "#440154"),
    (0.25, "#3b528b"),
    (0.50, "#21918c"),
    (0.75, "#5ec962"),
    (1.00, "#fde725"),
]


def to_float(value: str) -> float:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return float("nan")
    try:
        return float(text)
    except Exception:
        return float("nan")


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_dev = x - x_mean
    y_dev = y - y_mean
    denom = math.sqrt(float(np.sum(x_dev**2) * np.sum(y_dev**2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x_dev * y_dev) / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    return pearson_corr(rankdata_average(x), rankdata_average(y))


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def finite_xy_group(rows: Sequence[Dict[str, str]], x_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_vals: List[float] = []
    y_vals: List[float] = []
    group_vals: List[int] = []
    for row in rows:
        x = to_float(row.get(x_col, ""))
        y = to_float(row.get(y_col, ""))
        group_raw = to_float(row.get("group_num", ""))
        if not np.isfinite(group_raw):
            continue
        group = int(group_raw)
        if np.isfinite(x) and np.isfinite(y) and GROUP_MIN <= group <= GROUP_MAX:
            x_vals.append(x)
            y_vals.append(y)
            group_vals.append(group)
    return np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float), np.asarray(group_vals, dtype=int)


def nice_limits(values: np.ndarray) -> Tuple[float, float]:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmin == vmax:
        pad = 1.0 if vmin == 0 else abs(vmin) * 0.05
        return vmin - pad, vmax + pad
    pad = (vmax - vmin) * 0.06
    return vmin - pad, vmax + pad


def svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#fffdf8"/>',
    ]


def svg_footer() -> List[str]:
    return ["</svg>"]


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def interpolate_hex(color_a: str, color_b: str, t: float) -> str:
    rgb_a = hex_to_rgb(color_a)
    rgb_b = hex_to_rgb(color_b)
    rgb = tuple(int(round(a + (b - a) * t)) for a, b in zip(rgb_a, rgb_b))
    return rgb_to_hex(rgb)


def group_to_color(group_num: int) -> str:
    if group_num <= GROUP_MIN:
        return GROUP_COLOR_STOPS[0][1]
    if group_num >= GROUP_MAX:
        return GROUP_COLOR_STOPS[-1][1]

    norm = (group_num - GROUP_MIN) / float(GROUP_MAX - GROUP_MIN)
    for (left_pos, left_color), (right_pos, right_color) in zip(GROUP_COLOR_STOPS[:-1], GROUP_COLOR_STOPS[1:]):
        if norm <= right_pos:
            span = right_pos - left_pos
            frac = 0.0 if span == 0 else (norm - left_pos) / span
            return interpolate_hex(left_color, right_color, frac)
    return GROUP_COLOR_STOPS[-1][1]


def scatter_svg(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    pearson_r: float,
    spearman_r: float,
    out_path: Path,
) -> None:
    width = 1020
    height = 700
    left = 90
    right = 150
    top = 90
    bottom = 85
    plot_w = width - left - right
    plot_h = height - top - bottom

    xmin, xmax = nice_limits(x)
    ymin, ymax = nice_limits(y)

    def xpix(val: float) -> float:
        return left + (val - xmin) / (xmax - xmin) * plot_w

    def ypix(val: float) -> float:
        return top + plot_h - (val - ymin) / (ymax - ymin) * plot_h

    lines = svg_header(width, height)
    lines.append("<defs>")
    lines.append('<linearGradient id="group-gradient" x1="0%" y1="100%" x2="0%" y2="0%">')
    for pos, color in GROUP_COLOR_STOPS:
        lines.append(f'<stop offset="{pos * 100:.1f}%" stop-color="{color}"/>')
    lines.append("</linearGradient>")
    lines.append("</defs>")
    lines.append(f'<text x="{width/2:.1f}" y="38" text-anchor="middle" font-size="24" font-family="Arial" fill="#222">{html.escape(title)}</text>')
    lines.append(f'<text x="{width/2:.1f}" y="66" text-anchor="middle" font-size="16" font-family="Arial" fill="#444">n={len(x)} | Pearson r={pearson_r:.4f} | Spearman r={spearman_r:.4f}</text>')
    lines.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#333" stroke-width="1.5"/>')

    for frac in np.linspace(0, 1, 6):
        xx = left + frac * plot_w
        yy = top + frac * plot_h
        lines.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{top + plot_h}" stroke="#ece7db" stroke-width="1"/>')
        lines.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left + plot_w}" y2="{yy:.2f}" stroke="#ece7db" stroke-width="1"/>')

    for val in np.linspace(xmin, xmax, 6):
        xx = xpix(float(val))
        lines.append(f'<line x1="{xx:.2f}" y1="{top + plot_h}" x2="{xx:.2f}" y2="{top + plot_h + 6}" stroke="#333" stroke-width="1.5"/>')
        lines.append(f'<text x="{xx:.2f}" y="{top + plot_h + 26}" text-anchor="middle" font-size="12" font-family="Arial" fill="#444">{val:.3g}</text>')

    for val in np.linspace(ymin, ymax, 6):
        yy = ypix(float(val))
        lines.append(f'<line x1="{left - 6}" y1="{yy:.2f}" x2="{left}" y2="{yy:.2f}" stroke="#333" stroke-width="1.5"/>')
        lines.append(f'<text x="{left - 10}" y="{yy + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#444">{val:.3g}</text>')

    slope, intercept = fit_line(x, y)
    if np.isfinite(slope) and np.isfinite(intercept):
        x1v, x2v = float(np.min(x)), float(np.max(x))
        y1v = slope * x1v + intercept
        y2v = slope * x2v + intercept
        lines.append(
            f'<line x1="{xpix(x1v):.2f}" y1="{ypix(y1v):.2f}" '
            f'x2="{xpix(x2v):.2f}" y2="{ypix(y2v):.2f}" stroke="#b22222" stroke-width="2.5"/>'
        )

    for xv, yv, group_num in zip(x, y, groups):
        fill = group_to_color(int(group_num))
        lines.append(
            f'<circle cx="{xpix(float(xv)):.2f}" cy="{ypix(float(yv)):.2f}" r="4.2" '
            f'fill="{fill}" fill-opacity="0.85" stroke="#1f1f1f" stroke-width="0.5"/>'
        )

    cbar_x = left + plot_w + 42
    cbar_y = top
    cbar_w = 22
    cbar_h = plot_h
    lines.append(f'<rect x="{cbar_x}" y="{cbar_y}" width="{cbar_w}" height="{cbar_h}" fill="url(#group-gradient)" stroke="#333" stroke-width="1"/>')
    tick_groups = [1, 16, 32, 48, 64]
    lines.append(f'<text x="{cbar_x + cbar_w/2:.1f}" y="{cbar_y - 14}" text-anchor="middle" font-size="14" font-family="Arial" fill="#222">Group</text>')
    for tick_group in tick_groups:
        tick_frac = (tick_group - GROUP_MIN) / float(GROUP_MAX - GROUP_MIN)
        tick_y = cbar_y + cbar_h - tick_frac * cbar_h
        lines.append(f'<line x1="{cbar_x + cbar_w}" y1="{tick_y:.2f}" x2="{cbar_x + cbar_w + 8}" y2="{tick_y:.2f}" stroke="#333" stroke-width="1.2"/>')
        lines.append(
            f'<text x="{cbar_x + cbar_w + 12}" y="{tick_y + 4:.2f}" text-anchor="start" font-size="12" '
            f'font-family="Arial" fill="#444">G{tick_group}</text>'
        )

    lines.append(f'<text x="{left + plot_w/2:.1f}" y="{height - 24}" text-anchor="middle" font-size="16" font-family="Arial" fill="#222">{html.escape(x_label)}</text>')
    lines.append(
        f'<text x="28" y="{top + plot_h/2:.1f}" text-anchor="middle" font-size="16" '
        f'font-family="Arial" fill="#222" transform="rotate(-90 28 {top + plot_h/2:.1f})">{html.escape(y_label)}</text>'
    )
    lines.extend(svg_footer())
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_bar_svg(results: Sequence[Dict[str, object]], target_col: str, out_path: Path) -> None:
    ranked = sorted(results, key=lambda row: abs(float(row["pearson_r"])) if np.isfinite(float(row["pearson_r"])) else -1.0, reverse=True)
    width = 1100
    height = 650
    left = 260
    right = 70
    top = 80
    bottom = 70
    bar_gap = 14
    bar_h = (height - top - bottom - bar_gap * (len(ranked) - 1)) / len(ranked)
    plot_w = width - left - right

    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2:.1f}" y="38" text-anchor="middle" font-size="24" font-family="Arial" fill="#222">Feature vs {html.escape(target_col)} Correlation Ranking</text>')
    lines.append(f'<text x="{width/2:.1f}" y="62" text-anchor="middle" font-size="15" font-family="Arial" fill="#555">Bars show absolute Pearson correlation; labels show signed r.</text>')
    lines.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{height-top-bottom}" fill="#ffffff" stroke="#333" stroke-width="1.5"/>')

    for frac in np.linspace(0, 1, 6):
        xx = left + frac * plot_w
        val = frac
        lines.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{height-bottom}" stroke="#ece7db" stroke-width="1"/>')
        lines.append(f'<text x="{xx:.2f}" y="{height-bottom+24}" text-anchor="middle" font-size="12" font-family="Arial" fill="#444">{val:.1f}</text>')

    for i, row in enumerate(ranked):
        y = top + i * (bar_h + bar_gap)
        bar_w = max(0.0, min(1.0, abs(float(row["pearson_r"])) if np.isfinite(float(row["pearson_r"])) else 0.0)) * plot_w
        color = "#0f766e" if float(row["pearson_r"]) >= 0 else "#b45309"
        lines.append(f'<text x="{left - 12}" y="{y + bar_h*0.68:.2f}" text-anchor="end" font-size="14" font-family="Arial" fill="#222">{html.escape(str(row["feature_short"]))}</text>')
        lines.append(f'<rect x="{left}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{color}" fill-opacity="0.85"/>')
        lines.append(f'<text x="{left + bar_w + 8:.2f}" y="{y + bar_h*0.68:.2f}" text-anchor="start" font-size="13" font-family="Arial" fill="#333">r={float(row["pearson_r"]):.4f}</text>')

    lines.extend(svg_footer())
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, target_col: str, rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        f"# Feature Correlation Summary for {target_col}",
        "",
        "| Rank | Step | Feature | Valid N | Pearson r | Spearman r |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    ranked = sorted(rows, key=lambda row: abs(float(row["pearson_r"])) if np.isfinite(float(row["pearson_r"])) else -1.0, reverse=True)
    for idx, row in enumerate(ranked, start=1):
        lines.append(
            f"| {idx} | {row['step']} | {row['feature_short']} | {row['valid_n']} | "
            f"{float(row['pearson_r']):.4f} | {float(row['spearman_r']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html_index(
    path: Path,
    target_col: str,
    rows: Sequence[Dict[str, object]],
    plot_dir_name: str,
    summary_svg: Path,
) -> None:
    ranked = sorted(rows, key=lambda row: abs(float(row["pearson_r"])) if np.isfinite(float(row["pearson_r"])) else -1.0, reverse=True)
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>Feature Correlation Analysis - {html.escape(target_col)}</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#fffdf8;color:#222;}",
        "h1,h2{margin:0 0 12px 0;}",
        "table{border-collapse:collapse;margin:18px 0;width:100%;max-width:980px;}",
        "th,td{border:1px solid #d7d0c0;padding:8px 10px;text-align:left;}",
        "th{background:#f4efe3;}",
        ".plot{margin:26px 0;padding:16px;border:1px solid #ddd3bf;background:#fff;max-width:980px;}",
        "img{max-width:100%;height:auto;display:block;}",
        "</style></head><body>",
        f"<h1>Feature Correlation Analysis for {html.escape(target_col)}</h1>",
        "<p>Scatter plots and correlation statistics for the 10 engineered lifetime features.</p>",
        "<h2>Ranking Summary</h2>",
        f'<div class="plot"><img src="{html.escape(summary_svg.name)}" alt="summary ranking"></div>',
        "<table><thead><tr><th>Rank</th><th>Step</th><th>Feature</th><th>Valid N</th><th>Pearson r</th><th>Spearman r</th></tr></thead><tbody>",
    ]
    for idx, row in enumerate(ranked, start=1):
        parts.append(
            "<tr>"
            f"<td>{idx}</td><td>{html.escape(str(row['step']))}</td><td>{html.escape(str(row['feature_short']))}</td>"
            f"<td>{row['valid_n']}</td><td>{float(row['pearson_r']):.4f}</td><td>{float(row['spearman_r']):.4f}</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")
    parts.append("<h2>Scatter Plots</h2>")
    for row in ranked:
        parts.append(
            f'<div class="plot"><h3>{html.escape(str(row["feature_short"]))}</h3>'
            f'<img src="{html.escape(plot_dir_name + "/" + str(row["plot_file"]))}" alt="{html.escape(str(row["feature_short"]))}"></div>'
        )
    parts.append("</body></html>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows(DATA_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for target_col in TARGET_COLS:
        target_suffix = target_col.replace("lifetime_weeks_", "")
        target_dir = OUT_DIR / target_suffix
        plots_dir = target_dir / f"feature_plots_{target_suffix}"
        plots_dir.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, object]] = []
        for feat in FEATURES:
            x, y, groups = finite_xy_group(rows, feat["key"], target_col)
            pearson_r = pearson_corr(x, y)
            spearman_r = spearman_corr(x, y)
            plot_file = f"step{feat['step'].zfill(2)}_{feat['key']}.svg"
            scatter_svg(
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
            results.append(
                {
                    "step": feat["step"],
                    "feature_key": feat["key"],
                    "feature_short": feat["label"],
                    "target_key": target_col,
                    "valid_n": len(x),
                    "pearson_r": pearson_r,
                    "spearman_r": spearman_r,
                    "plot_file": plot_file,
                }
            )

        summary_svg = target_dir / f"feature_lifetime_correlations_{target_suffix}_ranked.svg"
        summary_csv = target_dir / f"feature_lifetime_correlations_{target_suffix}.csv"
        summary_md = target_dir / f"feature_lifetime_correlations_{target_suffix}.md"
        index_html = target_dir / f"feature_lifetime_correlations_{target_suffix}_index.html"

        summary_bar_svg(results, target_col, summary_svg)
        write_csv(
            summary_csv,
            results,
            ["step", "feature_key", "feature_short", "target_key", "valid_n", "pearson_r", "spearman_r", "plot_file"],
        )
        write_markdown(summary_md, target_col, results)
        write_html_index(index_html, target_col, results, plots_dir.name, summary_svg)

        print(f"[INFO] target        : {target_col}")
        print(f"[INFO] plots dir     : {plots_dir}")
        print(f"[INFO] summary csv   : {summary_csv}")
        print(f"[INFO] summary md    : {summary_md}")
        print(f"[INFO] summary svg   : {summary_svg}")
        print(f"[INFO] index html    : {index_html}")


if __name__ == "__main__":
    main()
