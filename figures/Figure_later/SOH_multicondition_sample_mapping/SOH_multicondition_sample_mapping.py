#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


ROOT = Path(r"E:\Datasets\IVAS")
SOH_CSV = ROOT / "Data" / "Processing_Data" / "SOHest" / "rpt_samples_feature_soh.csv"
FEATURE_CSV = (
    ROOT
    / "week_based"
    / "Final"
    / "EOL70"
    / "features"
    / "feature_table_all_cells_multiweek_EOL70.csv"
)
OUT_DIR = ROOT / "Figure" / "Figure_later" / "SOH_multicondition_sample_mapping"

FIG_W_CM = 7.92
FIG_H_CM = 4.50

# Ten cells from different working-condition groups. Labels encode Group and Cell.
SELECTED_CELLS = [
    "G1C1",
    "G3C1",
    "G6C2",
    "G7C3",
    "G26C2",
    "G34C2",
    "G42C4",
    "G49C1",
    "G50C3",
    "G59C1",
]

SAMPLE_WEEK_TAGS = ["0", "3", "5", "6", "7", "8", "9", "10", "15"]


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 0.75,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_soh_rows() -> Dict[str, List[Dict[str, float]]]:
    by_cell: Dict[str, List[Dict[str, float]]] = {}
    with SOH_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cell = row["cell"]
            if cell not in SELECTED_CELLS:
                continue
            item = {
                "rpt_idx": int(float(row["rpt_idx"])),
                "time_week": float(row["time_week"]),
                "soh": float(row["soh"]),
            }
            by_cell.setdefault(cell, []).append(item)

    for rows in by_cell.values():
        rows.sort(key=lambda item: item["time_week"])
        if max(item["soh"] for item in rows) <= 2.0:
            for item in rows:
                item["soh"] *= 100.0
    return by_cell


def load_cell_info() -> Dict[str, Dict[str, str]]:
    info: Dict[str, Dict[str, str]] = {}
    with FEATURE_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cell = row["cell"]
            if cell not in SELECTED_CELLS:
                continue
            info[cell] = {
                "group": row["group_num"],
                "release": row["release"],
                "chg": row["chg_c_rate"],
                "dchg": row["dchg_c_rate"],
                "eol70": row["lifetime_weeks_EOL70"],
            }
            for tag in SAMPLE_WEEK_TAGS:
                info[cell][f"week{tag}_rpt_idx"] = row.get(f"week{tag}_rpt_idx", "")
    return info


def sample_points_for_cell(
    rows: List[Dict[str, float]], info: Dict[str, str]
) -> Tuple[List[float], List[float]]:
    by_rpt = {int(row["rpt_idx"]): row for row in rows}
    points: List[Tuple[float, float]] = []
    seen = set()
    for tag in SAMPLE_WEEK_TAGS:
        raw_idx = info.get(f"week{tag}_rpt_idx", "")
        if raw_idx in ("", "nan", "None"):
            continue
        rpt_idx = int(float(raw_idx))
        if rpt_idx in seen or rpt_idx not in by_rpt:
            continue
        seen.add(rpt_idx)
        row = by_rpt[rpt_idx]
        points.append((row["time_week"], row["soh"]))
    points.sort()
    return [p[0] for p in points], [p[1] for p in points]


def eol70_crossing(rows: List[Dict[str, float]]) -> Tuple[float, float]:
    prev = rows[0]
    for row in rows[1:]:
        y0 = prev["soh"]
        y1 = row["soh"]
        if (y0 - 70.0) * (y1 - 70.0) <= 0 and y0 != y1:
            ratio = (70.0 - y0) / (y1 - y0)
            x = prev["time_week"] + ratio * (row["time_week"] - prev["time_week"])
            return x, 70.0
        prev = row
    closest = min(rows, key=lambda item: abs(item["soh"] - 70.0))
    return closest["time_week"], closest["soh"]


def style_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    ax.tick_params(axis="both", which="major", length=2.2, width=0.6, pad=1.5, labelsize=7)
    ax.tick_params(axis="both", which="minor", length=1.5, width=0.5)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    soh_rows = load_soh_rows()
    cell_info = load_cell_info()

    colors = [
        "#1f6f78",
        "#c23b22",
        "#2f5597",
        "#d18f00",
        "#5b8c3a",
        "#7a4ea3",
        "#a65e2e",
        "#3f7f93",
        "#8f3d56",
        "#4d4d4d",
    ]

    fig, ax = plt.subplots(figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54))
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.185, top=0.965)

    ax.axvspan(0, 15, color="#cfcfcf", alpha=0.22, lw=0, zorder=0)
    ax.axvline(15, color="#777777", lw=0.55, ls=(0, (2.5, 2.0)), zorder=1)
    ax.axhline(70, color="#555555", lw=0.65, ls=(0, (3, 2)), zorder=1)

    for color, cell in zip(colors, SELECTED_CELLS):
        rows = soh_rows[cell]
        x = [row["time_week"] for row in rows]
        y = [row["soh"] for row in rows]
        ax.plot(x, y, color=color, lw=0.82, alpha=0.82, label=cell, zorder=2)

        sx, sy = sample_points_for_cell(rows, cell_info[cell])
        ax.scatter(
            sx,
            sy,
            s=6.5,
            color="#1f1f1f",
            edgecolor="white",
            linewidth=0.20,
            zorder=5,
        )
        ex, ey = eol70_crossing(rows)
        ax.scatter(
            [ex],
            [ey],
            s=13,
            color="#c7c7c7",
            edgecolor="#4d4d4d",
            linewidth=0.28,
            zorder=6,
        )

    ax.text(59.8, 71.0, "EOL70", fontsize=6.2, color="#333333", ha="right", va="bottom")

    ax.set_xlabel("Time (weeks)")
    ax.set_ylabel("SOH (%)")
    ax.set_xlim(-1.5, 64.5)
    ax.set_ylim(57, 103.5)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    style_axes(ax)

    handles, labels = ax.get_legend_handles_labels()
    point_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="#1f1f1f",
        markeredgecolor="white",
        markeredgewidth=0.25,
        markersize=3.6,
        label="SOH points",
    )
    rul_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="#c7c7c7",
        markeredgecolor="#4d4d4d",
        markeredgewidth=0.28,
        markersize=3.8,
        label="RUL points",
    )
    handles.append(point_handle)
    labels.append("SOH points")
    handles.append(rul_handle)
    labels.append("RUL points")
    split_at = 5
    legend_left = ax.legend(
        handles[:split_at],
        labels[:split_at],
        loc="upper left",
        bbox_to_anchor=(0.565, 0.998),
        ncol=1,
        frameon=False,
        fontsize=6.0,
        handlelength=1.3,
        labelspacing=0.18,
        borderaxespad=0.0,
    )
    ax.add_artist(legend_left)
    ax.legend(
        handles[split_at:],
        labels[split_at:],
        loc="upper left",
        bbox_to_anchor=(0.755, 0.998),
        ncol=1,
        frameon=False,
        fontsize=6.0,
        handlelength=1.3,
        labelspacing=0.18,
        borderaxespad=0.0,
    )

    png_path = OUT_DIR / "SOH_multicondition_sample_mapping.png"
    pdf_path = OUT_DIR / "SOH_multicondition_sample_mapping.pdf"
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    print("Selected cells:")
    for cell in SELECTED_CELLS:
        info = cell_info[cell]
        rows = soh_rows[cell]
        print(
            f"{cell}: group {info['group']}, {info['release']}, "
            f"charge/discharge {info['chg']}/{info['dchg']} C, "
            f"EOL70 {float(info['eol70']):.2f} weeks, final SOH {rows[-1]['soh']:.2f}%"
        )
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
