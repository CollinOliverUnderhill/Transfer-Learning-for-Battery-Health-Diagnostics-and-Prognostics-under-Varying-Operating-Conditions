#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


ROOT = Path(r"E:\Datasets\IVAS")
OUT_DIR = ROOT / "PPT_Figure" / "p6_figure"
SOH_CSV = ROOT / "Data" / "Processing_Data" / "SOHest" / "rpt_samples_feature_soh.csv"
COND_CSV = ROOT / "Groupcondi.csv"

FIG_W_CM = 7.92
FIG_H_CM = 4.50

COMPARISONS = [
    {
        "key": "charge_current",
        "title": "Charge current effect",
        "groups": [2, 13],
        "labels": ["Low charge current", "High charge current"],
        "note": "Dchg = 0.5 C; DoD is similar",
    },
    {
        "key": "discharge_current",
        "title": "Discharge current effect",
        "groups": [10, 47],
        "labels": ["Low discharge current", "High discharge current"],
        "note": "Charge rate and DoD are close",
    },
    {
        "key": "depth_of_discharge",
        "title": "Depth of discharge effect",
        "groups": [1, 16],
        "labels": ["Low DoD", "High DoD"],
        "note": "Chg = 0.5 C; Dchg = 0.5 C",
    },
]


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 0.75,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def group_num_from_cell(cell: str) -> int:
    head = cell.split("C", 1)[0]
    return int(head.replace("G", ""))


def parse_pct(text: str) -> float:
    return float(text.strip().replace("%", ""))


def load_conditions() -> Dict[int, Dict[str, float]]:
    conditions: Dict[int, Dict[str, float]] = {}
    with COND_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            group = int(row["Group"])
            conditions[group] = {
                "charge": float(row["Charging C-rate"]),
                "discharge": float(row["Discharging C-rate"]),
                "dod": parse_pct(row["Mean DoD"]),
                "life": float(row["Mean Lifetime [weeks]"]) if row["Mean Lifetime [weeks]"] else np.nan,
            }
    return conditions


def load_soh_by_group() -> Dict[int, Dict[str, List[Tuple[float, float]]]]:
    by_group: Dict[int, Dict[str, List[Tuple[float, float]]]] = {}
    with SOH_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cell = row["cell"]
            group = group_num_from_cell(cell)
            soh = float(row["soh"])
            if soh <= 2.0:
                soh *= 100.0
            by_group.setdefault(group, {}).setdefault(cell, []).append(
                (float(row["time_week"]), soh)
            )
    for cells in by_group.values():
        for points in cells.values():
            points.sort(key=lambda item: item[0])
    return by_group


def eol70_crossing(points: List[Tuple[float, float]]) -> float:
    prev_x, prev_y = points[0]
    for x, y in points[1:]:
        if (prev_y - 70.0) * (y - 70.0) <= 0 and prev_y != y:
            ratio = (70.0 - prev_y) / (y - prev_y)
            return prev_x + ratio * (x - prev_x)
        prev_x, prev_y = x, y
    return points[-1][0]


def representative_cell(
    group_cells: Dict[str, List[Tuple[float, float]]],
    target_life: float,
) -> Tuple[str, List[Tuple[float, float]], float, int]:
    candidates = []
    for cell, points in group_cells.items():
        if len(points) < 2:
            continue
        life = eol70_crossing(points)
        candidates.append((abs(life - target_life), cell, points, life))
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, cell, points, life = candidates[0]
    return cell, points, life, len(candidates)


def style_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    ax.tick_params(axis="both", which="major", length=2.2, width=0.6, pad=1.5, labelsize=7)
    ax.tick_params(axis="both", which="minor", length=1.5, width=0.5)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)


def label_for_group(
    group: int,
    base_label: str,
    cond: Dict[int, Dict[str, float]],
    cell: str,
) -> str:
    return str(base_label)


def plot_one(
    spec: Dict[str, object],
    cond: Dict[int, Dict[str, float]],
    soh_by_group: Dict[int, Dict[str, List[Tuple[float, float]]]],
) -> None:
    groups = list(spec["groups"])
    base_labels = list(spec["labels"])
    colors = ["#1f6f78", "#c23b22"]
    fig, ax = plt.subplots(figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54))
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.205, top=0.865)

    ax.axhline(70, color="#555555", lw=0.65, ls=(0, (3, 2)), zorder=1)
    ax.axvspan(0, 15, color="#cfcfcf", alpha=0.16, lw=0, zorder=0)
    ax.axvline(15, color="#777777", lw=0.55, ls=(0, (2.5, 2.0)), zorder=1)

    summary_rows = []
    for color, group, base_label in zip(colors, groups, base_labels):
        c = cond[int(group)]
        cell, points, cell_life, n = representative_cell(soh_by_group[int(group)], c["life"])
        x = np.asarray([p[0] for p in points], dtype=float)
        y = np.asarray([p[1] for p in points], dtype=float)
        ax.plot(
            x,
            y,
            color=color,
            lw=1.15,
            alpha=0.95,
            label=label_for_group(int(group), str(base_label), cond, cell),
            zorder=3,
        )
        step = max(1, len(x) // 8)
        ax.scatter(x[::step], y[::step], s=7.0, color=color, edgecolor="white", linewidth=0.25, zorder=4)
        summary_rows.append(
            {
                "comparison": spec["key"],
                "group": int(group),
                "label": base_label,
                "cell": cell,
                "charge_c": c["charge"],
                "discharge_c": c["discharge"],
                "dod_pct": c["dod"],
                "mean_lifetime_weeks": c["life"],
                "cell_eol70_weeks": cell_life,
                "n_cells": n,
            }
        )

    ax.text(63.5, 71.0, "EOL70", fontsize=6.2, color="#333333", ha="right", va="bottom")
    ax.set_title(str(spec["title"]), fontsize=8.6, pad=3.5)
    ax.set_xlabel("Time (weeks)")
    ax.set_ylabel("SOH (%)")
    ax.set_xlim(-1.5, 64.5)
    ax.set_ylim(57, 103.5)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    style_axes(ax)

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=5.8,
        handlelength=1.45,
        labelspacing=0.24,
        borderaxespad=0.0,
    )
    ax.text(
        0.015,
        0.035,
        str(spec["note"]),
        transform=ax.transAxes,
        fontsize=5.7,
        color="#555555",
        ha="left",
        va="bottom",
    )

    for suffix in ["png", "pdf", "svg"]:
        out = OUT_DIR / f"p6_soh_{spec['key']}.{suffix}"
        fig.savefig(out, dpi=600 if suffix == "png" else None, facecolor="white")
    plt.close(fig)
    return summary_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cond = load_conditions()
    soh_by_group = load_soh_by_group()

    all_rows = []
    for spec in COMPARISONS:
        all_rows.extend(plot_one(spec, cond, soh_by_group))

    summary_csv = OUT_DIR / "p6_condition_pairs_summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "comparison",
            "group",
            "label",
            "cell",
            "charge_c",
            "discharge_c",
            "dod_pct",
            "mean_lifetime_weeks",
            "cell_eol70_weeks",
            "n_cells",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print("Condition pairs:")
    for row in all_rows:
        print(
            f"{row['comparison']}: G{row['group']} {row['label']} {row['cell']} | "
            f"charge {row['charge_c']:.3g} C, discharge {row['discharge_c']:.3g} C, "
            f"DoD {row['dod_pct']:.1f}%, life {row['mean_lifetime_weeks']:.2f} weeks, "
            f"cell EOL70 {row['cell_eol70_weeks']:.2f} weeks, candidates n={row['n_cells']}"
        )
    print(f"Saved summary: {summary_csv}")


if __name__ == "__main__":
    main()
