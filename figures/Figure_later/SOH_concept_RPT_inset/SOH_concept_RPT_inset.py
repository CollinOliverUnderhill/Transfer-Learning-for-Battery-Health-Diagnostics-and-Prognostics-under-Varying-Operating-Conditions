#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MultipleLocator


ROOT = Path(r"E:\Datasets\IVAS")
CELL = "G3C1"
RELEASE = "Release 1.0"
MARK_RPT_IDX = 15

SOH_CSV = ROOT / "Data" / "Processing_Data" / "SOHest" / "rpt_samples_feature_soh.csv"
FEATURE_CSV = (
    ROOT
    / "week_based"
    / "Final"
    / "EOL70"
    / "features"
    / "feature_table_all_cells_multiweek_EOL70.csv"
)
RPT_JSON = ROOT / "Data" / "RPT_json" / RELEASE / f"{CELL}.json"
OUT_DIR = ROOT / "Figure" / "Figure_later" / "SOH_concept_RPT_inset"

FIG_W_CM = 7.92
FIG_H_CM = 4.50


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


def load_double_json(path: Path) -> Dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, str):
        obj = json.loads(obj)
    return obj


def load_eol70_week(cell: str) -> float:
    with FEATURE_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("cell") == cell:
                return float(row["lifetime_weeks_EOL70"])
    raise ValueError(f"Cell {cell!r} was not found in {FEATURE_CSV}")


def load_soh_series(cell: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with SOH_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("cell") != cell:
                continue
            rows.append(
                {
                    "rpt_idx": int(float(row["rpt_idx"])),
                    "time_week": float(row["time_week"]),
                    "soh": float(row["soh"]),
                    "capacity_ah": float(row["capacity_ah"]),
                }
            )

    if not rows:
        raise ValueError(f"No SOH rows found for {cell!r} in {SOH_CSV}")

    rows.sort(key=lambda item: item["time_week"])
    if max(item["soh"] for item in rows) <= 2.0:
        for item in rows:
            item["soh"] *= 100.0
    return rows


def pick_mark_point(rows: List[Dict[str, float]], rpt_idx: int) -> Dict[str, float]:
    for row in rows:
        if row["rpt_idx"] == rpt_idx:
            return row
    return min(rows, key=lambda item: abs(item["time_week"] - rows[-1]["time_week"] / 2.0))


def downsample_xy(x: List[float], y: List[float], max_points: int = 1400) -> Tuple[List[float], List[float]]:
    if len(x) <= max_points:
        return x, y
    step = max(1, len(x) // max_points)
    return x[::step], y[::step]


def rpt_curve(rpt: Dict, mode: str, rpt_idx: int) -> Tuple[List[float], List[float]]:
    qv = rpt[mode]
    q = [float(v) for v in qv["Q"][rpt_idx]]
    voltage = [float(v) for v in qv["V"][rpt_idx]]
    return downsample_xy(q, voltage)


def style_axes(ax, label_size: float = 8.0, tick_size: float = 7.0) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    ax.tick_params(axis="both", which="major", length=2.2, width=0.6, pad=1.5, labelsize=tick_size)
    ax.tick_params(axis="both", which="minor", length=1.5, width=0.5)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eol70_week = load_eol70_week(CELL)
    rows = load_soh_series(CELL)
    rpt = load_double_json(RPT_JSON)
    mark = pick_mark_point(rows, MARK_RPT_IDX)
    rpt_idx = int(mark["rpt_idx"])

    x = [row["time_week"] for row in rows]
    y = [row["soh"] for row in rows]

    fig, ax = plt.subplots(figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54))
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.185, top=0.910)

    main_color = "#1f6f78"
    mark_color = "#c23b22"

    ax.plot(x, y, color=main_color, lw=1.05, marker="o", ms=2.0, mfc="white", mec=main_color, mew=0.55)
    ax.scatter(
        [mark["time_week"]],
        [mark["soh"]],
        s=16,
        color=mark_color,
        edgecolor="white",
        linewidth=0.45,
        zorder=5,
    )

    ax.axhline(70, color="#555555", lw=0.65, ls=(0, (3, 2)))
    ax.axvline(eol70_week, color="#555555", lw=0.65, ls=(0, (3, 2)))
    ax.text(eol70_week - 0.65, 68.8, "EOL70", fontsize=6.5, color="#333333", ha="right", va="top")
    ax.text(
        mark["time_week"] - 0.65,
        mark["soh"] + 5.4,
        f"RPT #{rpt_idx}",
        fontsize=6.0,
        color=mark_color,
        ha="right",
        va="center",
    )

    ax.set_xlabel("Time (weeks)")
    ax.set_ylabel("SOH (%)")
    ax.text(0.035, 0.055, "Group 3, Cell 1", fontsize=6.5, color="#333333", transform=ax.transAxes)
    ax.set_xlim(-0.8, max(x) + 1.0)
    ax.set_ylim(57, 104)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    style_axes(ax)

    inset = ax.inset_axes([0.665, 0.730, 0.34, 0.31])
    q_chg, v_chg = rpt_curve(rpt, "QV_charge_C_5", rpt_idx)
    q_dchg, v_dchg = rpt_curve(rpt, "QV_discharge_C_5", rpt_idx)
    inset.plot(q_chg, v_chg, color="#d18f00", lw=0.75, label="Charge")
    inset.plot(q_dchg, v_dchg, color="#2f5597", lw=0.75, label="Discharge")
    inset.set_title("RPT Cycle", fontsize=5.8, pad=0.6)
    inset.set_xlabel("Capacity (Ah)", fontsize=5.0, labelpad=0.7)
    inset.set_ylabel("Voltage (V)", fontsize=5.0, labelpad=0.7)
    inset.tick_params(axis="both", which="major", labelsize=4.8, length=1.5, width=0.5, pad=0.8)
    for spine in inset.spines.values():
        spine.set_linewidth(0.6)
    inset.text(0.66, 0.72, "Charge", color="#d18f00", fontsize=4.8, transform=inset.transAxes)
    inset.text(0.55, 0.22, "Discharge", color="#2f5597", fontsize=4.8, transform=inset.transAxes)

    con = ConnectionPatch(
        xyA=(mark["time_week"], mark["soh"]),
        coordsA=ax.transData,
        xyB=(-0.06, -0.06),
        coordsB=inset.transAxes,
        arrowstyle="->",
        shrinkA=4,
        shrinkB=1,
        mutation_scale=8,
        lw=0.65,
        color=mark_color,
    )
    fig.add_artist(con)

    png_path = OUT_DIR / "SOH_concept_RPT_inset.png"
    pdf_path = OUT_DIR / "SOH_concept_RPT_inset.pdf"
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    print(f"Cell: {CELL}, {RELEASE}")
    print(f"RPT points: {len(rows)}, final week: {x[-1]:.2f}, final SOH: {y[-1]:.2f}%")
    print(f"EOL70 week: {eol70_week:.2f}")
    print(f"Marked RPT: #{rpt_idx}, week {mark['time_week']:.2f}, SOH {mark['soh']:.2f}%")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
