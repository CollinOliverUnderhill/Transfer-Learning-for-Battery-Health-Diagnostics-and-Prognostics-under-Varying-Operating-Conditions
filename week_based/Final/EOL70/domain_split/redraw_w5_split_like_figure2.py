#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


ROOT = Path(r"E:\Datasets\IVAS")
SPLIT_DIR = ROOT / "week_based" / "Final" / "EOL70" / "domain_split"
SPLIT_CSV = SPLIT_DIR / "cell_split_targetspread_w5_EOL70.csv"
GROUP_CONDI = ROOT / "Groupcondi.csv"
OUT_PNG = SPLIT_DIR / "plot_condition_split_selection_3d_cell_jitter_w5_EOL70.png"
FIG10_PNG = ROOT / "Figure" / "figure10" / "plot_condition_split_selection_3d_cell_jitter_w5_EOL70.png"

GRID_COLOR = "#C7D0D4"
FRAME_COLOR = "#092431"
FIG_W_CM = 7.92
FIG_H_CM = 5.83


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.linewidth": 0.75,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def parse_float(value: object) -> float:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return float("nan")
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        return float(text)
    except ValueError:
        return float("nan")


def is_finite(value: float) -> bool:
    return math.isfinite(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_group_conditions(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for row in read_csv(path):
        group = int(parse_float(row["Group"]))
        chg = parse_float(row["Charging C-rate"])
        dchg = parse_float(row["Discharging C-rate"])
        dod = parse_float(row["Mean DoD"])
        if all(is_finite(v) for v in (chg, dchg, dod)):
            out[group] = {
                "charging_crate": chg,
                "discharging_crate": dchg,
                "dod_pct": dod,
            }
    return out


def cell_jitter(cell_idx: int, radius: float) -> tuple[float, float, float]:
    pattern = {
        1: (-1.0, -1.0, -0.4),
        2: (1.0, -1.0, 0.4),
        3: (-1.0, 1.0, 0.4),
        4: (1.0, 1.0, -0.4),
    }
    dx, dy, dz = pattern.get(cell_idx, (0.0, 0.0, 0.0))
    return dx * radius, dy * radius, dz * radius * 12.0


def build_points() -> tuple[list[dict[str, object]], dict[str, int]]:
    cond_by_group = load_group_conditions(GROUP_CONDI)
    points: list[dict[str, object]] = []
    counts = {"train": 0, "fine_tune": 0, "test": 0}
    for row in read_csv(SPLIT_CSV):
        split = str(row.get("split", "")).strip()
        if split not in counts:
            continue
        group = int(parse_float(row["group_num"]))
        cell_idx = int(parse_float(row["cell_idx"]))
        cond = cond_by_group.get(group)
        if not cond:
            continue
        dx, dy, dz = cell_jitter(cell_idx, 0.030)
        points.append(
            {
                "split": split,
                "x": cond["discharging_crate"] + dx,
                "y": cond["charging_crate"] + dy,
                "z": cond["dod_pct"] + dz,
            }
        )
        counts[split] += 1
    return points, counts


def style_3d_axis(ax) -> None:
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor(GRID_COLOR)
        axis._axinfo["grid"]["color"] = (0.84, 0.87, 0.89, 0.70)
        axis._axinfo["grid"]["linewidth"] = 0.42
        axis._axinfo["axisline"]["color"] = FRAME_COLOR
        axis._axinfo["axisline"]["linewidth"] = 0.75
    ax.tick_params(axis="both", which="major", labelsize=7, width=0.6, pad=-2)
    ax.zaxis.set_tick_params(labelsize=7, width=0.6, pad=-1)


def plot() -> None:
    setup_style()
    points, counts = build_points()
    if not points:
        raise ValueError(f"No plottable points found in {SPLIT_CSV}")

    colors = {
        "train": "#F58518",
        "fine_tune": "#54A24B",
        "test": "#4C78A8",
    }
    markers = {
        "train": "o",
        "fine_tune": "^",
        "test": "s",
    }
    labels = {
        "train": "Train",
        "fine_tune": "Fine-tune",
        "test": "Test",
    }
    sizes = {
        "train": 15.0,
        "fine_tune": 20.0,
        "test": 20.0,
    }

    fig = plt.figure(figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54))
    ax = fig.add_axes([-0.080, 0.110, 0.790, 0.900], projection="3d")
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.25, 1.05, 1.14))

    for split in ("train", "fine_tune", "test"):
        sub = [p for p in points if p["split"] == split]
        if not sub:
            continue
        ax.scatter(
            [float(p["x"]) for p in sub],
            [float(p["y"]) for p in sub],
            [float(p["z"]) for p in sub],
            s=sizes[split],
            c=colors[split],
            marker=markers[split],
            alpha=0.86,
            edgecolors="#3C3C3C",
            linewidths=0.24,
            depthshade=False,
            rasterized=True,
            label=labels[split],
        )

    ax.set_xlabel("Discharging C-rate", fontsize=8, labelpad=-6)
    ax.set_ylabel("Charging C-rate", fontsize=8, labelpad=-6)
    ax.set_zlabel("Mean DoD (%)", fontsize=8, labelpad=-4)
    ax.set_xlim(0.35, 3.15)
    ax.set_ylim(0.35, 3.15)
    ax.set_zlim(0, 104)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_zticks([0, 20, 40, 60, 80, 100])
    ax.view_init(elev=30, azim=-45)
    style_3d_axis(ax)

    handles = [
        mlines.Line2D(
            [],
            [],
            linestyle="None",
            marker=markers[split],
            markersize=4.6,
            markerfacecolor=colors[split],
            markeredgecolor="#3C3C3C",
            markeredgewidth=0.45,
            label=labels[split],
        )
        for split in ("train", "fine_tune", "test")
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.420, 0.305),
        frameon=False,
        borderpad=0.0,
        handletextpad=0.35,
        labelspacing=0.36,
        fontsize=5.2,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(0.95)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, facecolor="white")
    fig.savefig(OUT_PNG.with_suffix(".pdf"), facecolor="white")
    FIG10_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG10_PNG, dpi=600, facecolor="white")
    fig.savefig(FIG10_PNG.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)

    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PNG.with_suffix('.pdf')}")
    print(f"Saved {FIG10_PNG}")
    print(f"Saved {FIG10_PNG.with_suffix('.pdf')}")
    print(
        "Counts: "
        + ", ".join(f"{split}={counts[split]}" for split in ("train", "fine_tune", "test"))
    )


if __name__ == "__main__":
    plot()
