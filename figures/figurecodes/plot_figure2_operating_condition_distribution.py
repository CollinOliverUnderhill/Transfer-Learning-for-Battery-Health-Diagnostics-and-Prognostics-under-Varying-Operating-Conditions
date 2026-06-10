#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d import proj3d


IVAS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = IVAS_ROOT / "Groupcondi.csv"
DEFAULT_OUTPUT = IVAS_ROOT / "Figure" / "figure2" / "figure2.png"

TEXT_COLOR = "#111111"
GRID_COLOR = "#C7D0D4"
FRAME_COLOR = "#092431"
FIXED_MARKER_SIZE = 36.0
FIG_W_CM = 7.92
FIG_H_CM = 5.83
LABEL_OFFSET_OVERRIDES = {
    9: (0.090, 0.040, 3.30),
    16: (-0.110, -0.050, 2.60),
    20: (-0.065, 0.070, 3.10),
    21: (0.070, -0.075, 1.95),
    22: (-0.120, 0.030, 2.20),
    24: (0.105, 0.060, 3.55),
    36: (0.120, -0.015, 2.75),
    37: (-0.130, -0.020, 1.85),
    38: (0.105, -0.080, 2.10),
    39: (-0.085, 0.085, 2.90),
    5: (-0.085, 0.040, 2.45),
    64: (0.095, -0.035, 1.85),
    10: (-0.090, -0.055, 1.95),
    11: (0.090, 0.050, 2.55),
    12: (0.110, -0.045, 2.25),
    13: (-0.105, 0.070, 2.70),
    29: (0.100, -0.060, 2.25),
    33: (-0.105, 0.040, 2.65),
    56: (0.095, 0.075, 2.20),
    58: (-0.090, -0.060, 2.40),
}


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


def lifetime_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "pulsebat_lifetime",
        ["#4C78A8", "#3C9D9B", "#54A24B", "#F58518"],
        N=256,
    )


def colorbar_ticks(vmin: float, vmax: float) -> list[float]:
    step = (vmax - vmin) / 4.0
    return [vmin + step * i for i in range(5)]


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


def read_group_conditions(path: Path) -> list[dict[str, float]]:
    required = {
        "Group",
        "Charging C-rate",
        "Discharging C-rate",
        "Mean DoD",
        "Mean Lifetime [weeks]",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    out: list[dict[str, float]] = []
    for row in rows:
        group = int(parse_float(row["Group"]))
        point = {
            "group": group,
            "charging_crate": parse_float(row["Charging C-rate"]),
            "discharging_crate": parse_float(row["Discharging C-rate"]),
            "dod_pct": parse_float(row["Mean DoD"]),
            "lifetime_week": parse_float(row["Mean Lifetime [weeks]"]),
        }
        if all(is_finite(point[k]) for k in ("charging_crate", "discharging_crate", "dod_pct")):
            out.append(point)
    return sorted(out, key=lambda item: int(item["group"]))


def add_visual_separation(
    points: list[dict[str, float]],
    xy_radius: float,
    z_radius: float,
) -> list[dict[str, float]]:
    """Apply a small deterministic display-only offset to close points.

    The offset is intentionally small relative to the axis ranges. It reduces
    marker and label overlap in clusters without changing the source data.
    """
    buckets: dict[tuple[int, int, int], list[dict[str, float]]] = defaultdict(list)
    for point in points:
        key = (
            round(float(point["discharging_crate"]) / 0.35),
            round(float(point["charging_crate"]) / 0.35),
            round(float(point["dod_pct"]) / 8.0),
        )
        buckets[key].append(point)

    separated: list[dict[str, float]] = []
    for bucket_points in buckets.values():
        n = len(bucket_points)
        for idx, point in enumerate(sorted(bucket_points, key=lambda item: int(item["group"]))):
            item = dict(point)
            if n > 1 and xy_radius > 0:
                angle = 2.0 * math.pi * idx / n + (math.pi / 11.0) * (n % 3)
                ring = 0.68 + 0.28 * (idx % 3)
                item["plot_discharging_crate"] = float(item["discharging_crate"]) + xy_radius * ring * math.cos(angle)
                item["plot_charging_crate"] = float(item["charging_crate"]) + xy_radius * ring * math.sin(angle)
                item["plot_dod_pct"] = float(item["dod_pct"]) + z_radius * math.sin(angle + math.pi / 5.0)
            else:
                item["plot_discharging_crate"] = float(item["discharging_crate"])
                item["plot_charging_crate"] = float(item["charging_crate"])
                item["plot_dod_pct"] = float(item["dod_pct"])
            separated.append(item)
    return sorted(separated, key=lambda item: int(item["group"]))


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


def label_candidate_offsets() -> list[tuple[float, float]]:
    offsets = [(0.010, 0.010)]
    for radius in (0.020, 0.038, 0.058, 0.082, 0.108, 0.138, 0.170):
        for angle_deg in (10, 42, 75, 112, 148, 185, 220, 255, 292, 328):
            angle = math.radians(angle_deg)
            offsets.append((radius * math.cos(angle), radius * math.sin(angle)))
    return offsets


def projected_axes_position(fig, ax, point: dict[str, float]) -> tuple[float, float]:
    x_proj, y_proj, _ = proj3d.proj_transform(
        float(point["plot_discharging_crate"]),
        float(point["plot_charging_crate"]),
        float(point["plot_dod_pct"]),
        ax.get_proj(),
    )
    x_display, y_display = ax.transData.transform((x_proj, y_proj))
    x_axes, y_axes = ax.transAxes.inverted().transform((x_display, y_display))
    return float(x_axes), float(y_axes)


def rect_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left = max(a[0], b[0])
    right = min(a[2], b[2])
    bottom = max(a[1], b[1])
    top = min(a[3], b[3])
    if right <= left or top <= bottom:
        return 0.0
    return (right - left) * (top - bottom)


def add_projected_group_labels(fig, ax, points: list[dict[str, float]]) -> None:
    fig.canvas.draw()
    placed: list[tuple[float, float, float, float]] = []
    candidates = label_candidate_offsets()

    label_w = 0.046
    label_h = 0.027
    ordered = sorted(
        points,
        key=lambda item: (
            -float(item["plot_dod_pct"]),
            float(item["plot_discharging_crate"]),
            int(item["group"]),
        ),
    )

    for point in ordered:
        group = int(point["group"])
        base_x, base_y = projected_axes_position(fig, ax, point)

        best_score = float("inf")
        best_xy = (base_x + 0.010, base_y + 0.010)
        for dx, dy in candidates:
            x = base_x + dx
            y = base_y + dy
            rect = (x - 0.002, y - 0.003, x + label_w, y + label_h)
            overlap = sum(rect_overlap(rect, old) for old in placed)
            outside = (
                max(0.0, -rect[0])
                + max(0.0, rect[2] - 1.0)
                + max(0.0, -rect[1])
                + max(0.0, rect[3] - 1.0)
            )
            score = overlap * 7200.0 + outside * 55.0 + (dx * dx + dy * dy) * 2.0
            if score < best_score:
                best_score = score
                best_xy = (x, y)

        x, y = best_xy
        placed.append((x - 0.002, y - 0.003, x + label_w, y + label_h))
        text = ax.text2D(
            x,
            y,
            f"G{group}",
            transform=ax.transAxes,
            fontsize=3.7,
            color=TEXT_COLOR,
            zorder=20,
            clip_on=False,
        )
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=1.10, foreground="white", alpha=0.94),
                path_effects.Normal(),
            ]
        )


def plot_figure(points: list[dict[str, float]], out_path: Path, annotate: bool) -> None:
    setup_style()

    known = [p for p in points if is_finite(float(p["lifetime_week"]))]
    if not known:
        raise ValueError("No groups with finite lifetime are available for plotting.")

    finite_life = [float(p["lifetime_week"]) for p in known]
    vmin, vmax = min(finite_life), max(finite_life)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = lifetime_cmap()

    fig = plt.figure(figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54))
    ax = fig.add_axes([-0.050, 0.110, 0.875, 0.900], projection="3d")
    ax.set_box_aspect((1.25, 1.05, 1.14))

    sc = ax.scatter(
        [float(p["plot_discharging_crate"]) for p in known],
        [float(p["plot_charging_crate"]) for p in known],
        [float(p["plot_dod_pct"]) for p in known],
        c=[float(p["lifetime_week"]) for p in known],
        s=FIXED_MARKER_SIZE,
        cmap=cmap,
        norm=norm,
        marker="o",
        alpha=0.90,
        edgecolors="#3C3C3C",
        linewidths=0.28,
        depthshade=False,
        rasterized=True,
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
    ax.view_init(elev=22, azim=-43)
    style_3d_axis(ax)
    if annotate:
        add_projected_group_labels(fig, ax, known)

    cax = fig.add_axes([0.840, 0.245, 0.028, 0.565])
    cbar = fig.colorbar(sc, cax=cax)
    ticks = colorbar_ticks(vmin, vmax)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in ticks])
    cbar.set_label("Mean lifetime (weeks)", fontsize=8, labelpad=2)
    cbar.ax.tick_params(labelsize=7, length=2.0, width=0.6, pad=1.2)
    cbar.outline.set_linewidth(0.75)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw Fig. 2: operating-condition distribution of all IVAS battery groups."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--annotate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--visual_xy_separation",
        type=float,
        default=0.062,
        help="Small display-only C-rate offset for visually separating nearby groups.",
    )
    parser.add_argument(
        "--visual_z_separation",
        type=float,
        default=1.70,
        help="Small display-only DoD offset for visually separating nearby groups.",
    )
    args = parser.parse_args()

    points = read_group_conditions(args.input)
    if not points:
        raise ValueError(f"No plottable groups found in {args.input}")
    separated = add_visual_separation(points, args.visual_xy_separation, args.visual_z_separation)
    plot_figure(separated, args.output, annotate=bool(args.annotate))
    print(f"[ok] wrote {args.output}")
    print(f"[ok] wrote {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
