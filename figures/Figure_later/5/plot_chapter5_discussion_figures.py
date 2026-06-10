#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Chapter 5 discussion figures in the PulseBat-combo style."""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[2]
CODE_DIR = ROOT / "Figure" / "figurecodes"
sys.path.insert(0, str(CODE_DIR))

from pulsebat_iv_common import (  # noqa: E402
    CM,
    COLORS,
    DPI,
    ROOT,
    WEEK,
    finite,
    group_num_from_cell,
    read_csv,
    save_figure,
    setup_style,
    style_axes,
    to_float,
)


OUT_DIR = ROOT / "Figure" / "Figure_later" / "5"
MAIN_STAGE = (
    WEEK / "Final" / "EOL70" / "3step" / "outputs_400"
    / "BasicModel" / "stage3_final_rerun_400"
)
WEEK_AVAIL = WEEK / "Final" / "EOL70" / "features" / "week_availability_summary_EOL70.csv"
LOWCAP_STAGE = (
    WEEK / "Final" / "EOL70" / "3step" / "outputs_lowcapacity"
    / "lowcapacity_grid" / "stage3_final"
)
SEED_ROOT = (
    WEEK / "Final" / "EOL70" / "3step" / "outputs_400"
    / "random_w5_EOL70_10seeds_legacy400"
)
SEED_SUMMARY = SEED_ROOT / "random_w5_EOL70_10seeds_legacy400_benchmark_transfer_summary_all_splits.csv"

MARKERS = {
    "Source": ("o", "#9E9E9E"),
    "Fine-tune": ("^", COLORS["fine_tune"]),
    "Test": ("s", COLORS["test"]),
}


def parse_group_condition() -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for row in read_csv(ROOT / "Groupcondi.csv"):
        group = int(to_float(row.get("Group")))
        out[group] = {
            "charge": to_float(row.get("Charging C-rate")),
            "discharge": to_float(row.get("Discharging C-rate")),
            "dod": to_float(row.get("Mean DoD")),
            "life": to_float(row.get("Mean Lifetime [weeks]")),
        }
    return out


def sample_groups(path: Path) -> set[int]:
    groups: set[int] = set()
    for row in read_csv(path):
        group = to_float(row.get("group_num"))
        if finite(group):
            groups.add(int(group))
            continue
        group_num = group_num_from_cell(row.get("cell", ""))
        if group_num is not None:
            groups.add(group_num)
    return groups


def sample_group_sequence(path: Path) -> list[int]:
    groups: list[int] = []
    for row in read_csv(path):
        group = to_float(row.get("group_num"))
        if finite(group):
            groups.append(int(group))
            continue
        group_num = group_num_from_cell(row.get("cell", ""))
        if group_num is not None:
            groups.append(group_num)
    return groups


def metric(path: Path, key: str) -> float:
    rows = read_csv(path)
    return to_float(rows[0].get(key)) if rows else float("nan")


def improve_percent(bench: float, transfer: float) -> float:
    return (bench - transfer) / bench * 100.0


def format_distance(value: float) -> str:
    if abs(value) < 0.0005:
        return "0"
    return f"{value:.2f}"


def group_coords(
    groups: Iterable[int],
    cond: dict[int, dict[str, float]],
    jitter_key: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, zs = [], [], []
    for group in list(groups):
        if group not in cond:
            continue
        xs.append(cond[group]["charge"])
        ys.append(cond[group]["discharge"])
        zs.append(cond[group]["dod"])
    x = np.asarray(xs)
    y = np.asarray(ys)
    z = np.asarray(zs)
    if len(x) == 0:
        return x, y, z
    rng = np.random.default_rng(31017 + jitter_key)
    x = np.clip(x + rng.normal(0, 0.030, len(x)), 0.35, 3.15)
    y = np.clip(y + rng.normal(0, 0.030, len(y)), 0.35, 3.15)
    z = np.clip(z + rng.normal(0, 1.25, len(z)), 0.0, 105.0)
    return x, y, z


def draw_condition_space(
    ax,
    cond,
    split_groups: list[tuple[str, Iterable[int]]],
    title: str = "",
) -> None:
    keys = {"Source": 1, "Fine-tune": 2, "Test": 3}
    for label, groups in split_groups:
        marker, color = MARKERS[label]
        x, y, z = group_coords(groups, cond, keys[label])
        ax.scatter(
            x,
            y,
            z,
            s=13.0 if label == "Source" else 22.0,
            marker=marker,
            facecolor=color,
            edgecolor=COLORS["black"] if label != "Source" else "#666666",
            linewidth=0.35,
            alpha=0.92 if label != "Source" else 0.70,
            label=label,
            zorder=3 if label != "Source" else 1,
        )
    ax.set_xlabel("Charge C-rate", labelpad=-6, fontsize=6)
    ax.set_ylabel("Discharge C-rate", labelpad=-6, fontsize=6)
    ax.set_zlabel("Mean DoD (%)", labelpad=-7, fontsize=6)
    ax.set_xlim(0.35, 3.15)
    ax.set_ylim(0.35, 3.15)
    ax.set_zlim(0, 105)
    ax.set_xticks([0.5, 1.0, 2.0, 3.0])
    ax.set_yticks([0.5, 1.0, 2.0, 3.0])
    ax.set_zticks([20, 50, 80, 100])
    ax.tick_params(axis="both", which="major", pad=-4, labelsize=5.6)
    ax.tick_params(axis="z", which="major", pad=-4, labelsize=5.6)
    ax.view_init(elev=30, azim=-45)
    try:
        ax.set_proj_type("ortho")
        ax.set_box_aspect((1.0, 1.0, 0.88))
    except AttributeError:
        pass
    ax.grid(True, color=COLORS["grid"], lw=0.35)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor(COLORS["grid"])
    if title:
        ax.set_title(title, fontsize=7, pad=2.0)


def plot_main_condition_split() -> None:
    cond = parse_group_condition()
    splits = [
        ("Source", sample_group_sequence(MAIN_STAGE / "source_train_samples.csv")),
        ("Fine-tune", sample_group_sequence(MAIN_STAGE / "target_finetune_samples.csv")),
        ("Test", sample_group_sequence(MAIN_STAGE / "target_test_samples.csv")),
    ]
    fig = plt.figure(figsize=(10.0 * CM, 7.00 * CM), dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.010, right=0.965, bottom=0.095, top=0.965)
    draw_condition_space(ax, cond, splits)
    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=1.0,
        handletextpad=0.45,
        labelspacing=0.25,
        borderaxespad=0.2,
    )
    save_figure(fig, OUT_DIR, "Ch5_Fig1_main_condition_split")
    plt.close(fig)


def plot_week_availability() -> None:
    weeks = [3, 5, 6, 7, 8, 9, 10]
    labels, values = [], []
    for week in weeks:
        split_path = WEEK / "Final" / "EOL70" / "domain_split" / f"cell_split_targetspread_w{week}_EOL70.csv"
        rows = read_csv(split_path)
        labels.append(f"w{week}")
        values.append(float(len({row.get("cell") for row in rows if row.get("cell")})))
    x = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(8.72 * CM, 4.20 * CM), dpi=DPI)
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.210, top=0.920)
    ax.bar(x, values, width=0.62, color=COLORS["source"], edgecolor=COLORS["black"], linewidth=0.45, zorder=2)
    for xi, yi in zip(x, values):
        ax.text(xi, yi + max(values) * 0.020, f"{yi:.0f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Feature week", labelpad=2)
    ax.set_ylabel("Available cells", labelpad=2)
    ax.set_ylim(0, max(values) * 1.14)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(True, axis="y", color=COLORS["grid"], lw=0.35, zorder=1)
    style_axes(ax)
    save_figure(fig, OUT_DIR, "Ch5_Fig2_week_availability")
    plt.close(fig)


def plot_lowcapacity_domain_effect() -> None:
    cond = parse_group_condition()
    splits = [
        ("Source", sample_group_sequence(LOWCAP_STAGE / "source_train_samples.csv")),
        ("Fine-tune", sample_group_sequence(LOWCAP_STAGE / "target_finetune_samples.csv")),
        ("Test", sample_group_sequence(LOWCAP_STAGE / "target_test_samples.csv")),
    ]
    bm_mae = metric(LOWCAP_STAGE / "benchmark" / "test_overall_metrics.csv", "mae")
    tl_mae = metric(LOWCAP_STAGE / "transfer_model" / "test_overall_metrics.csv", "mae")
    delta = improve_percent(bm_mae, tl_mae)

    fig = plt.figure(figsize=(13.6 * CM, 5.90 * CM), dpi=DPI)
    axes = [fig.add_subplot(1, 2, 1, projection="3d"), fig.add_subplot(1, 2, 2)]
    fig.subplots_adjust(left=0.020, right=0.985, bottom=0.155, top=0.860, wspace=0.30)
    draw_condition_space(axes[0], cond, splits, "Low-capacity split")
    axes[0].legend(
        loc="upper right",
        frameon=False,
        handlelength=1.0,
        handletextpad=0.40,
        labelspacing=0.22,
        borderaxespad=0.1,
    )

    x = np.arange(2, dtype=float)
    vals = [bm_mae, tl_mae]
    axes[1].bar(
        x,
        vals,
        width=0.55,
        color=[COLORS["benchmark"], COLORS["transfer"]],
        edgecolor=COLORS["black"],
        linewidth=0.45,
        zorder=2,
    )
    for xi, yi in zip(x, vals):
        axes[1].text(xi, yi + max(vals) * 0.035, f"{yi:.2f}", ha="center", va="bottom", fontsize=6.5)
    axes[1].text(0.50, max(vals) * 1.12, f"Delta MAE = {delta:+.1f}%", ha="center", va="bottom", fontsize=6.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["BM", "TL"])
    axes[1].set_ylabel("Target-test MAE (weeks)", labelpad=2)
    axes[1].set_ylim(0, max(vals) * 1.28)
    axes[1].grid(True, axis="y", color=COLORS["grid"], lw=0.35, zorder=1)
    axes[1].set_title("Transfer outcome", fontsize=7, pad=2.0)
    style_axes(axes[1])
    save_figure(fig, OUT_DIR, "Ch5_Fig3_lowcapacity_domain_effect")
    plt.close(fig)


def norm_condition(group: int, cond: dict[int, dict[str, float]]) -> np.ndarray:
    row = cond[group]
    return np.asarray([row["charge"] / 3.0, row["discharge"] / 3.0, row["dod"] / 100.0], dtype=float)


def spread_score(groups: set[int], cond: dict[int, dict[str, float]]) -> float:
    vals = [norm_condition(g, cond) for g in sorted(groups) if g in cond]
    if len(vals) < 2:
        return float("nan")
    dists = [float(np.linalg.norm(a - b)) for a, b in combinations(vals, 2)]
    return float(np.mean(dists))


def ft_test_distance(ft_groups: set[int], test_groups: set[int], cond: dict[int, dict[str, float]]) -> float:
    vals = []
    ft_vectors = [norm_condition(g, cond) for g in sorted(ft_groups) if g in cond]
    if not ft_vectors:
        return float("nan")
    for group in sorted(test_groups):
        if group not in cond:
            continue
        test_vec = norm_condition(group, cond)
        vals.append(min(float(np.linalg.norm(test_vec - ft_vec)) for ft_vec in ft_vectors))
    return float(np.mean(vals)) if vals else float("nan")


def seed_record(stage_name: str, stage: Path, summary_row: dict[str, str] | None, cond) -> dict[str, object]:
    ft_groups = sample_groups(stage / "target_finetune_samples.csv")
    test_groups = sample_groups(stage / "target_test_samples.csv")
    ft_plot_groups = sample_group_sequence(stage / "target_finetune_samples.csv")
    test_plot_groups = sample_group_sequence(stage / "target_test_samples.csv")
    if summary_row is None:
        bm_mae = metric(stage / "benchmark" / "test_overall_metrics.csv", "mae")
        tl_mae = metric(stage / "transfer_model" / "test_overall_metrics.csv", "mae")
        delta = improve_percent(bm_mae, tl_mae)
    else:
        bm_mae = to_float(summary_row.get("bench_test_mae"))
        tl_mae = to_float(summary_row.get("transfer_test_mae"))
        delta = to_float(summary_row.get("test_transfer_vs_bench_mae_improve_percent"))
    return {
        "name": stage_name,
        "stage": stage,
        "ft_groups": ft_groups,
        "test_groups": test_groups,
        "ft_plot_groups": ft_plot_groups,
        "test_plot_groups": test_plot_groups,
        "spread": spread_score(ft_groups, cond),
        "distance": ft_test_distance(ft_groups, test_groups, cond),
        "bm_mae": bm_mae,
        "tl_mae": tl_mae,
        "delta": delta,
    }


def representative_seed_records() -> list[dict[str, object]]:
    cond = parse_group_condition()
    records = [
        seed_record("rerun400", MAIN_STAGE, None, cond),
    ]
    for row in read_csv(SEED_SUMMARY):
        name = row["stage3_dir"].split("/")[0]
        records.append(seed_record(name, SEED_ROOT / name / "stage3_final", row, cond))
    by_name = {str(record["name"]): record for record in records}
    selected = [
        ("High cov.", "seed016"),
        ("Medium cov.", "seed006"),
        ("Low cov.", "seed007"),
    ]
    return [{"level": level, **by_name[name]} for level, name in selected]


def all_seed_records() -> list[dict[str, object]]:
    cond = parse_group_condition()
    records = [
        seed_record("rerun400", MAIN_STAGE, None, cond),
    ]
    for row in read_csv(SEED_SUMMARY):
        name = row["stage3_dir"].split("/")[0]
        records.append(seed_record(name, SEED_ROOT / name / "stage3_final", row, cond))
    return sorted([r for r in records if finite(float(r["distance"]))], key=lambda item: float(item["distance"]))


def plot_selection_examples() -> None:
    cond = parse_group_condition()
    reps = representative_seed_records()
    source_groups = sample_group_sequence(MAIN_STAGE / "source_train_samples.csv")

    fig = plt.figure(figsize=(17.4 * CM, 9.40 * CM), dpi=DPI)
    axes_top = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(3)]
    axes_bottom = [fig.add_subplot(2, 3, i + 4) for i in range(3)]
    fig.subplots_adjust(left=0.075, right=0.990, bottom=0.105, top=0.905, wspace=0.34, hspace=0.36)

    for col, rec in enumerate(reps):
        ax = axes_top[col]
        title = f"{rec['level']}: {rec['name']}"
        draw_condition_space(
            ax,
            cond,
            [
                ("Source", source_groups),
                ("Fine-tune", rec["ft_plot_groups"]),
                ("Test", rec["test_plot_groups"]),
            ],
            title,
        )
        bx = axes_bottom[col]
        vals = [float(rec["bm_mae"]), float(rec["tl_mae"])]
        x = np.arange(2, dtype=float)
        bx.bar(
            x,
            vals,
            width=0.55,
            color=[COLORS["benchmark"], COLORS["transfer"]],
            edgecolor=COLORS["black"],
            linewidth=0.45,
            zorder=2,
        )
        for xi, yi in zip(x, vals):
            bx.text(xi, yi + max(vals) * 0.035, f"{yi:.2f}", ha="center", va="bottom", fontsize=6.2)
        bx.text(0.50, max(vals) * 1.13, f"Delta MAE = {float(rec['delta']):+.1f}%", ha="center", va="bottom", fontsize=6.2)
        bx.set_xticks(x)
        bx.set_xticklabels(["BM", "TL"])
        bx.set_ylim(0, max(vals) * 1.32)
        if col == 0:
            bx.set_ylabel("Target-test MAE (weeks)", labelpad=2)
        else:
            bx.set_ylabel("")
        bx.grid(True, axis="y", color=COLORS["grid"], lw=0.35, zorder=1)
        style_axes(bx)

    handles = [
        mpl.lines.Line2D([0], [0], marker=MARKERS[label][0], color="none",
                         markerfacecolor=MARKERS[label][1], markeredgecolor=COLORS["black"],
                         markersize=4.5, label=label)
        for label in ["Source", "Fine-tune", "Test"]
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.51, 0.995),
               ncol=3, frameon=False, handlelength=0.8, handletextpad=0.35,
               columnspacing=1.0, labelspacing=0.20, borderaxespad=0.0)

    save_figure(fig, OUT_DIR, "Ch5_Fig4_finetune_selection_examples")
    plt.close(fig)


def main() -> None:
    setup_style()
    if "--list-seeds" in sys.argv:
        for rec in all_seed_records():
            print(
                f"{rec['name']}\tdistance={float(rec['distance']):.4f}"
                f"\tdelta_mae={float(rec['delta']):+.1f}"
                f"\tbm_mae={float(rec['bm_mae']):.2f}"
                f"\ttl_mae={float(rec['tl_mae']):.2f}"
            )
        return
    plot_main_condition_split()
    plot_week_availability()
    plot_lowcapacity_domain_effect()
    plot_selection_examples()
    print(f"[OK] Saved Chapter 5 figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
