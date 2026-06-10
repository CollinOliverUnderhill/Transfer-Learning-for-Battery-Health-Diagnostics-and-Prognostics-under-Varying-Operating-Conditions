#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKSPACE_ROOT.parents[2]
DEFAULT_DOMAIN_SPLIT = WORKSPACE_ROOT / "domain_split"
DEFAULT_GROUP_CONDI = REPO_ROOT / "metadata" / "Groupcondi.csv"


def parse_float(value: object) -> float:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return float("nan")
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        return float(text)
    except Exception:
        return float("nan")


def is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_group_conditions(path: Path) -> dict[int, dict[str, float]]:
    rows = read_rows(path)
    required = {"Group", "Charging C-rate", "Discharging C-rate", "Mean DoD"}
    if rows:
        missing = required.difference(rows[0].keys())
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    out: dict[int, dict[str, float]] = {}
    for row in rows:
        group_num = int(parse_float(row["Group"]))
        chg = parse_float(row["Charging C-rate"])
        dchg = parse_float(row["Discharging C-rate"])
        dod = parse_float(row["Mean DoD"])
        if all(is_finite(v) for v in (chg, dchg, dod)):
            out[group_num] = {
                "charging_crate": chg,
                "discharging_crate": dchg,
                "dod_pct": dod,
            }
    return out


def split_label(split: str) -> str:
    labels = {
        "train": "Train",
        "fine_tune": "Fine-tune",
        "test": "Test",
    }
    return labels.get(split, split)


def seed_from_dir(path: Path) -> str:
    name = path.name
    if "seed" in name:
        return name.rsplit("seed", 1)[-1]
    return ""


def build_group_split_points(
    split_csv: Path,
    cond_by_group: dict[int, dict[str, float]],
) -> tuple[list[dict[str, object]], dict[str, int], dict[str, int]]:
    rows = read_rows(split_csv)
    required = {"split", "group_num", "cell"}
    if rows:
        missing = required.difference(rows[0].keys())
        if missing:
            raise ValueError(f"Missing columns in {split_csv}: {sorted(missing)}")

    cells_by_split: dict[str, set[str]] = defaultdict(set)
    groups_by_split: dict[str, set[int]] = defaultdict(set)
    seen_pairs: set[tuple[str, int]] = set()
    points: list[dict[str, object]] = []

    for row in rows:
        split = str(row.get("split", "")).strip()
        if split not in {"train", "fine_tune", "test"}:
            continue
        group_num = int(parse_float(row["group_num"]))
        cell = str(row.get("cell", "")).strip()
        if group_num not in cond_by_group:
            continue
        if cell:
            cells_by_split[split].add(cell)
        groups_by_split[split].add(group_num)

        pair = (split, group_num)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        cond = cond_by_group[group_num]
        points.append(
            {
                "split": split,
                "group_num": group_num,
                "discharging_crate": cond["discharging_crate"],
                "charging_crate": cond["charging_crate"],
                "dod_pct": cond["dod_pct"],
            }
        )

    cell_counts = {split: len(cells_by_split.get(split, set())) for split in ("train", "fine_tune", "test")}
    group_counts = {split: len(groups_by_split.get(split, set())) for split in ("train", "fine_tune", "test")}
    return points, cell_counts, group_counts


def add_overlap_offsets(points: list[dict[str, object]], delta: float = 0.018) -> list[dict[str, object]]:
    by_group: dict[int, list[dict[str, object]]] = defaultdict(list)
    for point in points:
        by_group[int(point["group_num"])].append(point)

    offset_by_split = {
        "train": (0.0, 0.0, 0.0),
        "fine_tune": (-delta, 0.0, 0.0),
        "test": (delta, 0.0, 0.0),
    }
    out: list[dict[str, object]] = []
    for group_points in by_group.values():
        multi_split = len({str(p["split"]) for p in group_points}) > 1
        for point in group_points:
            item = dict(point)
            if multi_split:
                dx, dy, dz = offset_by_split.get(str(point["split"]), (0.0, 0.0, 0.0))
                item["discharging_crate"] = float(item["discharging_crate"]) + dx
                item["charging_crate"] = float(item["charging_crate"]) + dy
                item["dod_pct"] = float(item["dod_pct"]) + dz
            out.append(item)
    return out


def plot_condition_split_3d(
    points: list[dict[str, object]],
    out_path: Path,
    title: str,
    cell_counts: dict[str, int],
    group_counts: dict[str, int],
    annotate: bool,
) -> None:
    colors = {
        "train": "#F58518",
        "fine_tune": "#54A24B",
        "test": "#4C78A8",
    }
    sizes = {
        "train": 62,
        "fine_tune": 74,
        "test": 74,
    }
    markers = {
        "train": "o",
        "fine_tune": "^",
        "test": "s",
    }
    order = ["train", "fine_tune", "test"]

    fig = plt.figure(figsize=(14.0, 9.2))
    ax = fig.add_subplot(111, projection="3d")

    for split in order:
        sub = [p for p in points if p["split"] == split]
        if not sub:
            continue
        label = (
            f"{split_label(split)} "
            f"(groups={group_counts.get(split, 0)}, cells={cell_counts.get(split, 0)})"
        )
        ax.scatter(
            [float(p["discharging_crate"]) for p in sub],
            [float(p["charging_crate"]) for p in sub],
            [float(p["dod_pct"]) for p in sub],
            s=sizes[split],
            c=colors[split],
            marker=markers[split],
            alpha=0.88,
            edgecolors="black",
            linewidths=0.45,
            label=label,
        )

    if annotate:
        label_points: dict[int, dict[str, object]] = {}
        for point in points:
            group_num = int(point["group_num"])
            label_points.setdefault(group_num, point)
        for group_num, point in sorted(label_points.items()):
            ax.text(
                float(point["discharging_crate"]) + 0.018,
                float(point["charging_crate"]) + 0.012,
                float(point["dod_pct"]) + 0.6,
                f"G{group_num}",
                fontsize=8.0,
                color="#303030",
            )

    ax.set_xlabel("Discharging C-rate", labelpad=12)
    ax.set_ylabel("Charging C-rate", labelpad=12)
    ax.set_zlabel("DoD (%)", labelpad=10)
    ax.set_title(title, pad=18)
    ax.view_init(elev=28, azim=-62)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_one(split_csv: Path, out_path: Path, title: str, cond_by_group: dict[int, dict[str, float]], annotate: bool) -> None:
    points, cell_counts, group_counts = build_group_split_points(split_csv, cond_by_group)
    points = add_overlap_offsets(points)
    if not points:
        raise ValueError(f"No plottable split points found in {split_csv}")
    plot_condition_split_3d(
        points=points,
        out_path=out_path,
        title=title,
        cell_counts=cell_counts,
        group_counts=group_counts,
        annotate=annotate,
    )


def find_week_split_dirs(domain_split: Path) -> list[Path]:
    protocol_dir = domain_split / "protocol_w6_10_from_stage3_final_rerun_400"
    dirs = []
    if protocol_dir.exists():
        dirs.append(protocol_dir)
    dirs.append(domain_split)

    seen: set[Path] = set()
    out: list[Path] = []
    for item in dirs:
        resolved = item.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot 3D operating-condition split points for EOL70 seed and week6-10 domain splits."
    )
    parser.add_argument("--domain_split", type=Path, default=DEFAULT_DOMAIN_SPLIT)
    parser.add_argument("--group_cond_csv", type=Path, default=DEFAULT_GROUP_CONDI)
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--weeks", default="6,7,8,9,10")
    parser.add_argument("--skip_seeds", action="store_true")
    parser.add_argument("--skip_weeks", action="store_true")
    args = parser.parse_args()

    domain_split = args.domain_split
    cond_by_group = load_group_conditions(args.group_cond_csv)
    written: list[Path] = []

    if not args.skip_seeds:
        seed_dirs = sorted(p for p in domain_split.glob("w5_EOL70_random_seed*") if p.is_dir())
        for seed_dir in seed_dirs:
            seed = seed_from_dir(seed_dir)
            split_csv = seed_dir / f"cell_split_targetrandom_w5_EOL70_seed{seed}.csv"
            if not split_csv.exists():
                print(f"[skip] missing split CSV: {split_csv}")
                continue
            out_path = seed_dir / f"plot_condition_split_selection_3d_w5_EOL70_seed{seed}.png"
            title = f"EOL70 w5 random split seed{seed}: train / fine-tune / test"
            plot_one(split_csv, out_path, title, cond_by_group, annotate=bool(args.annotate))
            written.append(out_path)
            print(f"[ok] {out_path}")

    if not args.skip_weeks:
        week_dirs = find_week_split_dirs(domain_split)
        weeks = [int(item.strip()) for item in str(args.weeks).split(",") if item.strip()]
        for week_dir in week_dirs:
            for week in weeks:
                split_csv = week_dir / f"cell_split_targetspread_w{week}_EOL70.csv"
                if not split_csv.exists():
                    print(f"[skip] missing split CSV for week {week}: {split_csv}")
                    continue
                out_path = week_dir / f"plot_condition_split_selection_3d_w{week}_EOL70.png"
                title = f"EOL70 w{week} protocol split: train / fine-tune / test"
                plot_one(split_csv, out_path, title, cond_by_group, annotate=bool(args.annotate))
                written.append(out_path)
                print(f"[ok] {out_path}")

    print(f"[done] wrote {len(written)} plot(s)")


if __name__ == "__main__":
    main()
