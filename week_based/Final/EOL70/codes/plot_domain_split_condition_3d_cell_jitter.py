#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
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


def seed_from_dir(path: Path) -> str:
    name = path.name
    if "seed" in name:
        return name.rsplit("seed", 1)[-1]
    return ""


def find_week_split_dirs(domain_split: Path) -> list[Path]:
    protocol_dir = domain_split / "protocol_w6_10_from_stage3_final_rerun_400"
    candidates = []
    if protocol_dir.exists():
        candidates.append(protocol_dir)
    candidates.append(domain_split)

    seen: set[Path] = set()
    out: list[Path] = []
    for item in candidates:
        resolved = item.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(item)
    return out


def split_label(split: str) -> str:
    return {
        "train": "Train",
        "fine_tune": "Fine-tune",
        "test": "Test",
    }.get(split, split)


def cell_jitter(cell_idx: int, radius: float) -> tuple[float, float, float]:
    # Fixed offsets make the same cell occupy the same relative position
    # across all seeds, which exposes split-label changes directly.
    pattern = {
        1: (-1.0, -1.0, -0.4),
        2: (1.0, -1.0, 0.4),
        3: (-1.0, 1.0, 0.4),
        4: (1.0, 1.0, -0.4),
    }
    if cell_idx in pattern:
        dx, dy, dz = pattern[cell_idx]
    else:
        angle = 2.0 * math.pi * ((cell_idx - 1) % 8) / 8.0
        dx = math.cos(angle)
        dy = math.sin(angle)
        dz = 0.0
    return dx * radius, dy * radius, dz * radius * 12.0


def build_cell_points(
    split_csv: Path,
    cond_by_group: dict[int, dict[str, float]],
    jitter_radius: float,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows = read_rows(split_csv)
    required = {"split", "group_num", "cell", "cell_idx"}
    if rows:
        missing = required.difference(rows[0].keys())
        if missing:
            raise ValueError(f"Missing columns in {split_csv}: {sorted(missing)}")

    counts = {"train": 0, "fine_tune": 0, "test": 0}
    points: list[dict[str, object]] = []
    for row in rows:
        split = str(row.get("split", "")).strip()
        if split not in counts:
            continue
        group_num = int(parse_float(row["group_num"]))
        cell_idx = int(parse_float(row["cell_idx"]))
        if group_num not in cond_by_group:
            continue
        dx, dy, dz = cell_jitter(cell_idx, jitter_radius)
        cond = cond_by_group[group_num]
        points.append(
            {
                "split": split,
                "group_num": group_num,
                "cell": str(row["cell"]).strip(),
                "cell_idx": cell_idx,
                "discharging_crate": cond["discharging_crate"] + dx,
                "charging_crate": cond["charging_crate"] + dy,
                "dod_pct": cond["dod_pct"] + dz,
            }
        )
        counts[split] += 1
    return points, counts


def plot_cell_jitter_3d(
    points: list[dict[str, object]],
    out_path: Path,
    title: str,
    counts: dict[str, int],
    annotate: bool,
) -> None:
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
    order = ["train", "fine_tune", "test"]

    fig = plt.figure(figsize=(14.0, 9.2))
    ax = fig.add_subplot(111, projection="3d")
    for split in order:
        sub = [p for p in points if p["split"] == split]
        if not sub:
            continue
        ax.scatter(
            [float(p["discharging_crate"]) for p in sub],
            [float(p["charging_crate"]) for p in sub],
            [float(p["dod_pct"]) for p in sub],
            s=44 if split == "train" else 58,
            c=colors[split],
            marker=markers[split],
            alpha=0.82,
            edgecolors="black",
            linewidths=0.35,
            label=f"{split_label(split)} cells={counts.get(split, 0)}",
        )

    if annotate:
        for point in points:
            if point["split"] == "train":
                continue
            ax.text(
                float(point["discharging_crate"]) + 0.012,
                float(point["charging_crate"]) + 0.008,
                float(point["dod_pct"]) + 0.45,
                str(point["cell"]),
                fontsize=6.8,
                color="#303030",
            )

    ax.set_xlabel("Discharging C-rate + cell jitter", labelpad=12)
    ax.set_ylabel("Charging C-rate + cell jitter", labelpad=12)
    ax.set_zlabel("DoD (%) + cell jitter", labelpad=10)
    ax.set_title(title, pad=18)
    ax.view_init(elev=28, azim=-62)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_one(
    split_csv: Path,
    out_path: Path,
    title: str,
    cond_by_group: dict[int, dict[str, float]],
    jitter_radius: float,
    annotate: bool,
) -> None:
    points, counts = build_cell_points(split_csv, cond_by_group, jitter_radius)
    if not points:
        raise ValueError(f"No plottable cell points found in {split_csv}")
    plot_cell_jitter_3d(points, out_path, title, counts, annotate=annotate)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cell-level 3D operating-condition split points with deterministic jitter."
    )
    parser.add_argument("--domain_split", type=Path, default=DEFAULT_DOMAIN_SPLIT)
    parser.add_argument("--group_cond_csv", type=Path, default=DEFAULT_GROUP_CONDI)
    parser.add_argument("--weeks", default="6,7,8,9,10")
    parser.add_argument("--jitter_radius", type=float, default=0.035)
    parser.add_argument("--annotate", action="store_true")
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
            out_path = seed_dir / f"plot_condition_split_selection_3d_cell_jitter_w5_EOL70_seed{seed}.png"
            title = f"EOL70 w5 random split seed{seed}: cell-level jitter"
            plot_one(split_csv, out_path, title, cond_by_group, args.jitter_radius, annotate=bool(args.annotate))
            written.append(out_path)
            print(f"[ok] {out_path}")

    if not args.skip_weeks:
        weeks = [int(item.strip()) for item in str(args.weeks).split(",") if item.strip()]
        for week_dir in find_week_split_dirs(domain_split):
            for week in weeks:
                split_csv = week_dir / f"cell_split_targetspread_w{week}_EOL70.csv"
                if not split_csv.exists():
                    print(f"[skip] missing split CSV for week {week}: {split_csv}")
                    continue
                out_path = week_dir / f"plot_condition_split_selection_3d_cell_jitter_w{week}_EOL70.png"
                title = f"EOL70 w{week} protocol split: cell-level jitter"
                plot_one(split_csv, out_path, title, cond_by_group, args.jitter_radius, annotate=bool(args.annotate))
                written.append(out_path)
                print(f"[ok] {out_path}")

    print(f"[done] wrote {len(written)} plot(s)")


if __name__ == "__main__":
    main()
