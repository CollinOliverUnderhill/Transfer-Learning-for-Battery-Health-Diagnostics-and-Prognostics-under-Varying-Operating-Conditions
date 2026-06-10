#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_CSV = ROOT / "domain_split" / "cell_split_targetspread_w5_EOL70.csv"
DEFAULT_GROUP_CSV = ROOT / "domain_split" / "group_split_targetspread_w5_EOL70.csv"
DEFAULT_OUT_ROOT = ROOT / "domain_split"


def parse_seed_spec(text: str) -> list[int]:
    seeds: list[int] = []
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start = int(left.strip())
            stop = int(right.strip())
            step = 1 if stop >= start else -1
            seeds.extend(range(start, stop + step, step))
        else:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("No seeds specified.")
    return seeds


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_int(value: object) -> int:
    return int(float(str(value).strip()))


def choose_fine_tune_cells(
    target_rows: list[dict[str, str]],
    ft_quota: int,
    seed: int,
) -> set[str]:
    rng = random.Random(int(seed))
    rows_by_group: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in target_rows:
        rows_by_group[to_int(row["group_num"])].append(row)

    selected: set[str] = set()
    extra_pool: list[str] = []

    for group_num in sorted(rows_by_group):
        group_rows = sorted(
            rows_by_group[group_num],
            key=lambda row: (to_int(row["cell_idx"]), str(row["cell"])),
        )
        cells = [str(row["cell"]) for row in group_rows]
        rng.shuffle(cells)

        # Keep the target-domain coverage stable: every target group contributes
        # at least one fine-tune cell; groups with >1 cells keep one held-out test.
        selected.add(cells[0])
        if len(cells) > 2:
            extra_pool.extend(cells[2:])

    remaining = int(ft_quota) - len(selected)
    if remaining < 0:
        raise ValueError(
            f"ft_quota={ft_quota} is smaller than the target-group count={len(selected)}."
        )
    if remaining > len(extra_pool):
        raise ValueError(
            f"ft_quota={ft_quota} needs {remaining} extra cells, but only {len(extra_pool)} are available."
        )

    selected.update(rng.sample(extra_pool, remaining))
    return selected


def build_split_for_seed(
    base_rows: list[dict[str, str]],
    base_fieldnames: list[str],
    base_group_rows: list[dict[str, str]],
    base_group_fieldnames: list[str],
    seed: int,
    ft_frac: float,
    out_root: Path,
    overwrite: bool,
) -> None:
    seed_name = f"seed{seed:03d}"
    out_dir = out_root / f"w5_EOL70_random_{seed_name}"
    split_csv = out_dir / f"cell_split_targetrandom_w5_EOL70_{seed_name}.csv"
    group_csv = out_dir / f"group_split_targetrandom_w5_EOL70_{seed_name}.csv"
    summary_csv = out_dir / f"split_summary_targetrandom_w5_EOL70_{seed_name}.csv"

    if split_csv.exists() and not overwrite:
        print(f"[skip] {split_csv} exists. Use --overwrite to rebuild it.")
        return

    target_rows = [row for row in base_rows if row.get("target_domain") == "target"]
    ft_quota = round_half_up(len(base_rows) * float(ft_frac))
    ft_cells = choose_fine_tune_cells(target_rows, ft_quota=ft_quota, seed=seed)

    split_rows: list[dict[str, object]] = []
    for row in base_rows:
        out_row: dict[str, object] = dict(row)
        if row.get("target_domain") != "target":
            out_row["split"] = "train"
            out_row["split_label"] = "Train"
        elif str(row["cell"]) in ft_cells:
            out_row["split"] = "fine_tune"
            out_row["split_label"] = "Fine-tune"
        else:
            out_row["split"] = "test"
            out_row["split_label"] = "Test"
        split_rows.append(out_row)

    split_rows.sort(
        key=lambda row: (
            str(row["split"]),
            to_int(row["group_num"]),
            to_int(row["cell_idx"]),
            str(row["cell"]),
        )
    )

    ft_counts: dict[int, int] = defaultdict(int)
    test_counts: dict[int, int] = defaultdict(int)
    for row in split_rows:
        group_num = to_int(row["group_num"])
        if row["split"] == "fine_tune":
            ft_counts[group_num] += 1
        elif row["split"] == "test":
            test_counts[group_num] += 1

    group_rows: list[dict[str, object]] = []
    for row in base_group_rows:
        out_row: dict[str, object] = dict(row)
        group_num = to_int(row["group_num"])
        out_row["ft_cell_count"] = ft_counts[group_num]
        out_row["test_cell_count"] = test_counts[group_num]
        group_rows.append(out_row)

    summary_rows = [
        {
            "feature_week": "w5",
            "usable_cells": len(base_rows),
            "train_cells": sum(1 for row in split_rows if row["split"] == "train"),
            "fine_tune_cells": sum(1 for row in split_rows if row["split"] == "fine_tune"),
            "test_cells": sum(1 for row in split_rows if row["split"] == "test"),
            "train_groups": sum(1 for row in base_group_rows if row.get("domain") == "source"),
            "target_groups": sum(1 for row in base_group_rows if row.get("domain") == "target"),
        }
    ]

    write_rows(split_csv, split_rows, base_fieldnames)
    write_rows(group_csv, group_rows, base_group_fieldnames)
    write_rows(summary_csv, summary_rows, list(summary_rows[0].keys()))
    print(f"[ok] wrote {seed_name}: {split_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build additional random fine-tune/test splits for EOL70 week 5."
    )
    parser.add_argument("--seeds", default="10-19", help="Comma/range spec, e.g. 10-19 or 10,11,12.")
    parser.add_argument("--base_split_csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--base_group_csv", type=Path, default=DEFAULT_GROUP_CSV)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--ft_frac", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_rows, base_fieldnames = read_rows(args.base_split_csv)
    base_group_rows, base_group_fieldnames = read_rows(args.base_group_csv)

    required = {"split", "split_label", "group_num", "cell", "cell_idx", "target_domain"}
    missing = sorted(required.difference(base_fieldnames))
    if missing:
        raise ValueError(f"Base split CSV is missing required columns: {missing}")

    for seed in parse_seed_spec(args.seeds):
        build_split_for_seed(
            base_rows=base_rows,
            base_fieldnames=base_fieldnames,
            base_group_rows=base_group_rows,
            base_group_fieldnames=base_group_fieldnames,
            seed=seed,
            ft_frac=float(args.ft_frac),
            out_root=args.out_root,
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    main()
