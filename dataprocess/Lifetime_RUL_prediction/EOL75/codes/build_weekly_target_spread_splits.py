#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence


ROOT = Path(r"E:\Datasets\IVAS\Lifetime_RUL_prediction\EOL75")
SOURCE_CSV = Path(r"E:\Datasets\IVAS\Processing_Data_dd_exclude\EOL75\cell_split_by_lifetime_EOL75_augmented_weeks.csv")
WEEKS = (3, 5, 10, 15)
TARGET_COL = "lifetime_weeks_EOL75"
TRAIN_GROUP_FRAC = 0.70
FT_CELL_FRAC = 0.15


def to_float(value: object) -> float:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return float("nan")
    try:
        return float(text)
    except Exception:
        return float("nan")


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def usable_for_week(row: Dict[str, str], week: int) -> bool:
    for idx in range(1, 11):
        if not math.isfinite(to_float(row.get(f"f{idx}_w{week}", ""))):
            return False
    return True


def evenly_spaced_indices(n: int, k: int) -> List[int]:
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    idxs = [int(round(i * (n - 1) / float(k - 1))) for i in range(k)]
    deduped: List[int] = []
    seen = set()
    for idx in idxs:
        idx = max(0, min(n - 1, idx))
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    cur = 0
    while len(deduped) < k:
        if cur not in seen:
            seen.add(cur)
            deduped.append(cur)
        cur += 1
    return sorted(deduped)


def pick_ft_cells_spread(target_groups: Sequence[Dict[str, object]], target_rows: Sequence[Dict[str, str]], ft_quota: int) -> List[str]:
    rows_by_group: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in target_rows:
        rows_by_group[int(to_float(row["group_num"]))].append(row)

    ordered_group_rows: List[List[Dict[str, str]]] = []
    for group in target_groups:
        group_num = int(group["group_num"])
        cell_rows = sorted(
            rows_by_group[group_num],
            key=lambda r: (int(to_float(r["cell_idx"])), str(r["cell"])),
        )
        if cell_rows:
            ordered_group_rows.append(cell_rows)

    total_target_cells = sum(len(items) for items in ordered_group_rows)
    if total_target_cells == 0:
        return []
    if total_target_cells == 1:
        return [str(ordered_group_rows[0][0]["cell"])]

    ft_quota = max(1, min(int(ft_quota), total_target_cells - 1))
    selected: List[str] = []
    selected_set = set()

    first_pass = evenly_spaced_indices(len(ordered_group_rows), min(ft_quota, len(ordered_group_rows)))
    for group_idx in first_pass:
        row = ordered_group_rows[group_idx][0]
        cell = str(row["cell"])
        selected.append(cell)
        selected_set.add(cell)

    while len(selected) < ft_quota:
        added = False
        for group_rows in ordered_group_rows:
            for row in group_rows:
                cell = str(row["cell"])
                if cell in selected_set:
                    continue
                selected.append(cell)
                selected_set.add(cell)
                added = True
                break
            if len(selected) >= ft_quota:
                break
        if not added:
            break

    return selected


def build_group_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(to_float(row["group_num"]))].append(row)

    group_rows: List[Dict[str, object]] = []
    for group_num, items in grouped.items():
        lifetimes = [to_float(r[TARGET_COL]) for r in items if math.isfinite(to_float(r[TARGET_COL]))]
        if not lifetimes:
            continue
        sorted_lifetimes = sorted(lifetimes)
        if len(sorted_lifetimes) % 2 == 1:
            lifetime_median = sorted_lifetimes[len(sorted_lifetimes) // 2]
        else:
            right = len(sorted_lifetimes) // 2
            lifetime_median = (sorted_lifetimes[right - 1] + sorted_lifetimes[right]) / 2.0
        group_rows.append(
            {
                "group_num": group_num,
                "group_label": f"G{group_num}",
                "valid_cell_count": len(items),
                "lifetime_min": min(lifetimes),
                "lifetime_median": lifetime_median,
                "lifetime_mean": sum(lifetimes) / len(lifetimes),
                "lifetime_max": max(lifetimes),
                "cells": ",".join(sorted(str(r["cell"]) for r in items)),
            }
        )
    group_rows.sort(key=lambda row: (float(row["lifetime_median"]), float(row["lifetime_mean"]), int(row["group_num"])))
    return group_rows


def main() -> None:
    domain_dir = ROOT / "domain_split"
    features_dir = ROOT / "features"
    rows = read_rows(SOURCE_CSV)

    write_rows(
        features_dir / "feature_table_all_cells_multiweek_EOL75.csv",
        rows,
        list(rows[0].keys()),
    )

    manifest_rows: List[Dict[str, object]] = []

    for week in WEEKS:
        usable_rows = [dict(row) for row in rows if usable_for_week(row, week)]
        group_rows = build_group_rows(usable_rows)
        train_end = round_half_up(len(group_rows) * TRAIN_GROUP_FRAC)
        train_groups = group_rows[:train_end]
        target_groups = group_rows[train_end:]
        train_group_set = {int(row["group_num"]) for row in train_groups}
        target_group_set = {int(row["group_num"]) for row in target_groups}

        target_rows = [row for row in usable_rows if int(to_float(row["group_num"])) in target_group_set]
        ft_quota = round_half_up(len(usable_rows) * FT_CELL_FRAC)
        ft_cells = set(pick_ft_cells_spread(target_groups, target_rows, ft_quota))

        split_rows: List[Dict[str, object]] = []
        for row in usable_rows:
            group_num = int(to_float(row["group_num"]))
            cell = str(row["cell"])
            if group_num in train_group_set:
                split = "train"
                split_label = "Train"
                domain = "source"
            elif cell in ft_cells:
                split = "fine_tune"
                split_label = "Fine-tune"
                domain = "target"
            else:
                split = "test"
                split_label = "Test"
                domain = "target"
            split_rows.append(
                {
                    **row,
                    "feature_week": f"w{week}",
                    "target_domain": domain,
                    "split": split,
                    "split_label": split_label,
                }
            )

        split_rows.sort(key=lambda r: (str(r["split"]), int(to_float(r["group_num"])), int(to_float(r["cell_idx"])), str(r["cell"])))

        ft_counts = defaultdict(int)
        test_counts = defaultdict(int)
        for row in split_rows:
            g = int(to_float(row["group_num"]))
            if row["split"] == "fine_tune":
                ft_counts[g] += 1
            elif row["split"] == "test":
                test_counts[g] += 1

        group_export: List[Dict[str, object]] = []
        for rank, group in enumerate(group_rows, start=1):
            group_num = int(group["group_num"])
            domain = "source" if group_num in train_group_set else "target"
            group_export.append(
                {
                    "feature_week": f"w{week}",
                    "rank_by_lifetime": rank,
                    "domain": domain,
                    "group_num": group_num,
                    "group_label": group["group_label"],
                    "valid_cell_count": group["valid_cell_count"],
                    "ft_cell_count": ft_counts[group_num],
                    "test_cell_count": test_counts[group_num],
                    "lifetime_min": group["lifetime_min"],
                    "lifetime_median": group["lifetime_median"],
                    "lifetime_mean": group["lifetime_mean"],
                    "lifetime_max": group["lifetime_max"],
                    "cells": group["cells"],
                }
            )

        split_summary = [
            {
                "feature_week": f"w{week}",
                "usable_cells": len(usable_rows),
                "train_cells": sum(1 for r in split_rows if r["split"] == "train"),
                "fine_tune_cells": sum(1 for r in split_rows if r["split"] == "fine_tune"),
                "test_cells": sum(1 for r in split_rows if r["split"] == "test"),
                "train_groups": len(train_groups),
                "target_groups": len(target_groups),
            }
        ]

        split_csv = domain_dir / f"cell_split_targetspread_w{week}_EOL75.csv"
        group_csv = domain_dir / f"group_split_targetspread_w{week}_EOL75.csv"
        summary_csv = domain_dir / f"split_summary_targetspread_w{week}_EOL75.csv"
        write_rows(split_csv, split_rows, list(split_rows[0].keys()))
        write_rows(group_csv, group_export, list(group_export[0].keys()))
        write_rows(summary_csv, split_summary, list(split_summary[0].keys()))

        manifest_rows.append(
            {
                "feature_week": f"w{week}",
                "split_csv": str(split_csv),
                "group_csv": str(group_csv),
                "summary_csv": str(summary_csv),
                "usable_cells": len(usable_rows),
            }
        )

    write_rows(
        domain_dir / "weekly_split_manifest_EOL75.csv",
        manifest_rows,
        list(manifest_rows[0].keys()),
    )


if __name__ == "__main__":
    main()
