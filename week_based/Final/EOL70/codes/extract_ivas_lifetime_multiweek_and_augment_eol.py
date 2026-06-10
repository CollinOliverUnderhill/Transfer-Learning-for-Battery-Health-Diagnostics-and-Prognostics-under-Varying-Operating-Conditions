#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build multi-week lifetime features for IVAS and augment existing dd_exclude EOL splits.

This keeps the original week3-only outputs untouched and exports new files with
explicit week blocks for week3 / week5 / week6 / week7 / week8 / week9 / week10 / week15.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import extract_ivas_lifetime_10features as base


WEEKS: Tuple[int, ...] = (3, 5, 6, 7, 8, 9, 10, 15)
EOL_PCTS: Tuple[int, ...] = (50, 55, 60, 65, 70, 75, 80)
MASTER_OUT_CSV = base.DEFAULT_OUT_DIR / "ivas_lifetime_10features_multiweek_per_cell.csv"
DD_EXCLUDE_ROOT = base.IVAS_ROOT / "Processing_Data_dd_exclude"


def to_float(value: object) -> float:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return float("nan")
    try:
        return float(text)
    except Exception:
        return float("nan")


def make_fieldnames() -> List[str]:
    fieldnames = [
        "group_num",
        "cell_idx",
        "cell",
        "release",
    ]
    fieldnames.extend(f"lifetime_weeks_EOL{pct}" for pct in (80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30))
    fieldnames.extend(
        [
            "week0_rpt_idx",
            "week0_time_week",
            "Q_initial_ah",
            "chg_c_rate",
            "dchg_c_rate",
        ]
    )
    for week in WEEKS:
        fieldnames.append(f"feature_status_w{week}")
        fieldnames.append(f"week{week}_rpt_idx")
        fieldnames.append(f"week{week}_time_week")
        fieldnames.extend(f"f{i}_w{week}" for i in range(1, 11))
    return fieldnames


def build_week_payload(
    *,
    week: int,
    group_num: int,
    time_week: np.ndarray,
    capacity_ah: np.ndarray,
    qmat: np.ndarray,
    rpt_dict: Dict[str, object],
    dod: float,
    cchg: float,
    dqdv: np.ndarray,
    dvdq: np.ndarray,
    low_mask: np.ndarray,
    mid_mask: np.ndarray,
) -> Dict[str, object]:
    w0_idx = base.week_to_rpt_idx(group_num, 0)
    wk_idx = base.week_to_rpt_idx(group_num, week)

    cv_time_w0 = (
        base.charge_cv_time_from_rpt_index(rpt_dict, w0_idx)
        if len(rpt_dict["QV_charge_C_5"]["I"]) > w0_idx
        else float("nan")
    )
    sqrt_stress = float(np.sqrt(max(cchg, 0.0)) * np.sqrt(max(dod, 0.0)))

    payload: Dict[str, object] = {
        f"feature_status_w{week}": "missing_week%d" % week,
        f"week{week}_rpt_idx": wk_idx,
        f"week{week}_time_week": float("nan"),
        f"f1_w{week}": float("nan"),
        f"f2_w{week}": float("nan"),
        f"f3_w{week}": float(dod),
        f"f4_w{week}": float("nan"),
        f"f5_w{week}": sqrt_stress,
        f"f6_w{week}": float(cchg),
        f"f7_w{week}": float("nan"),
        f"f8_w{week}": float("nan"),
        f"f9_w{week}": float("nan"),
        f"f10_w{week}": base.safe_log_abs(cv_time_w0),
    }

    if qmat.shape[1] <= max(w0_idx, wk_idx):
        return payload
    if len(time_week) <= max(w0_idx, wk_idx):
        return payload
    if len(rpt_dict["QV_charge_C_5"]["I"]) <= max(w0_idx, wk_idx):
        return payload

    delta_mid = dqdv[mid_mask, wk_idx] - dqdv[mid_mask, w0_idx]
    delta_low = dqdv[low_mask, wk_idx] - dqdv[low_mask, w0_idx]
    cv_time_wk = base.charge_cv_time_from_rpt_index(rpt_dict, wk_idx)
    delta_q1, delta_q3 = base.dva_delta_features(
        q_ref_curve=qmat[:, w0_idx],
        q_cur_curve=qmat[:, wk_idx],
        dvdq_ref_curve=dvdq[:, w0_idx],
        dvdq_cur_curve=dvdq[:, wk_idx],
    )

    payload.update(
        {
            f"feature_status_w{week}": "ok",
            f"week{week}_time_week": float(time_week[wk_idx]) if np.isfinite(time_week[wk_idx]) else float("nan"),
            f"f1_w{week}": base.safe_log_abs(base.finite_mean(delta_mid)),
            f"f2_w{week}": base.safe_log_abs(cv_time_wk - cv_time_w0),
            f"f4_w{week}": delta_q1,
            f"f7_w{week}": base.safe_log_abs(base.finite_var(delta_low)),
            f"f8_w{week}": delta_q3,
            f"f9_w{week}": base.safe_log_abs(base.finite_mean(delta_low)),
        }
    )
    return payload


def build_multiweek_row(
    *,
    cell: str,
    release: str,
    time_week: np.ndarray,
    capacity_ah: np.ndarray,
    qmat: np.ndarray,
    rpt_dict: Dict[str, object],
    dod: float,
    cchg: float,
    cdchg: float,
) -> Dict[str, object]:
    group_num = int(cell[1:-2])
    cell_idx = int(cell[-1])

    lifetime_cols = base.compute_lifetime_columns(time_week=time_week, capacity_ah=capacity_ah)
    w0_idx = base.week_to_rpt_idx(group_num, 0)

    dqdv = np.diff(qmat, axis=0) / np.diff(np.linspace(base.V_MIN, base.V_MAX, base.N_VOLTAGE))[:, None]
    dvdq = np.diff(np.linspace(base.V_MIN, base.V_MAX, base.N_VOLTAGE))[:, None] / np.diff(qmat, axis=0)
    dqdv_v = np.linspace(3.0, 4.2, dqdv.shape[0])
    low_mask = (dqdv_v >= 3.0) & (dqdv_v <= 3.6)
    mid_mask = (dqdv_v >= 3.6) & (dqdv_v <= 3.9)

    row: Dict[str, object] = {
        "group_num": group_num,
        "cell_idx": cell_idx,
        "cell": cell,
        "release": release,
        **lifetime_cols,
        "week0_rpt_idx": w0_idx,
        "week0_time_week": float(time_week[w0_idx]) if len(time_week) > w0_idx and np.isfinite(time_week[w0_idx]) else float("nan"),
        "Q_initial_ah": float(capacity_ah[w0_idx]) if len(capacity_ah) > w0_idx and np.isfinite(capacity_ah[w0_idx]) else float("nan"),
        "chg_c_rate": float(cchg),
        "dchg_c_rate": float(cdchg),
    }

    for week in WEEKS:
        row.update(
            build_week_payload(
                week=week,
                group_num=group_num,
                time_week=time_week,
                capacity_ah=capacity_ah,
                qmat=qmat,
                rpt_dict=rpt_dict,
                dod=dod,
                cchg=cchg,
                dqdv=dqdv,
                dvdq=dvdq,
                low_mask=low_mask,
                mid_mask=mid_mask,
            )
        )
    return row


def load_multiweek_rows() -> List[Dict[str, object]]:
    valid_cells = base.read_valid_cells(base.IVAS_ROOT / "Valid_cells.csv")
    dod_map = base.load_dod_map(base.AUTHOR_FEAT_DIR / "empirical_DoD.csv")
    cond_map = base.load_condition_map(base.AUTHOR_FEAT_DIR / "cycling_conditions_wo_DoD.csv")

    rows: List[Dict[str, object]] = []
    skipped: List[str] = []

    for cell in valid_cells:
        resolved = base.resolve_release_paths(cell)
        if resolved is None:
            skipped.append(f"{cell}:missing_release")
            continue

        release, cap_path, q_path, rpt_path = resolved
        group_num = int(cell[1:-2])
        if cell not in dod_map or group_num not in cond_map:
            skipped.append(f"{cell}:missing_meta")
            continue

        time_week, capacity_ah = base.read_capacity_fade(cap_path)
        qmat = base.drop_all_nan_cols(base.read_q_interpolated(q_path))
        rpt_dict = base.load_rpt_dict(rpt_path)

        n = min(
            len(time_week),
            len(capacity_ah),
            qmat.shape[1],
            len(rpt_dict["QV_charge_C_5"]["I"]),
        )
        if n == 0:
            skipped.append(f"{cell}:empty_timeseries")
            continue

        try:
            rows.append(
                build_multiweek_row(
                    cell=cell,
                    release=release,
                    time_week=time_week[:n],
                    capacity_ah=capacity_ah[:n],
                    qmat=qmat[:, :n],
                    rpt_dict=rpt_dict,
                    dod=float(dod_map[cell]),
                    cchg=float(cond_map[group_num]["Cchg"]),
                    cdchg=float(cond_map[group_num]["Cdchg"]),
                )
            )
        except Exception as exc:
            skipped.append(f"{cell}:{exc}")

    rows = base.sort_rows(rows)
    print(f"[INFO] multiweek rows        : {len(rows)}")
    print(f"[INFO] skipped cells        : {len(skipped)}")
    if skipped:
        print(f"[INFO] first skipped        : {skipped[:10]}")
    return rows


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def augment_split_file(path: Path, row_by_cell: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    source_rows = read_csv_rows(path)
    augmented_rows: List[Dict[str, object]] = []
    for row in source_rows:
        cell = str(row.get("cell", "")).strip()
        merged = dict(row)
        merged.update(row_by_cell.get(cell, {}))
        augmented_rows.append(merged)
    return augmented_rows


def build_week_availability_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    total_cells = len(rows)
    summary_rows: List[Dict[str, object]] = []
    for week in WEEKS:
        feature_cols = [f"f{i}_w{week}" for i in range(1, 11)]
        usable_non_nan = 0
        status_ok = 0
        for row in rows:
            if str(row.get(f"feature_status_w{week}", "")).strip() == "ok":
                status_ok += 1
            if all(np.isfinite(to_float(row.get(col, ""))) for col in feature_cols):
                usable_non_nan += 1
        summary_rows.append(
            {
                "week": week,
                "usable_non_nan_cells": usable_non_nan,
                "status_ok_cells": status_ok,
                "missing_or_nan_cells": total_cells - usable_non_nan,
                "total_cells": total_cells,
            }
        )
    return summary_rows


def augment_eol_folders(rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    row_by_cell = {str(row["cell"]).strip(): row for row in rows}
    extra_fields = [name for name in fieldnames if name != "cell"]

    for pct in EOL_PCTS:
        eol_dir = DD_EXCLUDE_ROOT / f"EOL{pct}"
        overall_name = f"cell_split_by_lifetime_EOL{pct}.csv"
        file_names = [
            overall_name,
            f"cell_split_train_by_lifetime_EOL{pct}.csv",
            f"cell_split_ft_by_lifetime_EOL{pct}.csv",
            f"cell_split_test_by_lifetime_EOL{pct}.csv",
        ]

        augmented_overall_rows: List[Dict[str, object]] = []
        for file_name in file_names:
            src_path = eol_dir / file_name
            if not src_path.exists():
                continue
            augmented_rows = augment_split_file(src_path, row_by_cell)
            out_name = file_name.replace(".csv", "_augmented_weeks.csv")
            out_path = eol_dir / out_name
            out_fields = list(read_csv_rows(src_path)[0].keys()) + [f for f in extra_fields if f not in read_csv_rows(src_path)[0]]
            write_csv_rows(out_path, augmented_rows, out_fields)
            if file_name == overall_name:
                augmented_overall_rows = augmented_rows
            print(f"[INFO] saved augmented csv   : {out_path}")

        if augmented_overall_rows:
            summary_path = eol_dir / f"week_availability_summary_EOL{pct}.csv"
            write_csv_rows(
                summary_path,
                build_week_availability_summary(augmented_overall_rows),
                ["week", "usable_non_nan_cells", "status_ok_cells", "missing_or_nan_cells", "total_cells"],
            )
            print(f"[INFO] saved week summary    : {summary_path}")


def main() -> None:
    rows = load_multiweek_rows()
    fieldnames = make_fieldnames()
    write_csv_rows(MASTER_OUT_CSV, rows, fieldnames)
    print(f"[INFO] saved master csv      : {MASTER_OUT_CSV}")
    augment_eol_folders(rows, fieldnames)


if __name__ == "__main__":
    main()
