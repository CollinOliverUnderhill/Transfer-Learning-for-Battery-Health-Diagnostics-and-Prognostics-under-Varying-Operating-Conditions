#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rebuild the 10 lifetime-prediction features from IVAS Table 1 at the cell level.

This script follows the author's early-life setup:
  - time-varying features are computed as week3 - week0
  - static features are taken directly from cycling condition / initial RPT

Outputs:
  E:/Datasets/IVAS/Processing_Data/Lifetime_prediction/
    - ivas_lifetime_10features_per_cell.csv
    - ivas_lifetime_10features_dictionary.csv
    - ivas_lifetime_10features_dictionary.md

Dependencies:
  - Python standard library
  - numpy

No pandas / scipy is required in the current environment.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


IVAS_ROOT = Path(r"E:\Datasets\IVAS")
AUTHOR_FEAT_DIR = (
    IVAS_ROOT
    / "Codes"
    / "tingkai-li-early-prediction-varying-usage-data-d1f5535"
    / "feature_extraction"
)
DEFAULT_OUT_DIR = IVAS_ROOT / "Processing_Data" / "Lifetime_prediction"

V_MIN = 3.0
V_MAX = 4.18
N_VOLTAGE = 1000
PCHIP_STEP = 0.001
RETENTION_LEVELS = (0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30)

FEATURE_SPECS: List[Dict[str, str]] = [
    {
        "step_number": "1",
        "feature_key": "step1_log_abs_mean_delta_dQdV_w3_w0_3p6_3p9",
        "formula": "log(|mean((dQ/dV)_w3 - (dQ/dV)_w0, V in [3.6, 3.9])|)",
        "author_source": "mean_dqdv_dchg_mid_3_0 -> log(abs(.))",
        "description": "Best incremental-capacity feature from 3.6V-3.9V.",
    },
    {
        "step_number": "2",
        "feature_key": "step2_log_abs_delta_CV_time_w3_w0",
        "formula": "log(|CV_time_w3 - CV_time_w0|)",
        "author_source": "delta_CV_time_3_0 -> log(abs(.))",
        "description": "Change in CV hold time from week 0 to week 3.",
    },
    {
        "step_number": "3",
        "feature_key": "step3_DoD",
        "formula": "DoD",
        "author_source": "DoD",
        "description": "Empirical depth of discharge.",
    },
    {
        "step_number": "4",
        "feature_key": "step4_delta_Q1_DVA_w3_w0",
        "formula": "Delta Q1_DVA(w3 - w0)",
        "author_source": "delta_Q_DVA1",
        "description": "Change in DVA-based capacity component Q_DVA,1.",
    },
    {
        "step_number": "5",
        "feature_key": "step5_sqrt_Cchg_sqrt_DoD",
        "formula": "Cchg^0.5 * DoD^0.5",
        "author_source": "chg_stress",
        "description": "Charge-induced stress from cycling conditions.",
    },
    {
        "step_number": "6",
        "feature_key": "step6_Cchg",
        "formula": "Cchg",
        "author_source": "Chg C-rate",
        "description": "Charging C-rate.",
    },
    {
        "step_number": "7",
        "feature_key": "step7_log_abs_var_delta_dQdV_w3_w0_3p0_3p6",
        "formula": "log(|var((dQ/dV)_w3 - (dQ/dV)_w0, V in [3.0, 3.6])|)",
        "author_source": "var_dqdv_dchg_low_3_0 -> log(abs(.))",
        "description": "Variance of low-voltage incremental-capacity delta.",
    },
    {
        "step_number": "8",
        "feature_key": "step8_delta_Q3_DVA_w3_w0",
        "formula": "Delta Q3_DVA(w3 - w0)",
        "author_source": "delta_Q_DVA3",
        "description": "Change in DVA-based capacity component Q_DVA,3.",
    },
    {
        "step_number": "9",
        "feature_key": "step9_log_abs_mean_delta_dQdV_w3_w0_3p0_3p6",
        "formula": "log(|mean((dQ/dV)_w3 - (dQ/dV)_w0, V in [3.0, 3.6])|)",
        "author_source": "mean_dqdv_dchg_low_3_0 -> log(abs(.))",
        "description": "Mean of low-voltage incremental-capacity delta.",
    },
    {
        "step_number": "10",
        "feature_key": "step10_log_abs_CV_time_w0",
        "formula": "log(|CV_time_w0|)",
        "author_source": "CV_time_0 -> log(abs(.))",
        "description": "Initial RPT CV hold time.",
    },
]


def read_dict_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_valid_cells(path: Path) -> List[str]:
    rows = read_dict_rows(path)
    return [str(row.get("Cell", "")).strip() for row in rows if str(row.get("Cell", "")).strip()]


def read_capacity_fade(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    time_week: List[float] = []
    capacity_ah: List[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row["Time"])
                q = float(row["Capacity"])
            except Exception:
                continue
            if np.isfinite(t) and np.isfinite(q):
                time_week.append(t)
                capacity_ah.append(q)
    return np.asarray(time_week, dtype=float), np.asarray(capacity_ah, dtype=float)


def read_q_interpolated(path: Path) -> np.ndarray:
    qmat = np.genfromtxt(path, delimiter=",", dtype=float)
    if qmat.ndim == 1:
        qmat = qmat.reshape(-1, 1)
    return qmat


def drop_all_nan_cols(qmat: np.ndarray) -> np.ndarray:
    if qmat.size == 0:
        return qmat
    keep = ~np.all(np.isnan(qmat), axis=0)
    return qmat[:, keep]


def parse_time_list(values: Sequence[str]) -> List[np.datetime64]:
    return [np.datetime64(v) for v in values]


def load_rpt_dict(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = json.loads(json.load(f))

    for i, start_time in enumerate(data["start_stop_time"]["start"]):
        if start_time != "[]":
            data["start_stop_time"]["start"][i] = np.datetime64(start_time)
            data["start_stop_time"]["stop"][i] = np.datetime64(data["start_stop_time"]["stop"][i])
        else:
            data["start_stop_time"]["start"][i] = []
            data["start_stop_time"]["stop"][i] = []

    for i in range(len(data["start_stop_time"]["start"])):
        for key in ["QV_charge_C_2", "QV_discharge_C_2", "QV_charge_C_5", "QV_discharge_C_5"]:
            data[key]["t"][i] = parse_time_list(data[key]["t"][i])
    return data


def resolve_release_paths(cell: str) -> Optional[Tuple[str, Path, Path, Path]]:
    for release in ("Release 1.0", "Release 2.0"):
        cap_path = IVAS_ROOT / "capacity_fade" / release / f"{cell}.csv"
        q_path = IVAS_ROOT / "Q_interpolated" / release / f"{cell}.csv"
        rpt_path = IVAS_ROOT / "RPT_json" / release / f"{cell}.json"
        if cap_path.exists() and q_path.exists() and rpt_path.exists():
            return release, cap_path, q_path, rpt_path
    return None


def load_dod_map(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in read_dict_rows(path):
        try:
            out[str(row["Cell"]).strip()] = float(row["DoD"])
        except Exception:
            continue
    return out


def load_condition_map(path: Path) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for row in read_dict_rows(path):
        try:
            group_num = int(float(row["Group#"]))
            out[group_num] = {
                "Cchg": float(row["Charging C-rate"]),
                "Cdchg": float(row["Discharging C-rate"]),
            }
        except Exception:
            continue
    return out


def safe_log_abs(x: float, eps: float = 1e-12) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(np.log(max(abs(x), eps)))


def week_to_rpt_idx(group_num: int, week_num: int) -> int:
    if group_num < 20:
        return int(week_num)
    if week_num == 0:
        return 0
    return int(week_num + 1)


def charge_cv_time_from_rpt_index(
    rpt_dict: Dict[str, object],
    rpt_idx: int,
    current_threshold: float = 0.0495,
) -> float:
    try:
        current = np.asarray(rpt_dict["QV_charge_C_5"]["I"][rpt_idx], dtype=float)
        time_arr = rpt_dict["QV_charge_C_5"]["t"][rpt_idx]
        if len(current) == 0 or len(time_arr) == 0:
            return float("nan")
        idx = int(np.where(current < float(current_threshold))[0][0])
        return float((time_arr[-1] - time_arr[idx]) / np.timedelta64(1, "s"))
    except Exception:
        return float("nan")


def local_extrema_indices(arr: np.ndarray, mode: str, order: int = 10) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size < 2 * order + 1:
        return np.asarray([], dtype=int)

    idxs: List[int] = []
    for i in range(order, arr.size - order):
        center = arr[i]
        if not np.isfinite(center):
            continue
        left = arr[i - order : i]
        right = arr[i + 1 : i + 1 + order]
        neigh = np.concatenate([left, right])
        if not np.all(np.isfinite(neigh)):
            continue
        if mode == "max":
            if np.all(center > left) and np.all(center >= right):
                idxs.append(i)
        elif mode == "min":
            if np.all(center < left) and np.all(center <= right):
                idxs.append(i)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return np.asarray(idxs, dtype=int)


def maybe_extend_last_peak(q: np.ndarray, dvdq: np.ndarray, max_idxs: np.ndarray) -> np.ndarray:
    if max_idxs.size < 2:
        return max_idxs
    if q[max_idxs[-1]] >= 0.19:
        return max_idxs

    tail = dvdq[max_idxs[1] :]
    if tail.size < 25:
        return max_idxs
    min_ind = local_extrema_indices(tail, mode="min", order=10)
    diff_arr = np.diff(tail)
    zero_diff_ind = np.where((diff_arr < 0.006) & (diff_arr > -0.006))[0]
    if zero_diff_ind.size == 0:
        return max_idxs

    chosen: Optional[int] = None
    if min_ind.size > 0 and not (zero_diff_ind[-1] > min_ind[-1] + 10):
        valid = zero_diff_ind[zero_diff_ind < min_ind[-1] - 10]
        if valid.size > 0:
            chosen = int(valid[-1])
    else:
        chosen = int(zero_diff_ind[-1])

    if chosen is None:
        return max_idxs
    return np.append(max_idxs, max_idxs[1] + chosen)


def choose_qne_index(q: np.ndarray, dvdq: np.ndarray, max_idxs: np.ndarray) -> Optional[int]:
    if max_idxs.size < 2:
        return None
    best = int(max_idxs[1])
    for ind in max_idxs:
        qi = q[int(ind)]
        if qi < 0.12 or qi > 0.18:
            continue
        if dvdq[int(ind)] > dvdq[best]:
            best = int(ind)
    return best


def dva_delta_features(
    q_ref_curve: np.ndarray,
    q_cur_curve: np.ndarray,
    dvdq_ref_curve: np.ndarray,
    dvdq_cur_curve: np.ndarray,
) -> Tuple[float, float]:
    y_lim = 6.0

    idx_ref = (q_ref_curve[:-1] <= 0.28) & (-dvdq_ref_curve <= y_lim) & (dvdq_ref_curve <= 0)
    idx_cur = (q_cur_curve[:-1] <= 0.28) & (-dvdq_cur_curve <= y_lim) & (dvdq_cur_curve <= 0)

    q_ref = q_ref_curve[:-1][idx_ref][::-1]
    q_cur = q_cur_curve[:-1][idx_cur][::-1]
    dvdq_ref = (-dvdq_ref_curve[idx_ref])[::-1]
    dvdq_cur = (-dvdq_cur_curve[idx_cur])[::-1]

    if q_ref.size < 40 or q_cur.size < 40:
        return float("nan"), float("nan")

    max_ref = local_extrema_indices(dvdq_ref, mode="max", order=10)
    max_cur = local_extrema_indices(dvdq_cur, mode="max", order=10)
    if max_ref.size < 2 or max_cur.size < 2:
        return float("nan"), float("nan")

    max_ref = maybe_extend_last_peak(q_ref, dvdq_ref, max_ref)
    max_cur = maybe_extend_last_peak(q_cur, dvdq_cur, max_cur)

    qne_ref = choose_qne_index(q_ref, dvdq_ref, max_ref)
    qne_cur = choose_qne_index(q_cur, dvdq_cur, max_cur)
    if qne_ref is None or qne_cur is None:
        return float("nan"), float("nan")

    delta_q1 = (q_ref[-1] - q_ref[max_ref[-1]]) - (q_cur[-1] - q_cur[max_cur[-1]])
    delta_q3 = (q_ref[-1] - q_ref[qne_ref]) - (q_cur[-1] - q_cur[qne_cur])
    return float(delta_q1), float(delta_q3)


def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.size
    if n == 2:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return np.asarray([slope, slope], dtype=float)

    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n, dtype=float)

    for k in range(1, n - 1):
        if delta[k - 1] == 0.0 or delta[k] == 0.0 or np.sign(delta[k - 1]) != np.sign(delta[k]):
            d[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    d0 = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if np.sign(d0) != np.sign(delta[0]):
        d0 = 0.0
    elif np.sign(delta[0]) != np.sign(delta[1]) and abs(d0) > abs(3.0 * delta[0]):
        d0 = 3.0 * delta[0]
    d[0] = d0

    dn = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if np.sign(dn) != np.sign(delta[-1]):
        dn = 0.0
    elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(dn) > abs(3.0 * delta[-1]):
        dn = 3.0 * delta[-1]
    d[-1] = dn

    return d


def pchip_interpolate_numpy(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xq = np.asarray(xq, dtype=float)
    d = pchip_slopes(x, y)

    idx = np.searchsorted(x, xq, side="right") - 1
    idx = np.clip(idx, 0, len(x) - 2)
    h = x[idx + 1] - x[idx]
    t = (xq - x[idx]) / h

    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    return (
        h00 * y[idx]
        + h10 * h * d[idx]
        + h01 * y[idx + 1]
        + h11 * h * d[idx + 1]
    )


def lifetime_from_capacity_fade_by_retention(
    time_week: np.ndarray,
    capacity_ah: np.ndarray,
    retention: float,
    step: float = PCHIP_STEP,
) -> float:
    mask = np.isfinite(time_week) & np.isfinite(capacity_ah)
    x = np.asarray(time_week[mask], dtype=float)
    y = np.asarray(capacity_ah[mask], dtype=float)
    if x.size < 2:
        return float("nan")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    unique_x, unique_idx = np.unique(x, return_index=True)
    x = unique_x
    y = y[unique_idx]
    if x.size < 2:
        return float("nan")

    finite_pos = y[np.isfinite(y) & (y > 0)]
    if finite_pos.size == 0:
        return float("nan")
    threshold = float(finite_pos[0]) * float(retention)

    x_grid = np.arange(0.0, np.ceil(np.max(x)), step, dtype=float)
    if x_grid.size == 0:
        x_grid = np.asarray([0.0], dtype=float)
    y_grid = pchip_interpolate_numpy(x, y, x_grid)
    if threshold < float(np.nanmin(y_grid)) or threshold > float(np.nanmax(y_grid)):
        return float("nan")
    life_idx = int(np.argmin(np.abs(y_grid - float(threshold))))
    return float(np.round(x_grid[life_idx], 3))


def compute_lifetime_columns(time_week: np.ndarray, capacity_ah: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for retention in RETENTION_LEVELS:
        pct = int(round(retention * 100))
        out[f"lifetime_weeks_EOL{pct}"] = lifetime_from_capacity_fade_by_retention(
            time_week=time_week,
            capacity_ah=capacity_ah,
            retention=retention,
        )
    return out


def finite_mean(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def finite_var(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.var(finite))


def build_feature_row(
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

    w0_idx = week_to_rpt_idx(group_num, 0)
    w3_idx = week_to_rpt_idx(group_num, 3)
    if qmat.shape[1] <= max(w0_idx, w3_idx):
        raise ValueError(f"Missing week0/week3 RPT columns for {cell}")
    if len(time_week) <= max(w0_idx, w3_idx):
        raise ValueError(f"Missing week0/week3 time entries for {cell}")

    v = np.linspace(V_MIN, V_MAX, N_VOLTAGE)
    dqdv = np.diff(qmat, axis=0) / np.diff(v)[:, None]
    dvdq = np.diff(v)[:, None] / np.diff(qmat, axis=0)
    # Match the author's masking grid in curve_difference_varying_window(...),
    # where dQ/dV windows are selected on linspace(3.0, 4.2, 999).
    dqdv_v = np.linspace(3.0, 4.2, dqdv.shape[0])

    low_mask = (dqdv_v >= 3.0) & (dqdv_v <= 3.6)
    mid_mask = (dqdv_v >= 3.6) & (dqdv_v <= 3.9)

    delta_mid = dqdv[mid_mask, w3_idx] - dqdv[mid_mask, w0_idx]
    delta_low = dqdv[low_mask, w3_idx] - dqdv[low_mask, w0_idx]

    cv_time_w0 = charge_cv_time_from_rpt_index(rpt_dict, w0_idx)
    cv_time_w3 = charge_cv_time_from_rpt_index(rpt_dict, w3_idx)
    delta_q1, delta_q3 = dva_delta_features(
        q_ref_curve=qmat[:, w0_idx],
        q_cur_curve=qmat[:, w3_idx],
        dvdq_ref_curve=dvdq[:, w0_idx],
        dvdq_cur_curve=dvdq[:, w3_idx],
    )

    lifetime_cols = compute_lifetime_columns(time_week=time_week, capacity_ah=capacity_ah)
    sqrt_stress = float(np.sqrt(max(cchg, 0.0)) * np.sqrt(max(dod, 0.0)))

    return {
        "feature_status": "ok",
        "group_num": group_num,
        "cell_idx": cell_idx,
        "cell": cell,
        "release": release,
        **lifetime_cols,
        "week0_rpt_idx": w0_idx,
        "week3_rpt_idx": w3_idx,
        "week0_time_week": float(time_week[w0_idx]) if np.isfinite(time_week[w0_idx]) else float("nan"),
        "week3_time_week": float(time_week[w3_idx]) if np.isfinite(time_week[w3_idx]) else float("nan"),
        "Q_initial_ah": float(capacity_ah[w0_idx]) if np.isfinite(capacity_ah[w0_idx]) else float("nan"),
        "step1_log_abs_mean_delta_dQdV_w3_w0_3p6_3p9": safe_log_abs(finite_mean(delta_mid)),
        "step2_log_abs_delta_CV_time_w3_w0": safe_log_abs(cv_time_w3 - cv_time_w0),
        "step3_DoD": float(dod),
        "step4_delta_Q1_DVA_w3_w0": delta_q1,
        "step5_sqrt_Cchg_sqrt_DoD": sqrt_stress,
        "step6_Cchg": float(cchg),
        "step7_log_abs_var_delta_dQdV_w3_w0_3p0_3p6": safe_log_abs(finite_var(delta_low)),
        "step8_delta_Q3_DVA_w3_w0": delta_q3,
        "step9_log_abs_mean_delta_dQdV_w3_w0_3p0_3p6": safe_log_abs(finite_mean(delta_low)),
        "step10_log_abs_CV_time_w0": safe_log_abs(cv_time_w0),
        "raw_mean_delta_dQdV_w3_w0_3p6_3p9": finite_mean(delta_mid),
        "raw_var_delta_dQdV_w3_w0_3p0_3p6": finite_var(delta_low),
        "raw_mean_delta_dQdV_w3_w0_3p0_3p6": finite_mean(delta_low),
        "raw_CV_time_w0_sec": cv_time_w0,
        "raw_CV_time_w3_sec": cv_time_w3,
        "chg_c_rate": float(cchg),
        "dchg_c_rate": float(cdchg),
    }


def build_partial_row_missing_week3(
    cell: str,
    release: str,
    time_week: np.ndarray,
    capacity_ah: np.ndarray,
    rpt_dict: Dict[str, object],
    dod: float,
    cchg: float,
    cdchg: float,
) -> Dict[str, object]:
    group_num = int(cell[1:-2])
    cell_idx = int(cell[-1])
    w0_idx = week_to_rpt_idx(group_num, 0)
    w3_idx = week_to_rpt_idx(group_num, 3)

    cv_time_w0 = charge_cv_time_from_rpt_index(rpt_dict, w0_idx) if len(rpt_dict["QV_charge_C_5"]["I"]) > w0_idx else float("nan")
    lifetime_cols = compute_lifetime_columns(time_week=time_week, capacity_ah=capacity_ah)
    sqrt_stress = float(np.sqrt(max(cchg, 0.0)) * np.sqrt(max(dod, 0.0)))

    return {
        "feature_status": "missing_week3",
        "group_num": group_num,
        "cell_idx": cell_idx,
        "cell": cell,
        "release": release,
        **lifetime_cols,
        "week0_rpt_idx": w0_idx,
        "week3_rpt_idx": w3_idx,
        "week0_time_week": float(time_week[w0_idx]) if len(time_week) > w0_idx and np.isfinite(time_week[w0_idx]) else float("nan"),
        "week3_time_week": float("nan"),
        "Q_initial_ah": float(capacity_ah[w0_idx]) if len(capacity_ah) > w0_idx and np.isfinite(capacity_ah[w0_idx]) else float("nan"),
        "step1_log_abs_mean_delta_dQdV_w3_w0_3p6_3p9": float("nan"),
        "step2_log_abs_delta_CV_time_w3_w0": float("nan"),
        "step3_DoD": float(dod),
        "step4_delta_Q1_DVA_w3_w0": float("nan"),
        "step5_sqrt_Cchg_sqrt_DoD": sqrt_stress,
        "step6_Cchg": float(cchg),
        "step7_log_abs_var_delta_dQdV_w3_w0_3p0_3p6": float("nan"),
        "step8_delta_Q3_DVA_w3_w0": float("nan"),
        "step9_log_abs_mean_delta_dQdV_w3_w0_3p0_3p6": float("nan"),
        "step10_log_abs_CV_time_w0": safe_log_abs(cv_time_w0),
        "raw_mean_delta_dQdV_w3_w0_3p6_3p9": float("nan"),
        "raw_var_delta_dQdV_w3_w0_3p0_3p6": float("nan"),
        "raw_mean_delta_dQdV_w3_w0_3p0_3p6": float("nan"),
        "raw_CV_time_w0_sec": cv_time_w0,
        "raw_CV_time_w3_sec": float("nan"),
        "chg_c_rate": float(cchg),
        "dchg_c_rate": float(cdchg),
    }


def sort_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            int(float(row["group_num"])),
            int(float(row["cell_idx"])),
            str(row["cell"]),
        ),
    )


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_feature_markdown(path: Path, specs: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# IVAS Lifetime 10-Feature Dictionary",
        "",
        "| Step | Feature Key | Formula | Author Source | Description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for spec in specs:
        lines.append(
            f"| {spec['step_number']} | {spec['feature_key']} | {spec['formula']} | "
            f"{spec['author_source']} | {spec['description']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_eol_availability_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    total_cells = len(rows)
    for retention in sorted(RETENTION_LEVELS):
        pct = int(round(retention * 100))
        col = f"lifetime_weeks_EOL{pct}"
        available = 0
        for row in rows:
            try:
                val = float(row[col])
            except Exception:
                val = float("nan")
            if np.isfinite(val):
                available += 1
        out.append(
            {
                "EOL_percent": pct,
                "lifetime_column": col,
                "available_cells": available,
                "missing_cells": total_cells - available,
                "total_cells": total_cells,
            }
        )
    return out


def main() -> None:
    valid_cells = read_valid_cells(IVAS_ROOT / "Valid_cells.csv")
    dod_map = load_dod_map(AUTHOR_FEAT_DIR / "empirical_DoD.csv")
    cond_map = load_condition_map(AUTHOR_FEAT_DIR / "cycling_conditions_wo_DoD.csv")

    rows: List[Dict[str, object]] = []
    skipped: List[str] = []

    for cell in valid_cells:
        resolved = resolve_release_paths(cell)
        if resolved is None:
            skipped.append(f"{cell}:missing_release")
            continue

        release, cap_path, q_path, rpt_path = resolved
        group_num = int(cell[1:-2])
        if cell not in dod_map or group_num not in cond_map:
            skipped.append(f"{cell}:missing_meta")
            continue

        time_week, capacity_ah = read_capacity_fade(cap_path)
        qmat = drop_all_nan_cols(read_q_interpolated(q_path))
        rpt_dict = load_rpt_dict(rpt_path)

        n = min(
            len(time_week),
            len(capacity_ah),
            qmat.shape[1],
            len(rpt_dict["QV_charge_C_5"]["I"]),
        )
        if n <= week_to_rpt_idx(group_num, 3):
            rows.append(
                build_partial_row_missing_week3(
                    cell=cell,
                    release=release,
                    time_week=time_week[:n],
                    capacity_ah=capacity_ah[:n],
                    rpt_dict=rpt_dict,
                    dod=float(dod_map[cell]),
                    cchg=float(cond_map[group_num]["Cchg"]),
                    cdchg=float(cond_map[group_num]["Cdchg"]),
                )
            )
            skipped.append(f"{cell}:insufficient_week3")
            continue

        time_week = time_week[:n]
        capacity_ah = capacity_ah[:n]
        qmat = qmat[:, :n]

        try:
            row = build_feature_row(
                cell=cell,
                release=release,
                time_week=time_week,
                capacity_ah=capacity_ah,
                qmat=qmat,
                rpt_dict=rpt_dict,
                dod=float(dod_map[cell]),
                cchg=float(cond_map[group_num]["Cchg"]),
                cdchg=float(cond_map[group_num]["Cdchg"]),
            )
        except Exception as exc:
            skipped.append(f"{cell}:{exc}")
            continue
        rows.append(row)

    rows = sort_rows(rows)

    out_dir = DEFAULT_OUT_DIR
    feature_csv = out_dir / "ivas_lifetime_10features_per_cell.csv"
    dict_csv = out_dir / "ivas_lifetime_10features_dictionary.csv"
    dict_md = out_dir / "ivas_lifetime_10features_dictionary.md"
    availability_csv = out_dir / "ivas_lifetime_eol_availability.csv"

    feature_fieldnames = [
        "feature_status",
        "group_num",
        "cell_idx",
        "cell",
        "release",
        "lifetime_weeks_EOL80",
        "lifetime_weeks_EOL75",
        "lifetime_weeks_EOL70",
        "lifetime_weeks_EOL65",
        "lifetime_weeks_EOL60",
        "lifetime_weeks_EOL55",
        "lifetime_weeks_EOL50",
        "lifetime_weeks_EOL45",
        "lifetime_weeks_EOL40",
        "lifetime_weeks_EOL35",
        "lifetime_weeks_EOL30",
        "week0_rpt_idx",
        "week3_rpt_idx",
        "week0_time_week",
        "week3_time_week",
        "Q_initial_ah",
        "step1_log_abs_mean_delta_dQdV_w3_w0_3p6_3p9",
        "step2_log_abs_delta_CV_time_w3_w0",
        "step3_DoD",
        "step4_delta_Q1_DVA_w3_w0",
        "step5_sqrt_Cchg_sqrt_DoD",
        "step6_Cchg",
        "step7_log_abs_var_delta_dQdV_w3_w0_3p0_3p6",
        "step8_delta_Q3_DVA_w3_w0",
        "step9_log_abs_mean_delta_dQdV_w3_w0_3p0_3p6",
        "step10_log_abs_CV_time_w0",
        "raw_mean_delta_dQdV_w3_w0_3p6_3p9",
        "raw_var_delta_dQdV_w3_w0_3p0_3p6",
        "raw_mean_delta_dQdV_w3_w0_3p0_3p6",
        "raw_CV_time_w0_sec",
        "raw_CV_time_w3_sec",
        "chg_c_rate",
        "dchg_c_rate",
    ]
    write_csv(feature_csv, rows, feature_fieldnames)
    write_csv(
        dict_csv,
        list(FEATURE_SPECS),
        ["step_number", "feature_key", "formula", "author_source", "description"],
    )
    write_csv(
        availability_csv,
        build_eol_availability_rows(rows),
        ["EOL_percent", "lifetime_column", "available_cells", "missing_cells", "total_cells"],
    )
    write_feature_markdown(dict_md, FEATURE_SPECS)

    print(f"[INFO] saved feature table : {feature_csv}")
    print(f"[INFO] saved dictionary    : {dict_csv}")
    print(f"[INFO] saved availability  : {availability_csv}")
    print(f"[INFO] saved markdown      : {dict_md}")
    print(f"[INFO] total rows          : {len(rows)}")
    print(f"[INFO] skipped cells       : {len(skipped)}")
    if skipped:
        print(f"[INFO] first skipped       : {skipped[:10]}")


if __name__ == "__main__":
    main()
