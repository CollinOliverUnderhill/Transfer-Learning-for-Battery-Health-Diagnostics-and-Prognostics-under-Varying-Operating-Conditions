#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
domain_train_test_by_group.py

Ridge-based domain train/test workflow for IVAS RPT-level SOH estimation.

This script is intended for the next step after single-cell / single-feature
baseline experiments:
  1. Train on batteries from one set of operating-condition groups.
  2. Optionally fine-tune on a few labeled target-domain groups.
  3. Evaluate on held-out batteries from the target domain.

It supports two split modes:
  - explicit:
      You directly specify train/test groups and/or train/test cells.
      This covers:
        a) A-domain train, B-domain test
        b) A-domain train, B-domain split into target fine-tune vs target test
        b) Same protocol group, different cells for train vs test
  - within_group_random:
      For selected groups, randomly split cells into train/test inside each group.

It accepts a long-table CSV supplied by --data_csv.
The default remains the single-feature CSV:
  - IVAS/Processing_Data/rpt_samples_feature_soh.csv

Example 1: cross-group / cross-condition test
---------------------------------------------
python E:\Datasets\IVAS\Codes\chunqiu_codes\domain_train_test_by_group.py `
  --split_mode explicit `
  --data_csv E:\Datasets\IVAS\Processing_Data\rpt_samples_feature_soh.csv `
  --x_cols feature_mean_ic `
  --train_groups 1,2,3,4 `
  --test_groups 5,6,7,8 `
  --out_dir E:\Datasets\IVAS\Ridge_Results_DomainShift\g1234_to_g5678 `
  --save_predictions

Example 2: same groups, but split cells inside each group
---------------------------------------------------------
python E:\Datasets\IVAS\Codes\chunqiu_codes\domain_train_test_by_group.py `
  --split_mode within_group_random `
  --data_csv E:\Datasets\IVAS\Processing_Data\rpt_samples_feature_soh.csv `
  --x_cols feature_mean_ic `
  --groups 1,2,3 `
  --test_cell_frac 0.5 `
  --cell_split_seed 42 `
  --out_dir E:\Datasets\IVAS\Ridge_Results_DomainShift\within_g123 `
  --save_predictions

Example 3: source-train, few-shot target fine-tune, target holdout test
-----------------------------------------------------------------------
python E:\Datasets\IVAS\Codes\chunqiu_codes\domain_train_test_by_group.py `
  --split_mode explicit `
  --data_csv E:\Datasets\IVAS\Processing_Data\rpt_samples_feature_soh.csv `
  --x_cols feature_mean_ic `
  --train_groups 1,2,3,4,5,6,7,8 `
  --test_groups 25,26,27,28,29,30 `
  --target_ft_group_count 2 `
  --target_ft_seed 42 `
  --out_dir E:\Datasets\IVAS\Ridge_Results_DomainShift\src_to_tgt_fewshot `
  --save_predictions
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


ALPHA_FIXED: float = 1e-3
CELL_RE = re.compile(r"^G(\d+)C(\d+)$")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mae_median(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_true - y_pred)
    return float(np.median(abs_err)) if abs_err.size else float("nan")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def mape_percent(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0)


def mape_percent_median(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    ape_percent = np.abs((y_pred - y_true) / denom) * 100.0
    return float(np.median(ape_percent)) if ape_percent.size else float("nan")


def smape_percent(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(200.0 * np.abs(y_pred - y_true) / denom))


def smape_percent_median(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    sape_percent = 200.0 * np.abs(y_pred - y_true) / denom
    return float(np.median(sape_percent)) if sape_percent.size else float("nan")


def wmape_percent(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = max(float(np.sum(np.abs(y_true))), eps)
    return float(np.sum(np.abs(y_pred - y_true)) / denom * 100.0)


def abs_error_quantile(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_pred - y_true)
    if abs_err.size == 0:
        return float("nan")
    return float(np.quantile(abs_err, q))


def time_weighted_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    t: Optional[np.ndarray],
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_pred - y_true)

    if t is None:
        return float(np.mean(abs_err)) if abs_err.size else float("nan")

    t = np.asarray(t, dtype=float)
    if abs_err.size < 2 or t.size < 2:
        return float(np.mean(abs_err)) if abs_err.size else float("nan")

    mask = np.isfinite(t)
    if np.count_nonzero(mask) < 2:
        return float(np.mean(abs_err)) if abs_err.size else float("nan")

    t = t[mask]
    abs_err = abs_err[mask]

    order = np.argsort(t)
    t = t[order]
    abs_err = abs_err[order]

    span = float(t[-1] - t[0])
    if span <= 0:
        return float(np.mean(abs_err)) if abs_err.size else float("nan")

    dt = np.diff(t)
    integral = float(np.sum(0.5 * (abs_err[:-1] + abs_err[1:]) * dt))
    return float(integral / span)


def describe_errors(y_true: np.ndarray, y_pred: np.ndarray, tail_q: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_pred - y_true)
    denom = np.where(np.abs(y_true) < 1e-12, 1e-12, np.abs(y_true))
    ape_percent = np.abs((y_pred - y_true) / denom) * 100.0
    sape_denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-12)
    sape_percent = 200.0 * abs_err / sape_denom
    qs = np.quantile(abs_err, [0.5, 0.9, 0.95, 0.99]) if abs_err.size else [np.nan] * 4
    out = {
        "n": float(len(y_true)),
        "mae": mae(y_true, y_pred) if len(y_true) else float("nan"),
        "mae_mean": mae(y_true, y_pred) if len(y_true) else float("nan"),
        "mae_median": float(qs[0]),
        "rmse": rmse(y_true, y_pred) if len(y_true) else float("nan"),
        "r2": r2_score(y_true, y_pred) if len(y_true) else float("nan"),
        "mape_percent": mape_percent(y_true, y_pred) if len(y_true) else float("nan"),
        "mape_percent_mean": mape_percent(y_true, y_pred) if len(y_true) else float("nan"),
        "mape_percent_median": float(np.median(ape_percent)) if ape_percent.size else float("nan"),
        "mdape_percent": float(np.median(ape_percent)) if ape_percent.size else float("nan"),
        "smape_percent": smape_percent(y_true, y_pred) if len(y_true) else float("nan"),
        "smape_percent_mean": smape_percent(y_true, y_pred) if len(y_true) else float("nan"),
        "smape_percent_median": float(np.median(sape_percent)) if sape_percent.size else float("nan"),
        "wmape_percent": wmape_percent(y_true, y_pred) if len(y_true) else float("nan"),
        "abs_err_median": float(qs[0]),
        "abs_err_p90": float(qs[1]),
        "abs_err_p95": float(qs[2]),
        "abs_err_p99": float(qs[3]),
        f"abs_err_p{int(tail_q * 100)}": abs_error_quantile(y_true, y_pred, tail_q),
    }
    return out


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd <= 0, 1.0, sd)
    return mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return (X - mu) / sd


def ridge_fit_with_prior(
    X_s: np.ndarray,
    y: np.ndarray,
    alpha: float,
    beta_prior: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")

    X_s = np.asarray(X_s, dtype=float)
    y = np.asarray(y, dtype=float)
    p = X_s.shape[1]
    prior = np.zeros(p, dtype=float) if beta_prior is None else np.asarray(beta_prior, dtype=float)
    if prior.shape != (p,):
        raise ValueError(f"beta_prior shape mismatch: expected {(p,)}, got {prior.shape}")

    Z = np.column_stack([np.ones(X_s.shape[0], dtype=float), X_s])
    penalty = np.eye(p + 1, dtype=float)
    penalty[0, 0] = 0.0  # Keep the intercept unregularized.
    prior_full = np.concatenate([[0.0], prior])

    A = Z.T @ Z + alpha * penalty
    b = Z.T @ y + alpha * (penalty @ prior_full)
    theta = np.linalg.solve(A, b)
    intercept = float(theta[0])
    beta_s = np.asarray(theta[1:], dtype=float)
    return beta_s, intercept


def ridge_predict(X_s: np.ndarray, beta_s: np.ndarray, intercept: float) -> np.ndarray:
    X_s = np.asarray(X_s, dtype=float)
    beta_s = np.asarray(beta_s, dtype=float)
    return float(intercept) + X_s @ beta_s


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_percent_to_float(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    if s.endswith("%"):
        try:
            return float(s[:-1].strip())
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_str_list(arg: str) -> List[str]:
    s = str(arg).strip()
    if not s:
        return []
    if s.lower() in {"all", "*"}:
        return ["all"]
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_list(arg: str) -> List[int]:
    vals = parse_str_list(arg)
    if vals == ["all"]:
        return []
    out: List[int] = []
    for v in vals:
        out.append(int(v))
    return out


def parse_release_filter(arg: str) -> Optional[Set[str]]:
    vals = parse_str_list(arg)
    if not vals or vals == ["all"]:
        return None
    return {v for v in vals}


def infer_group_num(cell: str) -> Optional[int]:
    m = CELL_RE.match(str(cell).strip())
    if not m:
        return None
    return int(m.group(1))


def infer_cell_idx(cell: str) -> Optional[int]:
    m = CELL_RE.match(str(cell).strip())
    if not m:
        return None
    return int(m.group(2))


def cell_sort_key(cell: str) -> Tuple[int, int, str]:
    g = infer_group_num(cell)
    c = infer_cell_idx(cell)
    if g is None or c is None:
        return (10**9, 10**9, str(cell))
    return (g, c, str(cell))


def autodetect_feature_cols(df: pd.DataFrame) -> List[str]:
    feature_cols = [c for c in df.columns if isinstance(c, str) and re.match(r"^f\d+_", c)]
    if feature_cols:
        feature_cols = sorted(
            feature_cols,
            key=lambda name: (
                int(re.match(r"^f(\d+)_", name).group(1)) if re.match(r"^f(\d+)_", name) else 10**9,
                name,
            ),
        )
        return feature_cols

    fallback = []
    for cand in ["feature_mean_ic", "feature_mean_ic_delta"]:
        if cand in df.columns:
            fallback.append(cand)
    return fallback


def load_group_conditions(group_cond_csv: Path) -> pd.DataFrame:
    if not group_cond_csv.exists():
        raise FileNotFoundError(f"Group condition CSV not found: {group_cond_csv}")

    df = pd.read_csv(group_cond_csv)
    required = ["Group", "Charging C-rate", "Discharging C-rate", "Mean DoD"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"[Groupcondi.csv] missing column '{c}'. Found: {list(df.columns)}")

    out = df.copy()
    out["group_num"] = pd.to_numeric(out["Group"], errors="coerce")
    out["charging_crate"] = pd.to_numeric(out["Charging C-rate"], errors="coerce")
    out["discharging_crate"] = pd.to_numeric(out["Discharging C-rate"], errors="coerce")
    out["dod_pct"] = out["Mean DoD"].apply(parse_percent_to_float)
    if "Mean Lifetime [weeks]" in out.columns:
        out["lifetime_week"] = pd.to_numeric(out["Mean Lifetime [weeks]"], errors="coerce")
    else:
        out["lifetime_week"] = np.nan
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["group_num"]).copy()
    out["group_num"] = out["group_num"].astype(int)
    return out[
        [
            "group_num",
            "charging_crate",
            "discharging_crate",
            "dod_pct",
            "lifetime_week",
        ]
    ].drop_duplicates(subset=["group_num"])


def read_samples_csv(
    data_csv: Path,
    x_cols_arg: str,
    y_col: str,
    sort_col: str,
    time_col: str,
) -> Tuple[pd.DataFrame, List[str]]:
    if not data_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {data_csv}")

    df = pd.read_csv(data_csv)
    if "cell" not in df.columns:
        raise ValueError(f'Input CSV must contain column "cell". Found: {list(df.columns)}')
    if "release" not in df.columns:
        raise ValueError(f'Input CSV must contain column "release". Found: {list(df.columns)}')
    if "rpt_idx" not in df.columns:
        raise ValueError(f'Input CSV must contain column "rpt_idx". Found: {list(df.columns)}')
    if y_col not in df.columns:
        raise ValueError(f'Input CSV missing target column "{y_col}".')
    if sort_col not in df.columns:
        raise ValueError(f'Input CSV missing sort column "{sort_col}".')

    x_cols_arg = str(x_cols_arg).strip()
    if x_cols_arg.lower() == "auto":
        x_cols = autodetect_feature_cols(df)
        if not x_cols:
            raise ValueError(
                'x_cols="auto" found no usable feature columns. Expected '
                '"feature_mean_ic" or columns like "f1_...", "f2_...".'
            )
    else:
        x_cols = [c.strip() for c in x_cols_arg.split(",") if c.strip()]
        missing = [c for c in x_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Input CSV missing requested feature columns: {missing}")

    df = df.copy()
    df["cell"] = df["cell"].astype(str).str.strip()
    df["release"] = df["release"].astype(str).str.strip()
    df["rpt_idx"] = pd.to_numeric(df["rpt_idx"], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df[sort_col] = pd.to_numeric(df[sort_col], errors="coerce")
    if time_col in df.columns:
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    for c in x_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["group_num"] = df["cell"].apply(infer_group_num)
    df["cell_idx"] = df["cell"].apply(infer_cell_idx)

    need = ["cell", "release", "rpt_idx", y_col, sort_col, "group_num"]
    if time_col in df.columns:
        need.append(time_col)
    need.extend(x_cols)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=need).copy()
    df["rpt_idx"] = df["rpt_idx"].astype(int)
    df["group_num"] = df["group_num"].astype(int)
    if df["cell_idx"].notna().all():
        df["cell_idx"] = df["cell_idx"].astype(int)

    return df, x_cols


def apply_release_filter(df: pd.DataFrame, releases: Optional[Set[str]]) -> pd.DataFrame:
    if releases is None:
        return df.copy()
    return df[df["release"].isin(releases)].copy()


def build_cell_table(df: pd.DataFrame) -> pd.DataFrame:
    gb = df.groupby(["cell", "group_num", "release"], dropna=False)
    out = gb.agg(
        n_rows=("rpt_idx", "size"),
        min_rpt_idx=("rpt_idx", "min"),
        max_rpt_idx=("rpt_idx", "max"),
    ).reset_index()
    out["cell_idx"] = out["cell"].apply(infer_cell_idx)
    return out.sort_values(["group_num", "cell_idx", "cell"]).reset_index(drop=True)


def filter_cells_by_min_rows(cell_df: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    return cell_df[cell_df["n_rows"] >= int(min_rows)].copy()


def resolve_explicit_split(
    train_cell_df: pd.DataFrame,
    test_cell_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[List[str], List[str], Dict[str, object]]:
    train_groups = parse_int_list(args.train_groups)
    test_groups = parse_int_list(args.test_groups)
    train_cells_req = parse_str_list(args.train_cells)
    test_cells_req = parse_str_list(args.test_cells)

    if train_cells_req == ["all"]:
        train_cells_req = sorted(train_cell_df["cell"].unique().tolist(), key=cell_sort_key)
    if test_cells_req == ["all"]:
        test_cells_req = sorted(test_cell_df["cell"].unique().tolist(), key=cell_sort_key)

    train_pool = train_cell_df.copy()
    test_pool = test_cell_df.copy()

    if train_groups:
        train_pool = train_pool[train_pool["group_num"].isin(train_groups)].copy()
    if test_groups:
        test_pool = test_pool[test_pool["group_num"].isin(test_groups)].copy()

    if train_cells_req:
        train_pool = train_pool[train_pool["cell"].isin(train_cells_req)].copy()
    if test_cells_req:
        test_pool = test_pool[test_pool["cell"].isin(test_cells_req)].copy()

    train_cells = sorted(train_pool["cell"].unique().tolist(), key=cell_sort_key)
    test_cells = sorted(test_pool["cell"].unique().tolist(), key=cell_sort_key)

    overlap = sorted(set(train_cells) & set(test_cells), key=cell_sort_key)
    if overlap:
        raise ValueError(
            "Train/test cells overlap, which would cause leakage. "
            f"Overlapping cells: {overlap}"
        )

    if not train_cells:
        raise ValueError("Explicit split produced zero train cells. Check --train_groups/--train_cells.")
    if not test_cells:
        raise ValueError("Explicit split produced zero test cells. Check --test_groups/--test_cells.")

    info = {
        "split_mode": "explicit",
        "train_groups_arg": train_groups,
        "test_groups_arg": test_groups,
        "train_cells_arg": train_cells_req,
        "test_cells_arg": test_cells_req,
    }
    return train_cells, test_cells, info


def resolve_within_group_random_split(
    cell_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[List[str], List[str], Dict[str, object]]:
    groups = parse_int_list(args.groups)
    pool = cell_df.copy()
    if groups:
        pool = pool[pool["group_num"].isin(groups)].copy()
    if len(pool) == 0:
        raise ValueError("No eligible cells found for --groups after filtering.")

    rng = np.random.default_rng(int(args.cell_split_seed))
    test_frac = float(args.test_cell_frac)
    if not (0.0 < test_frac < 1.0):
        raise ValueError("--test_cell_frac must be in (0, 1).")

    train_cells: List[str] = []
    test_cells: List[str] = []
    group_rows: List[Dict[str, object]] = []

    for group_num, sub in pool.groupby("group_num", dropna=False):
        cells = sorted(sub["cell"].unique().tolist(), key=cell_sort_key)
        if len(cells) < 2:
            raise ValueError(
                f"Group {group_num} has only {len(cells)} eligible cell(s); cannot split into train/test."
            )

        cells_arr = np.array(cells, dtype=object)
        rng.shuffle(cells_arr)

        n_test = int(round(len(cells_arr) * test_frac))
        n_test = max(1, min(n_test, len(cells_arr) - 1))
        test_now = sorted(cells_arr[:n_test].tolist(), key=cell_sort_key)
        train_now = sorted(cells_arr[n_test:].tolist(), key=cell_sort_key)

        train_cells.extend(train_now)
        test_cells.extend(test_now)
        group_rows.append(
            {
                "group_num": int(group_num),
                "n_cells_total": int(len(cells)),
                "n_train_cells": int(len(train_now)),
                "n_test_cells": int(len(test_now)),
                "train_cells": train_now,
                "test_cells": test_now,
            }
        )

    info = {
        "split_mode": "within_group_random",
        "groups_arg": groups,
        "test_cell_frac": test_frac,
        "cell_split_seed": int(args.cell_split_seed),
        "per_group_split": group_rows,
    }
    return sorted(train_cells, key=cell_sort_key), sorted(test_cells, key=cell_sort_key), info


def resolve_target_finetune_split(
    test_cell_df: pd.DataFrame,
    train_cells: Sequence[str],
    candidate_test_cells: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[List[str], List[str], Dict[str, object]]:
    holdout_pool = test_cell_df[test_cell_df["cell"].isin(candidate_test_cells)].copy()
    if len(holdout_pool) == 0:
        raise ValueError("Target-domain candidate test pool is empty; cannot build fine-tune/test split.")

    ft_groups_explicit = parse_int_list(args.target_ft_groups)
    ft_group_count = int(args.target_ft_group_count)
    if ft_groups_explicit and ft_group_count > 0:
        raise ValueError("Use either --target_ft_groups or --target_ft_group_count, not both.")

    train_cell_set = set(train_cells)
    holdout_cell_set = set(candidate_test_cells)
    ft_pool = test_cell_df[
        (~test_cell_df["cell"].isin(train_cell_set)) &
        (~test_cell_df["cell"].isin(holdout_cell_set))
    ].copy()

    holdout_groups = sorted(holdout_pool["group_num"].unique().tolist())
    available_ft_groups = sorted(ft_pool["group_num"].unique().tolist())
    if ft_groups_explicit:
        overlap = sorted(set(ft_groups_explicit) & set(holdout_groups))
        if overlap:
            raise ValueError(
                "--target_ft_groups must be disjoint from --test_groups. "
                f"Overlapping groups: {overlap}"
            )
        missing = sorted(set(ft_groups_explicit) - set(available_ft_groups))
        if missing:
            raise ValueError(
                "--target_ft_groups must be chosen from eligible groups outside both train and explicit test pools. "
                f"Missing groups: {missing}"
            )
        ft_groups = sorted(ft_groups_explicit)
        selection_mode = "explicit_groups"
    elif ft_group_count > 0:
        if ft_group_count > len(available_ft_groups):
            raise ValueError(
                "--target_ft_group_count exceeds the number of eligible fine-tune groups outside train/test pools "
                f"({len(available_ft_groups)})."
            )
        rng = np.random.default_rng(int(args.target_ft_seed))
        ft_groups = sorted(rng.choice(np.asarray(available_ft_groups), size=ft_group_count, replace=False).tolist())
        selection_mode = "random_group_count"
    else:
        return [], sorted(candidate_test_cells, key=cell_sort_key), {
            "target_ft_enabled": False,
            "selection_mode": "disabled",
            "candidate_target_test_groups": holdout_groups,
            "candidate_target_ft_groups": available_ft_groups,
        }

    ft_group_set = set(ft_groups)
    ft_cells = sorted(ft_pool[ft_pool["group_num"].isin(ft_group_set)]["cell"].unique().tolist(), key=cell_sort_key)
    holdout_cells = sorted(candidate_test_cells, key=cell_sort_key)
    if not ft_cells:
        raise ValueError("Target fine-tune split produced zero fine-tune cells.")
    if not holdout_cells or not holdout_groups:
        raise ValueError("Target fine-tune split produced zero holdout target cells/groups.")

    info = {
        "target_ft_enabled": True,
        "selection_mode": selection_mode,
        "candidate_target_test_groups": holdout_groups,
        "candidate_target_ft_groups": available_ft_groups,
        "target_ft_groups": ft_groups,
        "target_test_groups": holdout_groups,
        "target_ft_group_count": int(len(ft_groups)),
        "target_test_group_count": int(len(holdout_groups)),
        "target_ft_seed": int(args.target_ft_seed),
    }
    return ft_cells, holdout_cells, info


def materialize_cells_frame(
    df_all: pd.DataFrame,
    cells: Sequence[str],
    min_rows: int,
    split_name_for_error: str,
    allow_empty: bool = False,
) -> pd.DataFrame:
    cells_sorted = sorted(set(cells), key=cell_sort_key)
    if not cells_sorted:
        if allow_empty:
            return df_all.iloc[0:0].copy()
        raise ValueError(f"{split_name_for_error} cells are empty after applying split.")

    out = df_all[df_all["cell"].isin(cells_sorted)].copy()
    counts = out.groupby("cell")["rpt_idx"].size().to_dict()
    bad_cells = sorted([c for c in cells_sorted if counts.get(c, 0) < min_rows], key=cell_sort_key)
    if bad_cells:
        raise ValueError(f"{split_name_for_error} cells below required min_rows={min_rows}: {bad_cells}")
    if len(out) == 0 and not allow_empty:
        raise ValueError(f"{split_name_for_error} dataframe is empty after applying split.")
    return out


def add_condition_columns(df: pd.DataFrame, cond_df: pd.DataFrame) -> pd.DataFrame:
    return df.merge(cond_df, on="group_num", how="left")


def fit_ridge_model(train_df: pd.DataFrame, x_cols: Sequence[str], y_col: str, alpha: float) -> Dict[str, object]:
    X_train = train_df[list(x_cols)].to_numpy(dtype=float)
    y_train = train_df[y_col].to_numpy(dtype=float)
    mu, sd = standardize_fit(X_train)
    X_train_s = standardize_apply(X_train, mu, sd)
    beta_s, intercept = ridge_fit_with_prior(X_train_s, y_train, alpha=alpha, beta_prior=None)
    y_pred_train = ridge_predict(X_train_s, beta_s, intercept)
    return {
        "mu": mu,
        "sd": sd,
        "beta_s": beta_s,
        "intercept": intercept,
        "y_pred_train": y_pred_train,
    }


def finetune_ridge_model(
    finetune_df: pd.DataFrame,
    x_cols: Sequence[str],
    y_col: str,
    alpha: float,
    base_model: Dict[str, object],
) -> Dict[str, object]:
    X_ft = finetune_df[list(x_cols)].to_numpy(dtype=float)
    y_ft = finetune_df[y_col].to_numpy(dtype=float)
    mu = np.asarray(base_model["mu"], dtype=float)
    sd = np.asarray(base_model["sd"], dtype=float)
    beta_prior = np.asarray(base_model["beta_s"], dtype=float)
    X_ft_s = standardize_apply(X_ft, mu, sd)
    beta_s, intercept = ridge_fit_with_prior(X_ft_s, y_ft, alpha=alpha, beta_prior=beta_prior)
    y_pred_ft = ridge_predict(X_ft_s, beta_s, intercept)
    return {
        "mu": mu,
        "sd": sd,
        "beta_s": beta_s,
        "intercept": intercept,
        "y_pred_train": y_pred_ft,
    }


def predict_df(
    df: pd.DataFrame,
    x_cols: Sequence[str],
    mu: np.ndarray,
    sd: np.ndarray,
    beta_s: np.ndarray,
    intercept: float,
) -> np.ndarray:
    X = df[list(x_cols)].to_numpy(dtype=float)
    X_s = standardize_apply(X, mu, sd)
    return ridge_predict(X_s, beta_s, intercept)


def build_prediction_df(
    df: pd.DataFrame,
    y_col: str,
    y_pred: np.ndarray,
    split_name: str,
) -> pd.DataFrame:
    out = df.copy()
    out["split"] = str(split_name)
    out["y_true"] = out[y_col].to_numpy(dtype=float)
    out["y_pred"] = np.asarray(y_pred, dtype=float)
    out["abs_err"] = np.abs(out["y_pred"] - out["y_true"])
    denom = np.where(np.abs(out["y_true"].to_numpy(dtype=float)) < 1e-12, 1e-12, np.abs(out["y_true"].to_numpy(dtype=float)))
    out["ape_percent"] = np.abs((out["y_pred"].to_numpy(dtype=float) - out["y_true"].to_numpy(dtype=float)) / denom) * 100.0
    smape_denom = np.maximum(np.abs(out["y_true"].to_numpy(dtype=float)) + np.abs(out["y_pred"].to_numpy(dtype=float)), 1e-12)
    out["smape_percent"] = 200.0 * out["abs_err"].to_numpy(dtype=float) / smape_denom
    return out


def summarize_overall(
    pred_df: pd.DataFrame,
    split_name: str,
    tail_q: float,
) -> pd.DataFrame:
    stats = describe_errors(
        pred_df["y_true"].to_numpy(dtype=float),
        pred_df["y_pred"].to_numpy(dtype=float),
        tail_q=tail_q,
    )
    row = {
        "split": str(split_name),
        "n_rows": int(len(pred_df)),
        "n_cells": int(pred_df["cell"].nunique()),
        "n_groups": int(pred_df["group_num"].nunique()),
        "mae": float(stats["mae"]),
        "mae_mean": float(stats["mae_mean"]),
        "mae_median": float(stats["mae_median"]),
        "rmse": float(stats["rmse"]),
        "r2": float(stats["r2"]),
        "mape_percent": float(stats["mape_percent"]),
        "mape_percent_mean": float(stats["mape_percent_mean"]),
        "mape_percent_median": float(stats["mape_percent_median"]),
        "mdape_percent": float(stats["mdape_percent"]),
        "smape_percent": float(stats["smape_percent"]),
        "smape_percent_mean": float(stats["smape_percent_mean"]),
        "smape_percent_median": float(stats["smape_percent_median"]),
        "wmape_percent": float(stats["wmape_percent"]),
        "abs_err_median": float(stats["abs_err_median"]),
        "abs_err_p90": float(stats["abs_err_p90"]),
        "abs_err_p95": float(stats["abs_err_p95"]),
        "abs_err_p99": float(stats["abs_err_p99"]),
        f"abs_err_p{int(tail_q * 100)}": float(stats[f"abs_err_p{int(tail_q * 100)}"]),
    }
    return pd.DataFrame([row])


def summarize_by_cell(
    pred_df: pd.DataFrame,
    sort_col: str,
    time_col: str,
    tail_q: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for cell, sub in pred_df.groupby("cell", dropna=False):
        sub = sub.sort_values(sort_col).reset_index(drop=True)
        y_true = sub["y_true"].to_numpy(dtype=float)
        y_pred = sub["y_pred"].to_numpy(dtype=float)
        t = sub[time_col].to_numpy(dtype=float) if time_col in sub.columns else None
        stats = describe_errors(y_true, y_pred, tail_q=tail_q)
        rows.append(
            {
                "cell": str(cell),
                "group_num": int(sub["group_num"].iloc[0]),
                "release": str(sub["release"].iloc[0]),
                "charging_crate": float(sub["charging_crate"].iloc[0]) if pd.notna(sub["charging_crate"].iloc[0]) else np.nan,
                "discharging_crate": float(sub["discharging_crate"].iloc[0]) if pd.notna(sub["discharging_crate"].iloc[0]) else np.nan,
                "dod_pct": float(sub["dod_pct"].iloc[0]) if pd.notna(sub["dod_pct"].iloc[0]) else np.nan,
                "n_eval": int(len(sub)),
                "mae": float(stats["mae"]),
                "mae_mean": float(stats["mae_mean"]),
                "mae_median": float(stats["mae_median"]),
                "rmse": float(stats["rmse"]),
                "r2": float(stats["r2"]),
                "mape_percent": float(stats["mape_percent"]),
                "mape_percent_mean": float(stats["mape_percent_mean"]),
                "mape_percent_median": float(stats["mape_percent_median"]),
                "mdape_percent": float(stats["mdape_percent"]),
                "smape_percent": float(stats["smape_percent"]),
                "smape_percent_mean": float(stats["smape_percent_mean"]),
                "smape_percent_median": float(stats["smape_percent_median"]),
                "wmape_percent": float(stats["wmape_percent"]),
                "wmae_time_weighted": float(time_weighted_mae(y_true, y_pred, t=t)),
                "abs_err_median": float(stats["abs_err_median"]),
                "abs_err_p90": float(stats["abs_err_p90"]),
                "abs_err_p95": float(stats["abs_err_p95"]),
                "abs_err_p99": float(stats["abs_err_p99"]),
                f"abs_err_p{int(tail_q * 100)}": float(stats[f"abs_err_p{int(tail_q * 100)}"]),
            }
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values(["group_num", "cell"], key=lambda col: col.map(cell_sort_key) if col.name == "cell" else col).reset_index(drop=True)


def summarize_by_group(
    pred_df: pd.DataFrame,
    cell_metrics_df: pd.DataFrame,
    tail_q: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    tail_name = f"abs_err_p{int(tail_q * 100)}"

    for group_num, sub_pred in pred_df.groupby("group_num", dropna=False):
        y_true = sub_pred["y_true"].to_numpy(dtype=float)
        y_pred = sub_pred["y_pred"].to_numpy(dtype=float)
        row_stats = describe_errors(y_true, y_pred, tail_q=tail_q)

        sub_cell = cell_metrics_df[cell_metrics_df["group_num"] == group_num].copy()
        first = sub_pred.iloc[0]

        rows.append(
            {
                "group_num": int(group_num),
                "charging_crate": float(first["charging_crate"]) if pd.notna(first["charging_crate"]) else np.nan,
                "discharging_crate": float(first["discharging_crate"]) if pd.notna(first["discharging_crate"]) else np.nan,
                "dod_pct": float(first["dod_pct"]) if pd.notna(first["dod_pct"]) else np.nan,
                "n_rows": int(len(sub_pred)),
                "n_cells": int(sub_pred["cell"].nunique()),
                "row_mae": float(row_stats["mae"]),
                "row_mae_mean": float(row_stats["mae_mean"]),
                "row_mae_median": float(row_stats["mae_median"]),
                "row_rmse": float(row_stats["rmse"]),
                "row_r2": float(row_stats["r2"]),
                "row_mape_percent": float(row_stats["mape_percent"]),
                "row_mape_percent_mean": float(row_stats["mape_percent_mean"]),
                "row_mape_percent_median": float(row_stats["mape_percent_median"]),
                "row_mdape_percent": float(row_stats["mdape_percent"]),
                "row_smape_percent": float(row_stats["smape_percent"]),
                "row_smape_percent_mean": float(row_stats["smape_percent_mean"]),
                "row_smape_percent_median": float(row_stats["smape_percent_median"]),
                "row_wmape_percent": float(row_stats["wmape_percent"]),
                "row_abs_err_median": float(row_stats["abs_err_median"]),
                "row_abs_err_p95": float(row_stats["abs_err_p95"]),
                f"row_{tail_name}": float(row_stats[tail_name]),
                "cell_mae_mean": float(sub_cell["mae"].mean()) if len(sub_cell) else np.nan,
                "cell_mae_median": float(sub_cell["mae"].median()) if len(sub_cell) else np.nan,
                "cell_mape_mean": float(sub_cell["mape_percent"].mean()) if len(sub_cell) else np.nan,
                "cell_mape_median": float(sub_cell["mape_percent"].median()) if len(sub_cell) else np.nan,
                "cell_smape_mean": float(sub_cell["smape_percent"].mean()) if len(sub_cell) else np.nan,
                "cell_smape_median": float(sub_cell["smape_percent"].median()) if len(sub_cell) else np.nan,
                "cell_wmape_mean": float(sub_cell["wmape_percent"].mean()) if len(sub_cell) else np.nan,
                "cell_wmape_median": float(sub_cell["wmape_percent"].median()) if len(sub_cell) else np.nan,
                "cell_wmae_mean": float(sub_cell["wmae_time_weighted"].mean()) if len(sub_cell) else np.nan,
                "cell_wmae_median": float(sub_cell["wmae_time_weighted"].median()) if len(sub_cell) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values("group_num").reset_index(drop=True)


def maybe_make_plots(
    train_pred_df: pd.DataFrame,
    target_ft_pred_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    cell_metrics_df: pd.DataFrame,
    group_metrics_df: pd.DataFrame,
    overall_metrics_df: pd.DataFrame,
    out_dir: Path,
    *,
    target_name: str = "SOH",
    train_scatter_label: str = "Train",
    train_summary_title: str = "Source Train Summary",
    train_summary_filename: str = "plot_train_summary_metrics.png",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[WARN] matplotlib is not installed. Skip plot generation.")
        return

    def _save_summary_card(summary_pred_df: pd.DataFrame, title: str, out_name: str) -> None:
        if len(summary_pred_df) == 0:
            return
        stats = describe_errors(
            summary_pred_df["y_true"].to_numpy(dtype=float),
            summary_pred_df["y_pred"].to_numpy(dtype=float),
            tail_q=0.95,
        )
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        ax.axis("off")
        summary_text = (
            f"{title}\n"
            f"Rows: {int(len(summary_pred_df))}    Cells: {int(summary_pred_df['cell'].nunique())}    Groups: {int(summary_pred_df['group_num'].nunique())}\n"
            f"MAE mean   : {float(stats['mae_mean']):.6f}\n"
            f"MAE median : {float(stats['mae_median']):.6f}\n"
            f"RMSE       : {float(stats['rmse']):.6f}\n"
            f"R2         : {float(stats['r2']):.6f}\n"
            f"MAPE mean  : {float(stats['mape_percent_mean']):.3f}%\n"
            f"MdAPE      : {float(stats['mdape_percent']):.3f}%\n"
            f"SMAPE mean : {float(stats['smape_percent_mean']):.3f}%\n"
            f"WMAPE      : {float(stats['wmape_percent']):.3f}%"
        )
        ax.text(0.02, 0.95, summary_text, va="top", ha="left", fontsize=12, family="monospace")
        fig.tight_layout()
        fig.savefig(out_dir / out_name, dpi=200)
        plt.close(fig)

    if len(pred_df) > 0:
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        if len(train_pred_df) > 0:
            ax.scatter(
                train_pred_df["y_true"],
                train_pred_df["y_pred"],
                s=10,
                alpha=0.45,
                color="#F58518",
                label=str(train_scatter_label),
            )
        if len(target_ft_pred_df) > 0:
            ax.scatter(
                target_ft_pred_df["y_true"],
                target_ft_pred_df["y_pred"],
                s=14,
                alpha=0.75,
                color="#54A24B",
                label="Target fine-tune",
            )
        ax.scatter(
            pred_df["y_true"],
            pred_df["y_pred"],
            s=10,
            alpha=0.55,
            color="#4C78A8",
            label="Test",
        )
        lo = float(
            min(
                pred_df["y_true"].min(),
                pred_df["y_pred"].min(),
                train_pred_df["y_true"].min() if len(train_pred_df) > 0 else np.inf,
                train_pred_df["y_pred"].min() if len(train_pred_df) > 0 else np.inf,
            )
        )
        hi = float(
            max(
                pred_df["y_true"].max(),
                pred_df["y_pred"].max(),
                train_pred_df["y_true"].max() if len(train_pred_df) > 0 else -np.inf,
                train_pred_df["y_pred"].max() if len(train_pred_df) > 0 else -np.inf,
                target_ft_pred_df["y_true"].max() if len(target_ft_pred_df) > 0 else -np.inf,
                target_ft_pred_df["y_pred"].max() if len(target_ft_pred_df) > 0 else -np.inf,
            )
        )
        lo = float(
            min(
                lo,
                target_ft_pred_df["y_true"].min() if len(target_ft_pred_df) > 0 else np.inf,
                target_ft_pred_df["y_pred"].min() if len(target_ft_pred_df) > 0 else np.inf,
            )
        )
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="black")
        ax.set_xlabel(f"True {target_name}")
        ax.set_ylabel(f"Predicted {target_name}")
        ax.set_title("Domain Test: Predicted vs True")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "plot_test_pred_vs_true.png", dpi=200)
        plt.close(fig)

    if len(cell_metrics_df) > 0:
        mae_col = "mae_mean" if "mae_mean" in cell_metrics_df.columns else "mae"
        mape_col = "mape_percent_mean" if "mape_percent_mean" in cell_metrics_df.columns else "mape_percent"
        tmp = cell_metrics_df.sort_values(mae_col, ascending=True).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(max(8.0, 0.4 * len(tmp)), 4.5))
        ax.bar(tmp["cell"], tmp[mae_col])
        ax.set_xlabel("Test cell")
        ax.set_ylabel("MAE mean")
        ax.set_title("Domain Test: Cell-wise MAE mean")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        fig.savefig(out_dir / "plot_test_cell_mae.png", dpi=200)
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(max(8.0, 0.4 * len(tmp)), 8.0), sharex=True)
        axes[0].bar(tmp["cell"], tmp[mae_col], color="#4C78A8")
        axes[0].set_ylabel("MAE mean")
        axes[0].set_title("Domain Test: Cell-wise MAE mean / MAPE mean")
        axes[0].grid(True, axis="y", alpha=0.3)

        axes[1].bar(tmp["cell"], tmp[mape_col], color="#F58518")
        axes[1].set_ylabel("MAPE mean (%)")
        axes[1].set_xlabel("Test cell")
        axes[1].grid(True, axis="y", alpha=0.3)
        axes[1].tick_params(axis="x", rotation=90)

        fig.tight_layout()
        fig.savefig(out_dir / "plot_test_cell_mae_mape.png", dpi=200)
        plt.close(fig)

    if len(group_metrics_df) > 0:
        tmpg = group_metrics_df.sort_values("group_num").reset_index(drop=True)
        labels = [f"G{int(g)}" for g in tmpg["group_num"].tolist()]

        fig, axes = plt.subplots(2, 1, figsize=(max(8.0, 0.7 * len(tmpg)), 7.5), sharex=True)
        axes[0].bar(labels, tmpg["cell_mae_median"], color="#54A24B")
        axes[0].set_ylabel("Median Cell MAE")
        axes[0].set_title("Domain Test: Group-wise MAE / MAPE")
        axes[0].grid(True, axis="y", alpha=0.3)

        axes[1].bar(labels, tmpg["cell_mape_median"], color="#E45756")
        axes[1].set_ylabel("Median Cell MAPE (%)")
        axes[1].set_xlabel("Test group")
        axes[1].grid(True, axis="y", alpha=0.3)
        axes[1].tick_params(axis="x", rotation=0)

        fig.tight_layout()
        fig.savefig(out_dir / "plot_test_group_mae_mape.png", dpi=200)
        plt.close(fig)

    _save_summary_card(train_pred_df, str(train_summary_title), str(train_summary_filename))
    _save_summary_card(target_ft_pred_df, "Target Fine-tune Summary", "plot_target_finetune_summary_metrics.png")
    _save_summary_card(pred_df, "Domain Test Summary", "plot_test_summary_metrics.png")


def format_overall_metrics_line(row: pd.Series) -> str:
    mae_mean_val = float(row["mae_mean"]) if "mae_mean" in row else float(row["mae"])
    mae_median_val = float(row["mae_median"]) if "mae_median" in row else float(row["abs_err_median"])
    mape_mean_val = float(row["mape_percent_mean"]) if "mape_percent_mean" in row else float(row["mape_percent"])
    mape_median_val = float(row["mape_percent_median"]) if "mape_percent_median" in row else float("nan")
    smape_mean_val = float(row["smape_percent_mean"]) if "smape_percent_mean" in row else float("nan")
    wmape_val = float(row["wmape_percent"]) if "wmape_percent" in row else float("nan")
    return (
        f"MAE(mean/median)={mae_mean_val:.6f}/{mae_median_val:.6f}, "
        f"RMSE={float(row['rmse']):.6f}, "
        f"R2={float(row['r2']):.6f}, "
        f"MAPE(mean/median)={mape_mean_val:.3f}%/{mape_median_val:.3f}%, "
        f"SMAPE={smape_mean_val:.3f}%, "
        f"WMAPE={wmape_val:.3f}%"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Ridge domain train/test workflow for IVAS battery groups and conditions."
    )

    ap.add_argument(
        "--data_csv",
        type=str,
        default="E:/Datasets/IVAS/Processing_Data/rpt_samples_feature_soh.csv",
        help="Input CSV. Can be single-feature or multi-feature long-table CSV.",
    )
    ap.add_argument(
        "--group_cond_csv",
        type=str,
        default="E:/Datasets/IVAS/Groupcondi.csv",
        help="Path to Groupcondi.csv.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="E:/Datasets/IVAS/Ridge_Results_DomainShift",
        help="Output directory.",
    )

    ap.add_argument(
        "--split_mode",
        type=str,
        default="explicit",
        choices=["explicit", "within_group_random"],
        help="explicit: manual group/cell split; within_group_random: random cell split inside selected groups.",
    )

    ap.add_argument("--train_groups", type=str, default="", help='Explicit mode: comma-separated train groups, e.g. "1,2,3".')
    ap.add_argument("--test_groups", type=str, default="", help='Explicit mode: comma-separated test groups, e.g. "4,5,6".')
    ap.add_argument("--train_cells", type=str, default="", help='Explicit mode: comma-separated train cells, e.g. "G1C1,G1C2".')
    ap.add_argument("--test_cells", type=str, default="", help='Explicit mode: comma-separated test cells, e.g. "G4C1,G4C2".')
    ap.add_argument(
        "--target_ft_groups",
        type=str,
        default="",
        help='Optional target-domain fine-tune groups, chosen from eligible groups outside both train and explicit test pools, e.g. "25,26".',
    )
    ap.add_argument(
        "--target_ft_group_count",
        type=int,
        default=0,
        help="Optional random number of target-domain groups to use for few-shot fine-tuning. Must be smaller than the target test pool.",
    )
    ap.add_argument("--target_ft_seed", type=int, default=42, help="Random seed for target-domain fine-tune group sampling.")

    ap.add_argument("--groups", type=str, default="all", help='within_group_random mode: groups to split, e.g. "1,2,3" or "all".')
    ap.add_argument("--test_cell_frac", type=float, default=0.5, help="within_group_random mode: fraction of cells in each group used for test.")
    ap.add_argument("--cell_split_seed", type=int, default=42, help="Random seed for within-group cell split.")

    ap.add_argument("--train_release", type=str, default="all", help='Release filter for train side, e.g. "Release 1.0" or "all".')
    ap.add_argument("--test_release", type=str, default="all", help='Release filter for test side, e.g. "Release 2.0" or "all".')

    ap.add_argument(
        "--x_cols",
        type=str,
        default="feature_mean_ic",
        help='Feature columns. Use "auto" for f1..f10 or default single-feature detection.',
    )
    ap.add_argument("--y_col", type=str, default="soh", help="Target column.")
    ap.add_argument("--sort_col", type=str, default="time_week", help="Sort key for per-cell evaluation.")
    ap.add_argument("--time_col", type=str, default="time_week", help="Time column for time-weighted MAE.")

    ap.add_argument("--alpha", type=float, default=ALPHA_FIXED, help="Fixed Ridge alpha.")
    ap.add_argument("--tail_q", type=float, default=0.95, help="Tail quantile for absolute error.")
    ap.add_argument("--min_train_rows", type=int, default=8, help="Minimum rows required for each train cell.")
    ap.add_argument("--min_test_rows", type=int, default=8, help="Minimum rows required for each test cell.")
    ap.add_argument("--save_predictions", action="store_true", help="If set, save train/test prediction tables.")
    ap.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=True,
        help="Generate test plots. Default: true.",
    )
    ap.add_argument(
        "--no_plot",
        dest="plot",
        action="store_false",
        help="Disable test plot generation.",
    )

    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    if float(args.alpha) < 0:
        raise ValueError(f"alpha must be non-negative, got {args.alpha}")
    if not (0.0 < float(args.tail_q) < 1.0):
        raise ValueError("--tail_q must be in (0,1).")
    if int(args.target_ft_group_count) < 0:
        raise ValueError("--target_ft_group_count must be non-negative.")

    data_csv = Path(args.data_csv)
    group_cond_csv = Path(args.group_cond_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw, x_cols = read_samples_csv(
        data_csv=data_csv,
        x_cols_arg=args.x_cols,
        y_col=args.y_col,
        sort_col=args.sort_col,
        time_col=args.time_col,
    )
    cond_df = load_group_conditions(group_cond_csv)

    train_release_filter = parse_release_filter(args.train_release)
    test_release_filter = parse_release_filter(args.test_release)
    train_df_all = add_condition_columns(apply_release_filter(df_raw, train_release_filter), cond_df)
    test_df_all = add_condition_columns(apply_release_filter(df_raw, test_release_filter), cond_df)

    train_cell_df = filter_cells_by_min_rows(build_cell_table(train_df_all), args.min_train_rows)
    test_cell_df = filter_cells_by_min_rows(build_cell_table(test_df_all), args.min_test_rows)

    if args.split_mode == "explicit":
        train_cells, candidate_test_cells, split_info = resolve_explicit_split(train_cell_df, test_cell_df, args)
    else:
        shared_cells = sorted(
            set(train_cell_df["cell"].unique().tolist()) & set(test_cell_df["cell"].unique().tolist()),
            key=cell_sort_key,
        )
        shared_df = train_cell_df[train_cell_df["cell"].isin(shared_cells)].copy()
        if len(shared_df) == 0:
            raise ValueError(
                "within_group_random found no shared eligible cells between train/test release filters. "
                "Use the same release filter on both sides, or switch to explicit mode."
            )
        train_cells, candidate_test_cells, split_info = resolve_within_group_random_split(shared_df, args)

    target_ft_cells, target_test_cells, target_ft_info = resolve_target_finetune_split(
        test_cell_df=test_cell_df,
        train_cells=train_cells,
        candidate_test_cells=candidate_test_cells,
        args=args,
    )
    split_info["target_ft"] = target_ft_info

    source_train_df = materialize_cells_frame(
        df_all=train_df_all,
        cells=train_cells,
        min_rows=int(args.min_train_rows),
        split_name_for_error="Source train",
    )
    target_ft_df = materialize_cells_frame(
        df_all=test_df_all,
        cells=target_ft_cells,
        min_rows=int(args.min_train_rows),
        split_name_for_error="Target fine-tune",
        allow_empty=True,
    )
    target_test_df = materialize_cells_frame(
        df_all=test_df_all,
        cells=target_test_cells,
        min_rows=int(args.min_test_rows),
        split_name_for_error="Target holdout test",
    )

    source_model = fit_ridge_model(train_df=source_train_df, x_cols=x_cols, y_col=args.y_col, alpha=float(args.alpha))
    source_train_pred_df = build_prediction_df(
        df=source_train_df,
        y_col=args.y_col,
        y_pred=source_model["y_pred_train"],
        split_name="source_train",
    )
    target_test_y_pred_source = predict_df(
        df=target_test_df,
        x_cols=x_cols,
        mu=source_model["mu"],
        sd=source_model["sd"],
        beta_s=source_model["beta_s"],
        intercept=float(source_model["intercept"]),
    )
    target_test_pred_df_source = build_prediction_df(
        df=target_test_df,
        y_col=args.y_col,
        y_pred=target_test_y_pred_source,
        split_name="target_test_source_only",
    )

    has_target_finetune = len(target_ft_df) > 0
    active_model = source_model
    target_ft_pred_df = pd.DataFrame()
    target_test_pred_df_final = target_test_pred_df_source.copy()
    if has_target_finetune:
        active_model = finetune_ridge_model(
            finetune_df=target_ft_df,
            x_cols=x_cols,
            y_col=args.y_col,
            alpha=float(args.alpha),
            base_model=source_model,
        )
        target_ft_pred_df = build_prediction_df(
            df=target_ft_df,
            y_col=args.y_col,
            y_pred=active_model["y_pred_train"],
            split_name="target_finetune",
        )
        target_test_y_pred_ft = predict_df(
            df=target_test_df,
            x_cols=x_cols,
            mu=active_model["mu"],
            sd=active_model["sd"],
            beta_s=active_model["beta_s"],
            intercept=float(active_model["intercept"]),
        )
        target_test_pred_df_final = build_prediction_df(
            df=target_test_df,
            y_col=args.y_col,
            y_pred=target_test_y_pred_ft,
            split_name="target_test_finetuned",
        )

    source_train_overall_df = summarize_overall(source_train_pred_df, split_name="source_train", tail_q=float(args.tail_q))
    target_test_source_overall_df = summarize_overall(
        target_test_pred_df_source,
        split_name="target_test_source_only",
        tail_q=float(args.tail_q),
    )
    target_test_overall_df = summarize_overall(
        target_test_pred_df_final,
        split_name="target_test_finetuned" if has_target_finetune else "test",
        tail_q=float(args.tail_q),
    )
    source_train_cell_metrics_df = summarize_by_cell(
        source_train_pred_df,
        sort_col=args.sort_col,
        time_col=args.time_col,
        tail_q=float(args.tail_q),
    )
    target_test_source_cell_metrics_df = summarize_by_cell(
        target_test_pred_df_source,
        sort_col=args.sort_col,
        time_col=args.time_col,
        tail_q=float(args.tail_q),
    )
    target_test_cell_metrics_df = summarize_by_cell(
        target_test_pred_df_final,
        sort_col=args.sort_col,
        time_col=args.time_col,
        tail_q=float(args.tail_q),
    )
    source_train_group_metrics_df = summarize_by_group(
        source_train_pred_df,
        source_train_cell_metrics_df,
        tail_q=float(args.tail_q),
    )
    target_test_source_group_metrics_df = summarize_by_group(
        target_test_pred_df_source,
        target_test_source_cell_metrics_df,
        tail_q=float(args.tail_q),
    )
    target_test_group_metrics_df = summarize_by_group(
        target_test_pred_df_final,
        target_test_cell_metrics_df,
        tail_q=float(args.tail_q),
    )
    target_ft_overall_df = (
        summarize_overall(target_ft_pred_df, split_name="target_finetune", tail_q=float(args.tail_q))
        if has_target_finetune
        else pd.DataFrame()
    )
    target_ft_cell_metrics_df = (
        summarize_by_cell(target_ft_pred_df, sort_col=args.sort_col, time_col=args.time_col, tail_q=float(args.tail_q))
        if has_target_finetune
        else pd.DataFrame()
    )
    target_ft_group_metrics_df = (
        summarize_by_group(target_ft_pred_df, target_ft_cell_metrics_df, tail_q=float(args.tail_q))
        if has_target_finetune
        else pd.DataFrame()
    )

    split_train_cells_df = build_cell_table(source_train_df).sort_values(["group_num", "cell_idx", "cell"]).reset_index(drop=True)
    split_test_cells_df = build_cell_table(target_test_df).sort_values(["group_num", "cell_idx", "cell"]).reset_index(drop=True)
    split_target_ft_cells_df = (
        build_cell_table(target_ft_df).sort_values(["group_num", "cell_idx", "cell"]).reset_index(drop=True)
        if has_target_finetune
        else pd.DataFrame(columns=split_train_cells_df.columns)
    )

    source_model_payload = {
        "alpha": float(args.alpha),
        "x_cols": list(x_cols),
        "y_col": str(args.y_col),
        "sort_col": str(args.sort_col),
        "time_col": str(args.time_col),
        "mu": np.asarray(source_model["mu"], dtype=float).tolist(),
        "sd": np.asarray(source_model["sd"], dtype=float).tolist(),
        "beta_s": np.asarray(source_model["beta_s"], dtype=float).tolist(),
        "intercept": float(source_model["intercept"]),
        "model_role": "source_only",
    }
    active_model_payload = {
        "alpha": float(args.alpha),
        "x_cols": list(x_cols),
        "y_col": str(args.y_col),
        "sort_col": str(args.sort_col),
        "time_col": str(args.time_col),
        "mu": np.asarray(active_model["mu"], dtype=float).tolist(),
        "sd": np.asarray(active_model["sd"], dtype=float).tolist(),
        "beta_s": np.asarray(active_model["beta_s"], dtype=float).tolist(),
        "intercept": float(active_model["intercept"]),
        "model_role": "finetuned" if has_target_finetune else "source_only",
    }

    config_payload = {
        "data_csv": str(data_csv),
        "group_cond_csv": str(group_cond_csv),
        "split_mode": str(args.split_mode),
        "train_release": None if train_release_filter is None else sorted(train_release_filter),
        "test_release": None if test_release_filter is None else sorted(test_release_filter),
        "min_train_rows": int(args.min_train_rows),
        "min_test_rows": int(args.min_test_rows),
        "tail_q": float(args.tail_q),
        "split_info": split_info,
    }

    split_train_cells_df.to_csv(out_dir / "train_cells.csv", index=False)
    split_test_cells_df.to_csv(out_dir / "test_cells.csv", index=False)
    source_train_overall_df.to_csv(out_dir / "train_overall_metrics.csv", index=False)
    target_test_overall_df.to_csv(out_dir / "test_overall_metrics.csv", index=False)
    source_train_cell_metrics_df.to_csv(out_dir / "train_cell_metrics.csv", index=False)
    target_test_cell_metrics_df.to_csv(out_dir / "test_cell_metrics.csv", index=False)
    source_train_group_metrics_df.to_csv(out_dir / "train_group_metrics.csv", index=False)
    target_test_group_metrics_df.to_csv(out_dir / "test_group_metrics.csv", index=False)
    if has_target_finetune:
        target_test_source_overall_df.to_csv(out_dir / "test_overall_metrics_source_only.csv", index=False)
        target_test_source_cell_metrics_df.to_csv(out_dir / "test_cell_metrics_source_only.csv", index=False)
        target_test_source_group_metrics_df.to_csv(out_dir / "test_group_metrics_source_only.csv", index=False)
        split_target_ft_cells_df.to_csv(out_dir / "target_finetune_cells.csv", index=False)
        target_ft_overall_df.to_csv(out_dir / "target_finetune_overall_metrics.csv", index=False)
        target_ft_cell_metrics_df.to_csv(out_dir / "target_finetune_cell_metrics.csv", index=False)
        target_ft_group_metrics_df.to_csv(out_dir / "target_finetune_group_metrics.csv", index=False)

    if args.save_predictions:
        source_train_pred_df.to_csv(out_dir / "predictions_train.csv", index=False)
        target_test_pred_df_final.to_csv(out_dir / "predictions_test.csv", index=False)
        if has_target_finetune:
            target_test_pred_df_source.to_csv(out_dir / "predictions_test_source_only.csv", index=False)
            target_ft_pred_df.to_csv(out_dir / "predictions_target_finetune.csv", index=False)

    save_json(out_dir / "model.json", active_model_payload)
    if has_target_finetune:
        save_json(out_dir / "model_source.json", source_model_payload)
        save_json(out_dir / "model_finetuned.json", active_model_payload)
    save_json(out_dir / "config.json", config_payload)

    if args.plot:
        maybe_make_plots(
            train_pred_df=source_train_pred_df,
            target_ft_pred_df=target_ft_pred_df,
            pred_df=target_test_pred_df_final,
            cell_metrics_df=target_test_cell_metrics_df,
            group_metrics_df=target_test_group_metrics_df,
            overall_metrics_df=target_test_overall_df,
            out_dir=out_dir,
        )

    print("[INFO] Domain train/test finished.")
    print(f"[INFO] x_cols            : {x_cols}")
    if has_target_finetune:
        print(f"[INFO] source groups     : {sorted(source_train_df['group_num'].unique().tolist())}")
        print(f"[INFO] target ft groups  : {sorted(target_ft_df['group_num'].unique().tolist())}")
        print(f"[INFO] target test groups: {sorted(target_test_df['group_num'].unique().tolist())}")
        print(f"[INFO] source cells      : {sorted(source_train_df['cell'].unique().tolist(), key=cell_sort_key)}")
        print(f"[INFO] target ft cells   : {sorted(target_ft_df['cell'].unique().tolist(), key=cell_sort_key)}")
        print(f"[INFO] target test cells : {sorted(target_test_df['cell'].unique().tolist(), key=cell_sort_key)}")
    else:
        print(f"[INFO] train groups      : {sorted(source_train_df['group_num'].unique().tolist())}")
        print(f"[INFO] test groups       : {sorted(target_test_df['group_num'].unique().tolist())}")
        print(f"[INFO] train cells       : {sorted(source_train_df['cell'].unique().tolist(), key=cell_sort_key)}")
        print(f"[INFO] test cells        : {sorted(target_test_df['cell'].unique().tolist(), key=cell_sort_key)}")
    print(f"[INFO] saved out_dir     : {out_dir}")
    print(f"[INFO] train overall     : {format_overall_metrics_line(source_train_overall_df.iloc[0])}")
    if has_target_finetune:
        print(f"[INFO] target ft overall : {format_overall_metrics_line(target_ft_overall_df.iloc[0])}")
        print(f"[INFO] target source-only: {format_overall_metrics_line(target_test_source_overall_df.iloc[0])}")
        print(f"[INFO] target final      : {format_overall_metrics_line(target_test_overall_df.iloc[0])}")
    else:
        print(f"[INFO] test overall      : {format_overall_metrics_line(target_test_overall_df.iloc[0])}")


if __name__ == "__main__":
    main()
