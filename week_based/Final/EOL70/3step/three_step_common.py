#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import sys
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKSPACE_ROOT.parents[2]
DEFAULT_PYTHON = os.environ.get("PYTHON", sys.executable)
DEFAULT_RUNNER = Path(__file__).resolve().parent / "three_step_transfer_runner.py"
DEFAULT_DATA_CSV = WORKSPACE_ROOT / "features" / "feature_table_all_cells_multiweek_EOL70.csv"
DEFAULT_SPLIT_CSV = WORKSPACE_ROOT / "domain_split" / "cell_split_targetspread_w5_EOL70.csv"
DEFAULT_GROUP_CSV = REPO_ROOT / "metadata" / "Groupcondi.csv"
DEFAULT_CANDIDATE_CSV = WORKSPACE_ROOT / "features" / "informed_feature_candidates_w5_EOL70.csv"


WEEK_BASE_RUNNER = WORKSPACE_ROOT / "codes" / "run_lifetime_transfer_mlp.py"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


week_runner = _load_module("week_runner_3step", WEEK_BASE_RUNNER)
mlp_base = week_runner.mlp_base


def ensure_child_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


def parse_csv_list(raw: str, cast=str) -> List:
    return [cast(item.strip()) for item in str(raw).split(",") if item.strip()]


def architecture_candidates(width_candidates: Sequence[int], max_depth: int) -> List[str]:
    archs: List[str] = []
    widths = sorted({int(v) for v in width_candidates})
    for depth in range(1, int(max_depth) + 1):
        for combo in combinations_with_replacement(widths, depth):
            archs.append(",".join(str(v) for v in reversed(combo)))
    return archs


def load_feature_candidates(candidate_csv: Path, max_candidates: int | None = None) -> List[str]:
    df = pd.read_csv(candidate_csv)
    if "features" not in df.columns:
        raise ValueError(f"{candidate_csv} is missing required column 'features'.")
    out: List[str] = []
    seen = set()
    for raw in df["features"].astype(str):
        features = ",".join(item.strip() for item in raw.split(",") if item.strip())
        if not features or features in seen:
            continue
        seen.add(features)
        out.append(features)
        if max_candidates is not None and len(out) >= int(max_candidates):
            break
    if not out:
        raise ValueError(f"No valid feature candidates found in {candidate_csv}.")
    return out


def select_support_cells(
    full_target_ft_df: pd.DataFrame,
    *,
    y_col: str,
    support_ratio: float,
    min_support_cells: int,
    mode: str,
    seed: int,
) -> List[str]:
    cell_df = (
        full_target_ft_df[["cell", y_col]]
        .drop_duplicates(subset=["cell"])
        .sort_values(["cell"])
        .reset_index(drop=True)
    )
    cells = cell_df["cell"].astype(str).tolist()
    if not cells:
        return []

    n_total = len(cells)
    n_pick = int(round(float(support_ratio) * n_total))
    n_pick = max(int(min_support_cells), n_pick)
    n_pick = min(n_total, n_pick)
    if n_pick >= n_total:
        return cells

    work = cell_df.copy()
    mode = str(mode).lower()
    if mode == "random":
        rng = np.random.default_rng(int(seed))
        chosen = sorted(rng.choice(np.array(cells, dtype=object), size=n_pick, replace=False).tolist(), key=week_runner.ridge_utils.cell_sort_key)
        return chosen

    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    if mode == "high_tail":
        work = work.sort_values([y_col, "cell"], ascending=[False, True]).reset_index(drop=True)
        chosen = work.head(n_pick)["cell"].astype(str).tolist()
    else:
        work = work.sort_values([y_col, "cell"]).reset_index(drop=True)
        idx = np.linspace(0, len(work) - 1, n_pick, dtype=int)
        chosen = work.iloc[idx]["cell"].astype(str).tolist()
    return sorted(dict.fromkeys(chosen), key=week_runner.ridge_utils.cell_sort_key)


def write_split_csv_with_support_subset(
    *,
    base_split_csv: Path,
    out_csv: Path,
    support_cells: Sequence[str],
) -> Path:
    split_df = pd.read_csv(base_split_csv)
    support_set = {str(cell).strip() for cell in support_cells}
    keep = []
    for _, row in split_df.iterrows():
        split_name = str(row["split"]).strip()
        cell = str(row["cell"]).strip()
        if split_name != "fine_tune" or cell in support_set:
            keep.append(row)
    pd.DataFrame(keep).to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv


def read_single_row_csv(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"Expected exactly one row in {path}, got {len(df)}")
    row = df.iloc[0].to_dict()
    out: Dict[str, float] = {}
    for key, value in row.items():
        try:
            out[str(key)] = float(value)
        except Exception:
            continue
    return out


def sort_frame_for_stage(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    if stage == "stage1":
        keys = ["source_val_mae", "source_val_rmse", "source_val_mape_percent_mean", "train_val_mae_gap"]
        ascending = [True, True, True, True]
    else:
        keys = ["target_ft_val_mae", "target_ft_val_rmse", "target_ft_val_mape_percent_mean"]
        ascending = [True, True, True]
    return df.sort_values(keys, ascending=ascending).reset_index(drop=True)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
