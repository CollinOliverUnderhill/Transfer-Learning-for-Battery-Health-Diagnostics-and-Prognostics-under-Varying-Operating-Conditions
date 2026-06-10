#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def resolve_mlp_base_path() -> Path:
    current = Path(__file__).resolve()
    candidates = [
        current.parents[3] / "SOHest" / "MLP_codes" / "domain_train_test_by_group_mlp.py",
        current.parents[1] / "SOHest" / "MLP_codes" / "domain_train_test_by_group_mlp.py",
        current.parents[2] / "Codes" / "chunqiu_codes" / "SOHest" / "MLP_codes" / "domain_train_test_by_group_mlp.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Required MLP base script not found. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


MLP_BASE_PATH = resolve_mlp_base_path()

_SPEC = importlib.util.spec_from_file_location("lifetime_mlp_base", MLP_BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Failed to load base MLP module from: {MLP_BASE_PATH}")
mlp_base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mlp_base)

ridge_utils = mlp_base.ridge_utils


FEATURE_ALIAS_MAP: Dict[str, str] = {
    "f1": "step1_log_abs_mean_delta_dQdV_w3_w0_3p6_3p9",
    "f2": "step2_log_abs_delta_CV_time_w3_w0",
    "f3": "step3_DoD",
    "f4": "step4_delta_Q1_DVA_w3_w0",
    "f5": "step5_sqrt_Cchg_sqrt_DoD",
    "f6": "step6_Cchg",
    "f7": "step7_log_abs_var_delta_dQdV_w3_w0_3p0_3p6",
    "f8": "step8_delta_Q3_DVA_w3_w0",
    "f9": "step9_log_abs_mean_delta_dQdV_w3_w0_3p0_3p6",
    "f10": "step10_log_abs_CV_time_w0",
}


FEATURE_ALIAS_RE = re.compile(r"^(f(?:10|[1-9]))(?:_w(\d+))?$")


def parse_feature_aliases(arg: str) -> Tuple[List[str], List[str], int]:
    aliases = [x.strip().lower() for x in str(arg).split(",") if x.strip()]
    if not aliases:
        raise ValueError("--features must contain at least one alias, e.g. f1,f3,f5")
    parsed: List[Tuple[str, Optional[int]]] = []
    bad_aliases: List[str] = []
    for alias in aliases:
        match = FEATURE_ALIAS_RE.match(alias)
        if match is None:
            bad_aliases.append(alias)
            continue
        base_alias = match.group(1)
        week_num = int(match.group(2)) if match.group(2) is not None else None
        parsed.append((base_alias, week_num))
    if bad_aliases:
        raise ValueError(
            "Unsupported feature aliases: "
            f"{bad_aliases}. Supported examples: f1,f3,f5 or f1_w5,f3_w5,f5_w5"
        )

    selected_weeks = sorted({week for _, week in parsed if week is not None})
    if len(selected_weeks) > 1:
        raise ValueError(f"All week-specific feature aliases must use the same week. Got: {selected_weeks}")
    selected_week = selected_weeks[0] if selected_weeks else 3

    if selected_week == 3 and all(week is None for _, week in parsed):
        cols = [FEATURE_ALIAS_MAP[base_alias] for base_alias, _ in parsed]
        return aliases, cols, selected_week

    cols = [f"{base_alias}_w{selected_week}" for base_alias, _ in parsed]
    normalized_aliases = [f"{base_alias}_w{selected_week}" for base_alias, _ in parsed]
    return normalized_aliases, cols, selected_week


def load_lifetime_frame(data_csv: Path, x_cols: Sequence[str], y_col: str, week_num: int = 3) -> pd.DataFrame:
    if not data_csv.exists():
        raise FileNotFoundError(f"Lifetime feature CSV not found: {data_csv}")

    df = pd.read_csv(data_csv)
    rpt_idx_col = f"week{week_num}_rpt_idx"
    time_week_col = f"week{week_num}_time_week"
    required = ["cell", "release", "group_num", "cell_idx", rpt_idx_col, time_week_col, y_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Lifetime CSV missing required columns: {missing}")
    missing_x = [c for c in x_cols if c not in df.columns]
    if missing_x:
        raise ValueError(f"Lifetime CSV missing requested feature columns: {missing_x}")

    out = df.copy()
    out["cell"] = out["cell"].astype(str).str.strip()
    out["release"] = out["release"].astype(str).str.strip()
    out["group_num"] = pd.to_numeric(out["group_num"], errors="coerce")
    out["cell_idx"] = pd.to_numeric(out["cell_idx"], errors="coerce")
    out["rpt_idx"] = pd.to_numeric(out[rpt_idx_col], errors="coerce")
    out["time_week"] = pd.to_numeric(out[time_week_col], errors="coerce")
    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    for col in x_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    need = ["cell", "release", "group_num", "cell_idx", "rpt_idx", "time_week", y_col, *x_cols]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=need).copy()
    out["group_num"] = out["group_num"].astype(int)
    out["cell_idx"] = out["cell_idx"].astype(int)
    out["rpt_idx"] = out["rpt_idx"].astype(int)
    return out


def load_split_cells(split_csv: Path) -> Dict[str, List[str]]:
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")
    df = pd.read_csv(split_csv)
    required = ["split", "cell"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Split CSV missing required columns: {missing}")

    out: Dict[str, List[str]] = {}
    for split_name, sub in df.groupby("split", dropna=False):
        out[str(split_name)] = sorted(sub["cell"].astype(str).str.strip().unique().tolist(), key=ridge_utils.cell_sort_key)
    return out


def select_cells(df: pd.DataFrame, cells: Sequence[str], split_name: str) -> pd.DataFrame:
    cells = sorted(set(str(c).strip() for c in cells if str(c).strip()), key=ridge_utils.cell_sort_key)
    out = df[df["cell"].isin(cells)].copy()
    if len(out) == 0:
        raise ValueError(f"{split_name} dataframe is empty after applying cells.")
    return out.sort_values(["group_num", "cell_idx", "cell"]).reset_index(drop=True)


def mark_active_split(df: pd.DataFrame, split: str, split_label: str) -> pd.DataFrame:
    out = df.copy()
    if "split" in out.columns and "original_split" not in out.columns:
        out["original_split"] = out["split"]
    if "split_label" in out.columns and "original_split_label" not in out.columns:
        out["original_split_label"] = out["split_label"]
    out["split"] = str(split)
    out["split_label"] = str(split_label)
    return out


def subset_prediction_df_by_cells(pred_df: pd.DataFrame, cells: Sequence[str], split_name: str) -> pd.DataFrame:
    cell_set = {str(c).strip() for c in cells if str(c).strip()}
    out = pred_df[pred_df["cell"].astype(str).isin(cell_set)].copy()
    if len(out) == 0:
        raise ValueError(f"{split_name} prediction dataframe is empty after applying cells.")
    return out.sort_values(["group_num", "cell_idx", "cell"]).reset_index(drop=True)


def summarize_prediction_split(
    pred_df: pd.DataFrame,
    split_name: str,
    tail_q: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_df = ridge_utils.summarize_overall(pred_df, split_name, tail_q)
    cell_metrics_df = ridge_utils.summarize_by_cell(pred_df, "time_week", "time_week", tail_q)
    group_metrics_df = ridge_utils.summarize_by_group(pred_df, cell_metrics_df, tail_q)
    return overall_df, cell_metrics_df, group_metrics_df


def save_prediction_summary_card(pred_df: pd.DataFrame, title: str, out_path: Path, tail_q: float) -> None:
    if len(pred_df) == 0:
        return
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[WARN] matplotlib is not installed. Skip summary card generation.")
        return

    stats = ridge_utils.describe_errors(
        pred_df["y_true"].to_numpy(dtype=float),
        pred_df["y_pred"].to_numpy(dtype=float),
        tail_q=float(tail_q),
    )
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.axis("off")
    summary_text = (
        f"{title}\n"
        f"Rows: {int(len(pred_df))}    Cells: {int(pred_df['cell'].nunique())}    Groups: {int(pred_df['group_num'].nunique())}\n"
        f"MAE mean   : {float(stats['mae_mean']):.6f}\n"
        f"MAE median : {float(stats['mae_median']):.6f}\n"
        f"RMSE       : {float(stats['rmse']):.6f}\n"
        f"R2         : {float(stats['r2']):.6f}\n"
        f"MAPE mean  : {float(stats['mape_percent_mean']):.3f}%\n"
        f"MAPE median: {float(stats['mape_percent_median']):.3f}%\n"
        f"SMAPE mean : {float(stats['smape_percent_mean']):.3f}%\n"
        f"WMAPE      : {float(stats['wmape_percent']):.3f}%"
    )
    ax.text(0.02, 0.95, summary_text, va="top", ha="left", fontsize=12, family="monospace")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_benchmark_train_df(source_train_df: pd.DataFrame, target_ft_df: pd.DataFrame) -> pd.DataFrame:
    if len(target_ft_df) == 0:
        return source_train_df.copy().reset_index(drop=True)
    combined = pd.concat([source_train_df, target_ft_df], ignore_index=True)
    return combined.sort_values(["group_num", "cell_idx", "cell"]).reset_index(drop=True)


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
        chosen = rng.choice(np.array(cells, dtype=object), size=n_pick, replace=False).tolist()
        return sorted(chosen, key=ridge_utils.cell_sort_key)

    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    if mode == "high_tail":
        work = work.sort_values([y_col, "cell"], ascending=[False, True]).reset_index(drop=True)
        chosen = work.head(n_pick)["cell"].astype(str).tolist()
    else:
        work = work.sort_values([y_col, "cell"]).reset_index(drop=True)
        idx = np.linspace(0, len(work) - 1, n_pick, dtype=int)
        chosen = work.iloc[idx]["cell"].astype(str).tolist()
    return sorted(dict.fromkeys(chosen), key=ridge_utils.cell_sort_key)


def average_state_dicts(states: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError("Cannot average an empty state list.")
    averaged: Dict[str, torch.Tensor] = {}
    for key in states[0]:
        first = states[0][key]
        if torch.is_floating_point(first):
            stacked = torch.stack([state[key].to(dtype=torch.float32) for state in states], dim=0)
            averaged[key] = stacked.mean(dim=0).to(dtype=first.dtype)
        else:
            averaged[key] = states[-1][key].clone()
    return averaged


def rolling_median(values: Sequence[float], window: int) -> float:
    if not values:
        return float("inf")
    n = max(1, int(window))
    return float(np.median(np.asarray(values[-n:], dtype=float)))


def train_stage_with_replay(
    *,
    train_df: pd.DataFrame,
    replay_df: Optional[pd.DataFrame],
    replay_loss_weight: float,
    x_cols: Sequence[str],
    y_col: str,
    mu: np.ndarray,
    sd: np.ndarray,
    hidden_dims: Sequence[int],
    dropout: float,
    activation: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    val_cell_frac: float,
    early_stop_patience: int,
    min_epochs_before_early_stop: int,
    seed: int,
    device: torch.device,
    init_state_dict: Dict[str, torch.Tensor],
    freeze_hidden_layers: int,
    min_val_cells: int,
    log_every: int,
    stage_name: str,
    ft_batch_mode: str = "mini",
    ft_selection_mode: str = "raw_best",
    ft_smooth_window: int = 25,
    ft_swa_window: int = 50,
    ft_l2sp_weight: float = 0.0,
) -> Dict[str, object]:
    mlp_base.set_seed(int(seed))
    train_part_df, val_part_df = mlp_base.split_train_val_by_cell(train_df, val_cell_frac=val_cell_frac, seed=seed)
    if len(val_part_df) == 0:
        raise ValueError(f"stage={stage_name} requires a non-empty validation split.")
    val_cell_count = int(val_part_df["cell"].astype(str).nunique())
    if val_cell_count < int(min_val_cells):
        raise ValueError(
            f"stage={stage_name} target_finetune val cell count {val_cell_count} is below required minimum {int(min_val_cells)}."
        )

    model = mlp_base.MLPRegressor(len(x_cols), hidden_dims, dropout=dropout, activation=activation).to(device)
    model.load_state_dict(init_state_dict)
    frozen_names = mlp_base.freeze_first_hidden_layers(model, int(freeze_hidden_layers))

    X_train_t, y_train_t = mlp_base.df_to_tensors(train_part_df, x_cols, y_col, mu, sd, device)
    full_batch = str(ft_batch_mode).strip().lower() == "full"
    train_batch_size = len(train_part_df) if full_batch else int(batch_size)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=max(1, min(train_batch_size, len(train_part_df))),
        shuffle=not full_batch,
    )

    use_replay = replay_df is not None and len(replay_df) > 0 and float(replay_loss_weight) > 0.0
    replay_loader = None
    if use_replay:
        X_replay_t, y_replay_t = mlp_base.df_to_tensors(replay_df, x_cols, y_col, mu, sd, device)
        replay_batch_size = len(replay_df) if full_batch else int(batch_size)
        replay_loader = DataLoader(
            TensorDataset(X_replay_t, y_replay_t),
            batch_size=max(1, min(replay_batch_size, len(replay_df))),
            shuffle=not full_batch,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain after freezing layers.")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(lr), weight_decay=float(weight_decay), foreach=False)
    loss_fn = nn.MSELoss()
    source_params = {
        name: init_state_dict[name].detach().to(device=device, dtype=param.dtype)
        for name, param in model.named_parameters()
        if param.requires_grad and name in init_state_dict
    }
    l2sp_numel = sum(param.numel() for param in source_params.values())

    best_metric = float("inf")
    best_epoch = 0
    best_state = mlp_base.clone_state_dict(model)
    history: List[Dict[str, float]] = []
    patience_left = int(early_stop_patience)
    val_losses: List[float] = []
    swa_states: List[Dict[str, torch.Tensor]] = []
    selection_mode = str(ft_selection_mode).strip().lower()
    if selection_mode not in {"raw_best", "smooth_best", "last_window_swa", "final"}:
        raise ValueError(f"Unsupported ft_selection_mode: {ft_selection_mode}")
    swa_start_epoch = max(1, int(epochs) - max(1, int(ft_swa_window)) + 1)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        running_loss = 0.0
        running_target_loss = 0.0
        running_replay_loss = 0.0
        running_l2sp_loss = 0.0
        n_seen = 0
        replay_iter = iter(replay_loader) if use_replay and replay_loader is not None else None

        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            target_loss = loss_fn(pred, yb)
            total_loss = target_loss
            replay_loss_value = 0.0
            if use_replay and replay_iter is not None:
                try:
                    rb, ry = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    rb, ry = next(replay_iter)
                replay_loss = loss_fn(model(rb), ry)
                total_loss = total_loss + float(replay_loss_weight) * replay_loss
                replay_loss_value = float(replay_loss.item())
            l2sp_loss_value = 0.0
            if float(ft_l2sp_weight) > 0.0 and source_params and l2sp_numel > 0:
                l2sp_loss = sum(
                    torch.sum((param - source_params[name]) ** 2)
                    for name, param in model.named_parameters()
                    if param.requires_grad and name in source_params
                ) / float(l2sp_numel)
                total_loss = total_loss + float(ft_l2sp_weight) * l2sp_loss
                l2sp_loss_value = float(l2sp_loss.item())
            total_loss.backward()
            optimizer.step()

            running_loss += float(total_loss.item()) * len(xb)
            running_target_loss += float(target_loss.item()) * len(xb)
            running_replay_loss += replay_loss_value * len(xb)
            running_l2sp_loss += l2sp_loss_value * len(xb)
            n_seen += len(xb)

        train_loss = running_loss / n_seen if n_seen > 0 else float("nan")
        val_loss = mlp_base.evaluate_loss(model, val_part_df, x_cols, y_col, mu, sd, device, batch_size)
        val_losses.append(float(val_loss))
        selection_metric = rolling_median(val_losses, int(ft_smooth_window)) if selection_mode == "smooth_best" else float(val_loss)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_target_loss": float(running_target_loss / n_seen if n_seen > 0 else float("nan")),
                "train_replay_loss": float(running_replay_loss / n_seen if n_seen > 0 else float("nan")),
                "train_l2sp_loss": float(running_l2sp_loss / n_seen if n_seen > 0 else float("nan")),
                "val_loss": float(val_loss),
                "selection_metric": float(selection_metric),
            }
        )
        if int(log_every) > 0 and (epoch == 1 or epoch % int(log_every) == 0):
            print(
                f"[LOSS] stage={stage_name} epoch={epoch} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} selection_metric={selection_metric:.6f}"
            )
        if selection_mode == "last_window_swa" and epoch >= swa_start_epoch:
            swa_states.append(mlp_base.clone_state_dict(model))
        if selection_mode == "final":
            best_metric = float(selection_metric)
            best_epoch = int(epoch)
            best_state = mlp_base.clone_state_dict(model)
        elif selection_mode != "last_window_swa" and selection_metric < best_metric:
            best_metric = float(selection_metric)
            best_epoch = int(epoch)
            best_state = mlp_base.clone_state_dict(model)
            patience_left = int(early_stop_patience)
        else:
            if epoch >= int(min_epochs_before_early_stop):
                patience_left -= 1
                if patience_left <= 0:
                    break

    if selection_mode == "last_window_swa":
        if not swa_states:
            swa_states.append(mlp_base.clone_state_dict(model))
        best_state = average_state_dicts(swa_states)
        best_epoch = int(history[-1]["epoch"]) if history else 0
        best_metric = rolling_median(val_losses, int(ft_smooth_window))

    model.load_state_dict(best_state)
    y_pred_train = mlp_base.predict_with_model(model, train_df, x_cols, mu, sd, device, batch_size)
    return {
        "model": model,
        "state_dict": best_state,
        "y_pred_train": y_pred_train,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "selection_mode": selection_mode,
        "ft_batch_mode": str(ft_batch_mode),
        "ft_smooth_window": int(ft_smooth_window),
        "ft_swa_window": int(ft_swa_window),
        "ft_l2sp_weight": float(ft_l2sp_weight),
        "used_validation": True,
        "freeze_hidden_layers": int(freeze_hidden_layers),
        "frozen_parameter_names": frozen_names,
        "train_cells": sorted(train_part_df["cell"].astype(str).unique().tolist(), key=ridge_utils.cell_sort_key),
        "val_cells": sorted(val_part_df["cell"].astype(str).unique().tolist(), key=ridge_utils.cell_sort_key),
        "replay_loss_weight": float(replay_loss_weight),
    }


def read_single_row_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected summary CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if len(df) != 1:
        raise ValueError(f"Expected exactly one row in {csv_path}, got {len(df)}")
    return df


def load_saved_overall_results(out_dir: Path, *, include_target_finetune: bool) -> Dict[str, pd.DataFrame]:
    results = {
        "source_inner_train_overall": read_single_row_csv(out_dir / "source_inner_train_overall_metrics.csv"),
        "source_val_overall": read_single_row_csv(out_dir / "source_val_overall_metrics.csv"),
        "test_overall": read_single_row_csv(out_dir / "test_overall_metrics.csv"),
    }
    if include_target_finetune:
        results.update(
            {
                "target_ft_inner_train_overall": read_single_row_csv(out_dir / "target_finetune_inner_train_overall_metrics.csv"),
                "target_ft_val_overall": read_single_row_csv(out_dir / "target_finetune_val_overall_metrics.csv"),
                "target_ft_overall": read_single_row_csv(out_dir / "target_finetune_overall_metrics.csv"),
                "test_source_only_overall": read_single_row_csv(out_dir / "test_overall_metrics_source_only.csv"),
            }
        )
    return results


def try_load_saved_overall_results(out_dir: Path, *, include_target_finetune: bool) -> Optional[Dict[str, pd.DataFrame]]:
    try:
        return load_saved_overall_results(out_dir, include_target_finetune=include_target_finetune)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[WARN] {exc}")
        return None


def append_summary_block(lines: List[str], title: str, overall_df: Optional[pd.DataFrame]) -> None:
    if overall_df is None or len(overall_df) == 0:
        return
    lines.extend(["", title, ridge_utils.format_overall_metrics_line(overall_df.iloc[0])])


def print_summary_metric(prefix: str, overall_df: Optional[pd.DataFrame]) -> None:
    if overall_df is None or len(overall_df) == 0:
        return
    print(f"[INFO] {prefix:<24}: {ridge_utils.format_overall_metrics_line(overall_df.iloc[0])}")


def save_stage_outputs(
    *,
    out_dir: Path,
    source_stage: Dict[str, object],
    active_stage: Dict[str, object],
    x_cols: Sequence[str],
    feature_aliases: Sequence[str],
    y_col: str,
    device,
    hidden_dims: Sequence[int],
    source_train_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
    target_ft_df: pd.DataFrame,
    source_train_pred_df: pd.DataFrame,
    source_inner_train_pred_df: pd.DataFrame,
    source_val_pred_df: pd.DataFrame,
    target_test_pred_df_source: pd.DataFrame,
    target_test_pred_df_final: pd.DataFrame,
    target_ft_pred_df: pd.DataFrame,
    target_ft_inner_train_pred_df: pd.DataFrame,
    target_ft_val_pred_df: pd.DataFrame,
    tail_q: float,
    source_all_train_overall_df: pd.DataFrame,
    source_inner_train_overall_df: pd.DataFrame,
    source_val_overall_df: pd.DataFrame,
    target_test_source_overall_df: pd.DataFrame,
    target_test_overall_df: pd.DataFrame,
    source_all_train_cell_metrics_df: pd.DataFrame,
    source_inner_train_cell_metrics_df: pd.DataFrame,
    source_val_cell_metrics_df: pd.DataFrame,
    target_test_source_cell_metrics_df: pd.DataFrame,
    target_test_cell_metrics_df: pd.DataFrame,
    source_all_train_group_metrics_df: pd.DataFrame,
    source_inner_train_group_metrics_df: pd.DataFrame,
    source_val_group_metrics_df: pd.DataFrame,
    target_test_source_group_metrics_df: pd.DataFrame,
    target_test_group_metrics_df: pd.DataFrame,
    target_ft_overall_df: pd.DataFrame,
    target_ft_inner_train_overall_df: pd.DataFrame,
    target_ft_val_overall_df: pd.DataFrame,
    target_ft_cell_metrics_df: pd.DataFrame,
    target_ft_inner_train_cell_metrics_df: pd.DataFrame,
    target_ft_val_cell_metrics_df: pd.DataFrame,
    target_ft_group_metrics_df: pd.DataFrame,
    target_ft_inner_train_group_metrics_df: pd.DataFrame,
    target_ft_val_group_metrics_df: pd.DataFrame,
    has_target_finetune: bool,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    split_train_cells_df = ridge_utils.build_cell_table(source_train_df).sort_values(["group_num", "cell_idx", "cell"]).reset_index(drop=True)
    split_test_cells_df = ridge_utils.build_cell_table(target_test_df).sort_values(["group_num", "cell_idx", "cell"]).reset_index(drop=True)
    split_target_ft_cells_df = (
        ridge_utils.build_cell_table(target_ft_df).sort_values(["group_num", "cell_idx", "cell"]).reset_index(drop=True)
        if has_target_finetune
        else pd.DataFrame(columns=split_train_cells_df.columns)
    )
    source_inner_train_cells_df = (
        split_train_cells_df[split_train_cells_df["cell"].isin(source_stage["train_cells"])].copy().reset_index(drop=True)
    )
    source_val_cells_df = (
        split_train_cells_df[split_train_cells_df["cell"].isin(source_stage["val_cells"])].copy().reset_index(drop=True)
    )
    active_inner_train_cells_df = (
        split_target_ft_cells_df[split_target_ft_cells_df["cell"].isin(active_stage["train_cells"])].copy().reset_index(drop=True)
        if has_target_finetune
        else pd.DataFrame(columns=split_target_ft_cells_df.columns)
    )
    active_val_cells_df = (
        split_target_ft_cells_df[split_target_ft_cells_df["cell"].isin(active_stage["val_cells"])].copy().reset_index(drop=True)
        if has_target_finetune
        else pd.DataFrame(columns=split_target_ft_cells_df.columns)
    )

    meta = {
        "model_type": "mlp_regressor",
        "task_type": "lifetime_prediction",
        "feature_aliases": list(feature_aliases),
        "x_cols": list(x_cols),
        "feature_week": int(args.feature_week),
        "y_col": str(y_col),
        "hidden_dims": list(hidden_dims),
        "dropout": float(args.dropout),
        "activation": str(args.activation),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "mu": np.asarray(args.mu, dtype=float).tolist(),
        "sd": np.asarray(args.sd, dtype=float).tolist(),
        "source_history": source_stage["history"],
        "active_history": active_stage["history"],
        "source_best_epoch": int(source_stage["best_epoch"]),
        "active_best_epoch": int(active_stage["best_epoch"]),
        "source_used_validation": bool(source_stage["used_validation"]),
        "active_used_validation": bool(active_stage["used_validation"]),
        "source_train_cells": list(source_stage["train_cells"]),
        "source_val_cells": list(source_stage["val_cells"]),
        "active_train_cells": list(active_stage["train_cells"]),
        "active_val_cells": list(active_stage["val_cells"]),
        "source_freeze_hidden_layers": int(source_stage["freeze_hidden_layers"]),
        "active_freeze_hidden_layers": int(active_stage["freeze_hidden_layers"]),
        "active_frozen_parameter_names": list(active_stage["frozen_parameter_names"]),
        "target_support_ratio": float(args.target_support_ratio),
        "support_subset_mode": str(args.support_subset_mode),
        "support_subset_seed": int(args.support_subset_seed),
        "min_support_cells": int(args.min_support_cells),
        "support_cells": list(getattr(args, "support_cells", [])),
        "support_cell_count": int(getattr(args, "support_cell_count", 0)),
        "transfer_replay_weight": float(args.transfer_replay_weight),
        "min_target_val_cells": int(args.min_target_val_cells),
        "ft_batch_mode": str(args.ft_batch_mode),
        "ft_selection_mode": str(args.ft_selection_mode),
        "ft_smooth_window": int(args.ft_smooth_window),
        "ft_swa_window": int(args.ft_swa_window),
        "ft_l2sp_weight": float(args.ft_l2sp_weight),
    }

    split_train_cells_df.to_csv(out_dir / "train_cells.csv", index=False)
    split_test_cells_df.to_csv(out_dir / "test_cells.csv", index=False)
    source_inner_train_cells_df.to_csv(out_dir / "source_train_inner_cells.csv", index=False)
    source_val_cells_df.to_csv(out_dir / "source_val_cells.csv", index=False)
    pd.DataFrame(source_stage["history"]).to_csv(out_dir / "training_history_source.csv", index=False)
    mlp_base.maybe_plot_training_history(source_stage["history"], out_dir / "plot_training_history_source.png", "Source Train: Loss vs Epoch")
    source_all_train_overall_df.to_csv(out_dir / "source_all_train_overall_metrics.csv", index=False)
    source_inner_train_overall_df.to_csv(out_dir / "source_inner_train_overall_metrics.csv", index=False)
    source_val_overall_df.to_csv(out_dir / "source_val_overall_metrics.csv", index=False)
    target_test_overall_df.to_csv(out_dir / "test_overall_metrics.csv", index=False)
    source_all_train_cell_metrics_df.to_csv(out_dir / "source_all_train_cell_metrics.csv", index=False)
    source_inner_train_cell_metrics_df.to_csv(out_dir / "source_inner_train_cell_metrics.csv", index=False)
    source_val_cell_metrics_df.to_csv(out_dir / "source_val_cell_metrics.csv", index=False)
    target_test_cell_metrics_df.to_csv(out_dir / "test_cell_metrics.csv", index=False)
    source_all_train_group_metrics_df.to_csv(out_dir / "source_all_train_group_metrics.csv", index=False)
    source_inner_train_group_metrics_df.to_csv(out_dir / "source_inner_train_group_metrics.csv", index=False)
    source_val_group_metrics_df.to_csv(out_dir / "source_val_group_metrics.csv", index=False)
    target_test_group_metrics_df.to_csv(out_dir / "test_group_metrics.csv", index=False)
    source_train_pred_df.to_csv(out_dir / "predictions_source_all_train.csv", index=False)
    source_inner_train_pred_df.to_csv(out_dir / "predictions_source_inner_train.csv", index=False)
    source_val_pred_df.to_csv(out_dir / "predictions_source_val.csv", index=False)
    target_test_pred_df_final.to_csv(out_dir / "predictions_test.csv", index=False)

    if has_target_finetune:
        pd.DataFrame(active_stage["history"]).to_csv(out_dir / "training_history_finetune.csv", index=False)
        mlp_base.maybe_plot_training_history(active_stage["history"], out_dir / "plot_training_history_finetune.png", "Target Fine-tune: Loss vs Epoch")
        target_test_source_overall_df.to_csv(out_dir / "test_overall_metrics_source_only.csv", index=False)
        target_test_source_cell_metrics_df.to_csv(out_dir / "test_cell_metrics_source_only.csv", index=False)
        target_test_source_group_metrics_df.to_csv(out_dir / "test_group_metrics_source_only.csv", index=False)
        split_target_ft_cells_df.to_csv(out_dir / "target_finetune_cells.csv", index=False)
        active_inner_train_cells_df.to_csv(out_dir / "target_finetune_inner_train_cells.csv", index=False)
        active_val_cells_df.to_csv(out_dir / "target_finetune_val_cells.csv", index=False)
        target_ft_overall_df.to_csv(out_dir / "target_finetune_overall_metrics.csv", index=False)
        target_ft_inner_train_overall_df.to_csv(out_dir / "target_finetune_inner_train_overall_metrics.csv", index=False)
        target_ft_val_overall_df.to_csv(out_dir / "target_finetune_val_overall_metrics.csv", index=False)
        target_ft_cell_metrics_df.to_csv(out_dir / "target_finetune_cell_metrics.csv", index=False)
        target_ft_inner_train_cell_metrics_df.to_csv(out_dir / "target_finetune_inner_train_cell_metrics.csv", index=False)
        target_ft_val_cell_metrics_df.to_csv(out_dir / "target_finetune_val_cell_metrics.csv", index=False)
        target_ft_group_metrics_df.to_csv(out_dir / "target_finetune_group_metrics.csv", index=False)
        target_ft_inner_train_group_metrics_df.to_csv(out_dir / "target_finetune_inner_train_group_metrics.csv", index=False)
        target_ft_val_group_metrics_df.to_csv(out_dir / "target_finetune_val_group_metrics.csv", index=False)
        target_test_pred_df_source.to_csv(out_dir / "predictions_test_source_only.csv", index=False)
        target_ft_pred_df.to_csv(out_dir / "predictions_target_finetune.csv", index=False)
        target_ft_inner_train_pred_df.to_csv(out_dir / "predictions_target_finetune_inner_train.csv", index=False)
        target_ft_val_pred_df.to_csv(out_dir / "predictions_target_finetune_val.csv", index=False)

    mlp_base.save_torch_checkpoint(out_dir / "model.pt", {"meta": meta, "state_dict": active_stage["state_dict"]})
    ridge_utils.save_json(out_dir / "model.json", meta)
    ridge_utils.save_json(
        out_dir / "config.json",
        {
            "task_type": "lifetime_prediction",
            "feature_aliases": list(feature_aliases),
            "x_cols": list(x_cols),
            "feature_week": int(args.feature_week),
            "y_col": str(y_col),
            "hidden_dims": list(hidden_dims),
            "dropout": float(args.dropout),
            "activation": str(args.activation),
            "epochs": int(args.epochs),
            "ft_epochs": int(args.ft_epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "ft_lr": float(args.ft_lr),
            "weight_decay": float(args.weight_decay),
            "ft_weight_decay": float(args.ft_weight_decay),
            "val_cell_frac": float(args.val_cell_frac),
            "early_stop_patience": int(args.early_stop_patience),
            "min_epochs_before_early_stop": int(args.min_epochs_before_early_stop),
            "ft_min_epochs_before_early_stop": int(args.ft_min_epochs_before_early_stop),
            "ft_freeze_hidden_layers": int(args.ft_freeze_hidden_layers),
            "target_support_ratio": float(args.target_support_ratio),
            "support_subset_mode": str(args.support_subset_mode),
            "support_subset_seed": int(args.support_subset_seed),
            "min_support_cells": int(args.min_support_cells),
            "support_cell_count": int(getattr(args, "support_cell_count", 0)),
            "transfer_replay_weight": float(args.transfer_replay_weight),
            "min_target_val_cells": int(args.min_target_val_cells),
            "ft_batch_mode": str(args.ft_batch_mode),
            "ft_selection_mode": str(args.ft_selection_mode),
            "ft_smooth_window": int(args.ft_smooth_window),
            "ft_swa_window": int(args.ft_swa_window),
            "ft_l2sp_weight": float(args.ft_l2sp_weight),
            "tail_q": float(tail_q),
            "source_used_validation": bool(source_stage["used_validation"]),
            "active_used_validation": bool(active_stage["used_validation"]),
            "source_best_epoch": int(source_stage["best_epoch"]),
            "active_best_epoch": int(active_stage["best_epoch"]),
        },
    )
    if has_target_finetune:
        mlp_base.save_torch_checkpoint(out_dir / "model_source.pt", {"meta": meta, "state_dict": source_stage["state_dict"]})
        mlp_base.save_torch_checkpoint(out_dir / "model_finetuned.pt", {"meta": meta, "state_dict": active_stage["state_dict"]})

    ridge_utils.maybe_make_plots(
        source_inner_train_pred_df,
        target_ft_inner_train_pred_df,
        target_test_pred_df_final,
        target_test_cell_metrics_df,
        target_test_group_metrics_df,
        target_test_overall_df,
        out_dir,
        target_name="Lifetime",
        train_scatter_label="Source inner-train",
        train_summary_title="Source Inner-Train Summary",
        train_summary_filename="plot_source_inner_train_summary_metrics.png",
    )
    save_prediction_summary_card(source_val_pred_df, "Source Val Summary", out_dir / "plot_source_val_summary_metrics.png", tail_q)
    if has_target_finetune:
        save_prediction_summary_card(
            target_ft_inner_train_pred_df,
            "Target Fine-tune Inner-Train Summary",
            out_dir / "plot_target_finetune_inner_train_summary_metrics.png",
            tail_q,
        )
        save_prediction_summary_card(
            target_ft_val_pred_df,
            "Target Fine-tune Val Summary",
            out_dir / "plot_target_finetune_val_summary_metrics.png",
            tail_q,
        )


def run_experiment(
    *,
    out_dir: Path,
    source_train_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
    target_ft_df: Optional[pd.DataFrame],
    x_cols: Sequence[str],
    feature_aliases: Sequence[str],
    y_col: str,
    hidden_dims: Sequence[int],
    args: argparse.Namespace,
    device,
) -> Dict[str, pd.DataFrame]:
    mu, sd = ridge_utils.standardize_fit(source_train_df[list(x_cols)].to_numpy(dtype=float))
    args.mu = mu
    args.sd = sd

    source_stage = mlp_base.train_stage(
        train_df=source_train_df,
        x_cols=x_cols,
        y_col=y_col,
        mu=mu,
        sd=sd,
        hidden_dims=hidden_dims,
        dropout=float(args.dropout),
        activation=args.activation,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        val_cell_frac=float(args.val_cell_frac),
        early_stop_patience=int(args.early_stop_patience),
        min_epochs_before_early_stop=int(args.min_epochs_before_early_stop),
        seed=int(args.seed),
        device=device,
        freeze_hidden_layers=0,
        log_every=int(args.log_every),
        stage_name="source_train",
        require_validation=True,
    )

    source_train_pred_df = ridge_utils.build_prediction_df(source_train_df, y_col, np.asarray(source_stage["y_pred_train"], dtype=float), "source_train")
    source_inner_train_pred_df = subset_prediction_df_by_cells(source_train_pred_df, source_stage["train_cells"], "source_inner_train")
    source_val_pred_df = subset_prediction_df_by_cells(source_train_pred_df, source_stage["val_cells"], "source_val")
    target_test_pred_df_source = ridge_utils.build_prediction_df(
        target_test_df,
        y_col,
        mlp_base.predict_with_model(source_stage["model"], target_test_df, x_cols, mu, sd, device, int(args.batch_size)),
        "target_test_source_only",
    )

    has_target_finetune = target_ft_df is not None and len(target_ft_df) > 0
    active_stage = source_stage
    target_ft_pred_df = pd.DataFrame()
    target_ft_inner_train_pred_df = pd.DataFrame()
    target_ft_val_pred_df = pd.DataFrame()
    target_test_pred_df_final = target_test_pred_df_source.copy()
    if has_target_finetune:
        active_stage = train_stage_with_replay(
            train_df=target_ft_df,
            replay_df=source_train_df,
            replay_loss_weight=float(args.transfer_replay_weight),
            x_cols=x_cols,
            y_col=y_col,
            mu=mu,
            sd=sd,
            hidden_dims=hidden_dims,
            dropout=float(args.dropout),
            activation=args.activation,
            epochs=int(args.ft_epochs),
            lr=float(args.ft_lr),
            weight_decay=float(args.ft_weight_decay),
            batch_size=int(args.batch_size),
            val_cell_frac=float(args.val_cell_frac),
            early_stop_patience=int(args.early_stop_patience),
            min_epochs_before_early_stop=int(args.ft_min_epochs_before_early_stop),
            seed=int(args.seed) + 1,
            device=device,
            init_state_dict=source_stage["state_dict"],
            freeze_hidden_layers=int(args.ft_freeze_hidden_layers),
            min_val_cells=int(args.min_target_val_cells),
            log_every=int(args.log_every),
            stage_name="target_finetune",
            ft_batch_mode=str(args.ft_batch_mode),
            ft_selection_mode=str(args.ft_selection_mode),
            ft_smooth_window=int(args.ft_smooth_window),
            ft_swa_window=int(args.ft_swa_window),
            ft_l2sp_weight=float(args.ft_l2sp_weight),
        )
        target_ft_pred_df = ridge_utils.build_prediction_df(target_ft_df, y_col, np.asarray(active_stage["y_pred_train"], dtype=float), "target_finetune")
        target_ft_inner_train_pred_df = subset_prediction_df_by_cells(
            target_ft_pred_df,
            active_stage["train_cells"],
            "target_finetune_inner_train",
        )
        target_ft_val_pred_df = subset_prediction_df_by_cells(
            target_ft_pred_df,
            active_stage["val_cells"],
            "target_finetune_val",
        )
        target_test_pred_df_final = ridge_utils.build_prediction_df(
            target_test_df,
            y_col,
            mlp_base.predict_with_model(active_stage["model"], target_test_df, x_cols, mu, sd, device, int(args.batch_size)),
            "target_test_finetuned",
        )

    tail_q = float(args.tail_q)
    source_all_train_overall_df = ridge_utils.summarize_overall(source_train_pred_df, "source_all_train", tail_q)
    source_inner_train_overall_df, source_inner_train_cell_metrics_df, source_inner_train_group_metrics_df = summarize_prediction_split(
        source_inner_train_pred_df,
        "source_inner_train",
        tail_q,
    )
    source_val_overall_df, source_val_cell_metrics_df, source_val_group_metrics_df = summarize_prediction_split(
        source_val_pred_df,
        "source_val",
        tail_q,
    )
    target_test_source_overall_df = ridge_utils.summarize_overall(target_test_pred_df_source, "target_test_source_only", tail_q)
    target_test_overall_df = ridge_utils.summarize_overall(target_test_pred_df_final, "target_test_finetuned" if has_target_finetune else "test", tail_q)
    source_all_train_cell_metrics_df = ridge_utils.summarize_by_cell(source_train_pred_df, "time_week", "time_week", tail_q)
    target_test_source_cell_metrics_df = ridge_utils.summarize_by_cell(target_test_pred_df_source, "time_week", "time_week", tail_q)
    target_test_cell_metrics_df = ridge_utils.summarize_by_cell(target_test_pred_df_final, "time_week", "time_week", tail_q)
    source_all_train_group_metrics_df = ridge_utils.summarize_by_group(source_train_pred_df, source_all_train_cell_metrics_df, tail_q)
    target_test_source_group_metrics_df = ridge_utils.summarize_by_group(target_test_pred_df_source, target_test_source_cell_metrics_df, tail_q)
    target_test_group_metrics_df = ridge_utils.summarize_by_group(target_test_pred_df_final, target_test_cell_metrics_df, tail_q)
    target_ft_overall_df = ridge_utils.summarize_overall(target_ft_pred_df, "target_finetune", tail_q) if has_target_finetune else pd.DataFrame()
    target_ft_inner_train_overall_df, target_ft_inner_train_cell_metrics_df, target_ft_inner_train_group_metrics_df = (
        summarize_prediction_split(target_ft_inner_train_pred_df, "target_finetune_inner_train", tail_q)
        if has_target_finetune
        else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    )
    target_ft_val_overall_df, target_ft_val_cell_metrics_df, target_ft_val_group_metrics_df = (
        summarize_prediction_split(target_ft_val_pred_df, "target_finetune_val", tail_q)
        if has_target_finetune
        else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    )
    target_ft_cell_metrics_df = ridge_utils.summarize_by_cell(target_ft_pred_df, "time_week", "time_week", tail_q) if has_target_finetune else pd.DataFrame()
    target_ft_group_metrics_df = ridge_utils.summarize_by_group(target_ft_pred_df, target_ft_cell_metrics_df, tail_q) if has_target_finetune else pd.DataFrame()

    save_stage_outputs(
        out_dir=out_dir,
        source_stage=source_stage,
        active_stage=active_stage,
        x_cols=x_cols,
        feature_aliases=feature_aliases,
        y_col=y_col,
        device=device,
        hidden_dims=hidden_dims,
        source_train_df=source_train_df,
        target_test_df=target_test_df,
        target_ft_df=target_ft_df if target_ft_df is not None else source_train_df.iloc[0:0].copy(),
        source_train_pred_df=source_train_pred_df,
        source_inner_train_pred_df=source_inner_train_pred_df,
        source_val_pred_df=source_val_pred_df,
        target_test_pred_df_source=target_test_pred_df_source,
        target_test_pred_df_final=target_test_pred_df_final,
        target_ft_pred_df=target_ft_pred_df,
        target_ft_inner_train_pred_df=target_ft_inner_train_pred_df,
        target_ft_val_pred_df=target_ft_val_pred_df,
        tail_q=tail_q,
        source_all_train_overall_df=source_all_train_overall_df,
        source_inner_train_overall_df=source_inner_train_overall_df,
        source_val_overall_df=source_val_overall_df,
        target_test_source_overall_df=target_test_source_overall_df,
        target_test_overall_df=target_test_overall_df,
        source_all_train_cell_metrics_df=source_all_train_cell_metrics_df,
        source_inner_train_cell_metrics_df=source_inner_train_cell_metrics_df,
        source_val_cell_metrics_df=source_val_cell_metrics_df,
        target_test_source_cell_metrics_df=target_test_source_cell_metrics_df,
        target_test_cell_metrics_df=target_test_cell_metrics_df,
        source_all_train_group_metrics_df=source_all_train_group_metrics_df,
        source_inner_train_group_metrics_df=source_inner_train_group_metrics_df,
        source_val_group_metrics_df=source_val_group_metrics_df,
        target_test_source_group_metrics_df=target_test_source_group_metrics_df,
        target_test_group_metrics_df=target_test_group_metrics_df,
        target_ft_overall_df=target_ft_overall_df,
        target_ft_inner_train_overall_df=target_ft_inner_train_overall_df,
        target_ft_val_overall_df=target_ft_val_overall_df,
        target_ft_cell_metrics_df=target_ft_cell_metrics_df,
        target_ft_inner_train_cell_metrics_df=target_ft_inner_train_cell_metrics_df,
        target_ft_val_cell_metrics_df=target_ft_val_cell_metrics_df,
        target_ft_group_metrics_df=target_ft_group_metrics_df,
        target_ft_inner_train_group_metrics_df=target_ft_inner_train_group_metrics_df,
        target_ft_val_group_metrics_df=target_ft_val_group_metrics_df,
        has_target_finetune=has_target_finetune,
        args=args,
    )

    return {
        "source_all_train_overall": source_all_train_overall_df,
        "source_inner_train_overall": source_inner_train_overall_df,
        "source_val_overall": source_val_overall_df,
        "test_overall": target_test_overall_df,
        "test_source_only_overall": target_test_source_overall_df,
        "target_ft_overall": target_ft_overall_df,
        "target_ft_inner_train_overall": target_ft_inner_train_overall_df,
        "target_ft_val_overall": target_ft_val_overall_df,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Lifetime prediction transfer experiment using fixed train/fine-tune/test cell splits.")
    ap.add_argument("--data_csv", type=str, default=r"E:/Datasets/IVAS/Processing_Data/Lifetime_prediction/ivas_lifetime_10features_per_cell.csv")
    ap.add_argument("--split_csv", type=str, default=r"E:/Datasets/IVAS/Processing_Data_dd_exclude/EOL60/cell_split_by_lifetime_EOL60.csv")
    ap.add_argument("--group_cond_csv", type=str, default=r"E:/Datasets/IVAS/Groupcondi.csv")
    ap.add_argument("--out_root", type=str, default=r"E:/Datasets/IVAS/Lifetime_Prediction/EOL60")
    ap.add_argument("--features", type=str, default="f1,f3,f5")
    ap.add_argument("--y_col", type=str, default="lifetime_weeks_EOL60")
    ap.add_argument("--hidden_dims", type=str, default="128,128,128,128")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"])
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--ft_epochs", type=int, default=60)
    ap.add_argument("--ft_freeze_hidden_layers", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft_lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--ft_weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_cell_frac", type=float, default=0.2)
    ap.add_argument("--early_stop_patience", type=int, default=25)
    ap.add_argument("--min_epochs_before_early_stop", type=int, default=0)
    ap.add_argument("--ft_min_epochs_before_early_stop", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--tail_q", type=float, default=0.95)
    ap.add_argument("--target_support_ratio", type=float, default=1.0)
    ap.add_argument("--support_subset_mode", type=str, default="quantile", choices=["quantile", "random", "high_tail"])
    ap.add_argument("--support_subset_seed", type=int, default=17)
    ap.add_argument("--min_support_cells", type=int, default=6)
    ap.add_argument("--transfer_replay_weight", type=float, default=0.0)
    ap.add_argument("--min_target_val_cells", type=int, default=3)
    ap.add_argument("--ft_batch_mode", type=str, default="mini", choices=["mini", "full"])
    ap.add_argument("--ft_selection_mode", type=str, default="raw_best", choices=["raw_best", "smooth_best", "last_window_swa", "final"])
    ap.add_argument("--ft_smooth_window", type=int, default=25)
    ap.add_argument("--ft_swa_window", type=int, default=50)
    ap.add_argument("--ft_l2sp_weight", type=float, default=0.0)
    ap.add_argument("--run_parts", type=str, default="both", choices=["both", "benchmark", "transfer"])
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    feature_aliases, x_cols, feature_week = parse_feature_aliases(args.features)
    args.feature_week = int(feature_week)
    hidden_dims = mlp_base.parse_hidden_dims(args.hidden_dims)
    device = mlp_base.resolve_device(args.device)
    mlp_base.set_seed(int(args.seed))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    benchmark_dir = out_root / "benchmark"
    transfer_dir = out_root / "transfer_model"

    df = load_lifetime_frame(Path(args.data_csv), x_cols, args.y_col, week_num=int(feature_week))
    cond_df = ridge_utils.load_group_conditions(Path(args.group_cond_csv))
    df = ridge_utils.add_condition_columns(df, cond_df)

    split_cells = load_split_cells(Path(args.split_csv))
    train_cells = split_cells.get("train", [])
    ft_cells = split_cells.get("fine_tune", [])
    test_cells = split_cells.get("test", [])
    if not train_cells or not test_cells:
        raise ValueError("Split CSV must provide non-empty train and test cells.")

    source_train_df = mark_active_split(select_cells(df, train_cells, "train"), "train", "Train")
    full_target_ft_df = (
        mark_active_split(select_cells(df, ft_cells, "fine_tune"), "fine_tune", "Fine-tune")
        if ft_cells
        else df.iloc[0:0].copy()
    )
    support_cells: List[str] = []
    target_ft_df = full_target_ft_df
    if len(full_target_ft_df) > 0:
        support_cells = select_support_cells(
            full_target_ft_df,
            y_col=args.y_col,
            support_ratio=float(args.target_support_ratio),
            min_support_cells=int(args.min_support_cells),
            mode=str(args.support_subset_mode),
            seed=int(args.support_subset_seed),
        )
        target_ft_df = mark_active_split(select_cells(df, support_cells, "fine_tune_support"), "fine_tune", "Fine-tune")
    target_test_df = mark_active_split(select_cells(df, test_cells, "test"), "test", "Test")
    args.support_cells = list(support_cells)
    args.support_cell_count = int(len(support_cells))

    benchmark_train_df = build_benchmark_train_df(source_train_df, target_ft_df)

    source_train_df.to_csv(out_root / "source_train_samples.csv", index=False)
    target_ft_df.to_csv(out_root / "target_finetune_samples.csv", index=False)
    full_target_ft_df.to_csv(out_root / "target_finetune_pool_samples.csv", index=False)
    target_test_df.to_csv(out_root / "target_test_samples.csv", index=False)
    benchmark_train_df.to_csv(out_root / "benchmark_train_samples.csv", index=False)

    run_benchmark = args.run_parts in {"both", "benchmark"}
    run_transfer = args.run_parts in {"both", "transfer"}

    benchmark_results = (
        run_experiment(
            out_dir=benchmark_dir,
            source_train_df=benchmark_train_df,
            target_test_df=target_test_df,
            target_ft_df=None,
            x_cols=x_cols,
            feature_aliases=feature_aliases,
            y_col=args.y_col,
            hidden_dims=hidden_dims,
            args=args,
            device=device,
        )
        if run_benchmark
        else load_saved_overall_results(benchmark_dir, include_target_finetune=False)
    )
    transfer_results = (
        run_experiment(
            out_dir=transfer_dir,
            source_train_df=source_train_df,
            target_test_df=target_test_df,
            target_ft_df=target_ft_df,
            x_cols=x_cols,
            feature_aliases=feature_aliases,
            y_col=args.y_col,
            hidden_dims=hidden_dims,
            args=args,
            device=device,
        )
        if run_transfer
        else try_load_saved_overall_results(transfer_dir, include_target_finetune=True)
    )

    comparison_lines: List[str] = [
        "Lifetime transfer experiment summary",
        "",
        f"task          : {args.y_col}",
        f"features      : {','.join(feature_aliases)}",
        f"x_cols        : {','.join(x_cols)}",
        f"feature_week  : w{int(feature_week)}",
        f"hidden_dims   : {hidden_dims}",
        f"device        : {device}",
        f"split_csv     : {args.split_csv}",
        f"train cells   : {len(train_cells)}",
        f"fine-tune pool cells: {len(ft_cells)}",
        f"selected fine-tune cells: {int(args.support_cell_count)}",
        f"benchmark train cells: {benchmark_train_df['cell'].nunique()}",
        f"test cells    : {len(test_cells)}",
        f"target_support_ratio: {float(args.target_support_ratio):.4f}",
        f"support_subset_mode: {args.support_subset_mode}",
        f"transfer_replay_weight: {float(args.transfer_replay_weight):.4f}",
        f"ft_batch_mode : {args.ft_batch_mode}",
        f"ft_selection_mode: {args.ft_selection_mode}",
        f"ft_smooth_window: {int(args.ft_smooth_window)}",
        f"ft_swa_window : {int(args.ft_swa_window)}",
        f"ft_l2sp_weight: {float(args.ft_l2sp_weight):.6g}",
        f"run_parts     : {args.run_parts}",
    ]
    append_summary_block(comparison_lines, "Benchmark source inner-train", benchmark_results["source_inner_train_overall"])
    append_summary_block(comparison_lines, "Benchmark source val", benchmark_results["source_val_overall"])
    append_summary_block(comparison_lines, "Benchmark target test", benchmark_results["test_overall"])
    if transfer_results is None:
        comparison_lines.extend(
            [
                "",
                "Transfer results",
                "Unavailable: existing transfer summary CSVs were not found for this root.",
            ]
        )
    else:
        append_summary_block(comparison_lines, "Transfer source inner-train", transfer_results.get("source_inner_train_overall"))
        append_summary_block(comparison_lines, "Transfer source val", transfer_results.get("source_val_overall"))
        append_summary_block(comparison_lines, "Transfer target fine-tune inner-train", transfer_results.get("target_ft_inner_train_overall"))
        append_summary_block(comparison_lines, "Transfer target fine-tune val", transfer_results.get("target_ft_val_overall"))
        append_summary_block(comparison_lines, "Transfer target fine-tune combined", transfer_results.get("target_ft_overall"))
        append_summary_block(comparison_lines, "Transfer target test source-only", transfer_results.get("test_source_only_overall"))
        append_summary_block(comparison_lines, "Transfer target test finetuned", transfer_results.get("test_overall"))
    (out_root / "report_summary.txt").write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")

    print("[INFO] Lifetime transfer experiment finished.")
    print(f"[INFO] out_root        : {out_root}")
    print_summary_metric("benchmark source inner", benchmark_results.get("source_inner_train_overall"))
    print_summary_metric("benchmark source val", benchmark_results.get("source_val_overall"))
    print_summary_metric("benchmark target test", benchmark_results.get("test_overall"))
    if transfer_results is None:
        print("[INFO] transfer summary        : unavailable (existing transfer summary CSVs not found)")
    else:
        print_summary_metric("transfer source inner", transfer_results.get("source_inner_train_overall"))
        print_summary_metric("transfer source val", transfer_results.get("source_val_overall"))
        print_summary_metric("transfer ft inner", transfer_results.get("target_ft_inner_train_overall"))
        print_summary_metric("transfer ft val", transfer_results.get("target_ft_val_overall"))
        print_summary_metric("transfer target ft", transfer_results.get("target_ft_overall"))
        print_summary_metric("transfer test src-only", transfer_results.get("test_source_only_overall"))
        print_summary_metric("transfer test final", transfer_results.get("test_overall"))


if __name__ == "__main__":
    main()

