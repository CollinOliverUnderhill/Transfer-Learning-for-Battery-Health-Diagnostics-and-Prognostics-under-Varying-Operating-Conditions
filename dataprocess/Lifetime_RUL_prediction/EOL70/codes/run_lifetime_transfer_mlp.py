#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def resolve_mlp_base_path() -> Path:
    current = Path(__file__).resolve()
    candidates = [
        current.parents[1] / "SOHest" / "MLP_codes" / "domain_train_test_by_group_mlp.py",
        current.parents[2] / "Codes" / "chunqiu_codes" / "SOHest" / "MLP_codes" / "domain_train_test_by_group_mlp.py",
        Path(r"E:\Datasets\IVAS\Codes\chunqiu_codes\SOHest\MLP_codes\domain_train_test_by_group_mlp.py"),
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


FEATURE_ALIAS_RE = re.compile(r"^(f(?:10|[1-9]))(?:_w(3|5|10|15))?$")


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
        f"MAPE median: {float(stats['mape_percent_median']):.3f}%"
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
        active_stage = mlp_base.train_stage(
            train_df=target_ft_df,
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
            log_every=int(args.log_every),
            stage_name="target_finetune",
            require_validation=True,
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

    source_train_df = select_cells(df, train_cells, "train")
    target_ft_df = select_cells(df, ft_cells, "fine_tune") if ft_cells else df.iloc[0:0].copy()
    target_test_df = select_cells(df, test_cells, "test")

    benchmark_train_df = build_benchmark_train_df(source_train_df, target_ft_df)

    source_train_df.to_csv(out_root / "source_train_samples.csv", index=False)
    target_ft_df.to_csv(out_root / "target_finetune_samples.csv", index=False)
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
        f"fine-tune cells: {len(ft_cells)}",
        f"benchmark train cells: {benchmark_train_df['cell'].nunique()}",
        f"test cells    : {len(test_cells)}",
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

