#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict

import pandas as pd

from three_step_common import DEFAULT_DATA_CSV, DEFAULT_GROUP_CSV, DEFAULT_PYTHON, DEFAULT_SPLIT_CSV, WORKSPACE_ROOT, ensure_child_env, save_json


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_single_row_csv(path: Path) -> Dict[str, object]:
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"Expected one row in {path}, got {len(df)}")
    return df.iloc[0].to_dict()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage 3 final evaluation and final-package export.")
    ap.add_argument("--python_exe", type=str, default=DEFAULT_PYTHON)
    ap.add_argument("--runner", type=str, default=str(WORKSPACE_ROOT / "codes" / "run_lifetime_transfer_mlp.py"))
    ap.add_argument("--data_csv", type=str, default=str(DEFAULT_DATA_CSV))
    ap.add_argument("--split_csv", type=str, default=str(DEFAULT_SPLIT_CSV))
    ap.add_argument("--group_cond_csv", type=str, default=str(DEFAULT_GROUP_CSV))
    ap.add_argument("--stage2_best_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--y_col", type=str, default="lifetime_weeks_EOL70")
    ap.add_argument("--dropout", type=float, default=-1.0)
    ap.add_argument("--activation", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_cell_frac", type=float, default=0.2)
    ap.add_argument("--early_stop_patience", type=int, default=25)
    ap.add_argument("--min_source_epochs", type=int, default=400)
    ap.add_argument("--min_ft_epochs", type=int, default=400)
    ap.add_argument("--min_epochs_before_early_stop", type=int, default=400)
    ap.add_argument("--ft_min_epochs_before_early_stop", type=int, default=400)
    ap.add_argument("--support_subset_mode", type=str, default="quantile", choices=["quantile", "random", "high_tail"])
    ap.add_argument("--support_subset_seed", type=int, default=17)
    ap.add_argument("--min_support_cells", type=int, default=6)
    ap.add_argument("--min_target_val_cells", type=int, default=3)
    ap.add_argument("--ft_batch_mode", type=str, default="mini", choices=["mini", "full"])
    ap.add_argument("--ft_selection_mode", type=str, default="raw_best", choices=["raw_best", "smooth_best", "last_window_swa", "final"])
    ap.add_argument("--ft_smooth_window", type=int, default=25)
    ap.add_argument("--ft_swa_window", type=int, default=50)
    ap.add_argument("--ft_l2sp_weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=101)
    return ap


def resolve_source_checkpoint(path_text: str, stage2_best_csv: Path) -> Path:
    raw_path = Path(str(path_text))
    if raw_path.exists():
        return raw_path
    parts = raw_path.parts
    if "stage1_runs" not in parts:
        return raw_path
    idx = parts.index("stage1_runs")
    trial_rel = Path(*parts[idx:])
    root = stage2_best_csv.parent.parent
    stage2_name = stage2_best_csv.parent.name.lower()
    stage1_names = ["stage1_expanded", "stage1"] if "expanded" in stage2_name else ["stage1", "stage1_expanded"]
    for stage1_name in stage1_names:
        candidate = root / stage1_name / trial_rel
        if candidate.exists():
            return candidate
    return raw_path


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_df = pd.read_csv(args.stage2_best_csv)
    if len(best_df) == 0:
        raise ValueError(f"No best config found in {args.stage2_best_csv}")
    best = best_df.iloc[0]
    best_row = best.to_dict()

    source_checkpoint = resolve_source_checkpoint(str(best["source_checkpoint"]), Path(args.stage2_best_csv))
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_checkpoint}")
    best_row["source_checkpoint"] = str(source_checkpoint)
    source_trial_dir = source_checkpoint.parent
    source_trial_summary_path = source_trial_dir / "trial_summary.json"
    if not source_trial_summary_path.exists():
        raise FileNotFoundError(f"Stage1 trial summary not found: {source_trial_summary_path}")
    source_summary = read_json(source_trial_summary_path)
    final_dropout = float(best["dropout"]) if "dropout" in best_df.columns and pd.notna(best["dropout"]) else (
        float(source_summary["dropout"]) if float(args.dropout) < 0.0 else float(args.dropout)
    )
    final_activation = str(best["activation"]).strip() if "activation" in best_df.columns and str(best["activation"]).strip() else (
        str(source_summary["activation"]).strip() if not str(args.activation).strip() else str(args.activation).strip()
    )
    final_source_epochs = max(int(source_summary["epochs"]), int(args.min_source_epochs))
    final_ft_epochs = max(int(best["ft_epochs"]), int(args.min_ft_epochs))
    final_source_min_early_stop = max(int(args.min_epochs_before_early_stop), int(args.min_source_epochs))
    final_ft_min_early_stop = max(int(args.ft_min_epochs_before_early_stop), int(args.min_ft_epochs))

    cmd = [
        str(args.python_exe),
        str(args.runner),
        "--data_csv",
        str(args.data_csv),
        "--split_csv",
        str(args.split_csv),
        "--group_cond_csv",
        str(args.group_cond_csv),
        "--out_root",
        str(out_dir),
        "--features",
        str(best["features"]),
        "--y_col",
        args.y_col,
        "--hidden_dims",
        str(best["hidden_dims"]),
        "--epochs",
        str(final_source_epochs),
        "--dropout",
        str(final_dropout),
        "--activation",
        final_activation,
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(float(source_summary["lr"])),
        "--weight_decay",
        str(float(source_summary["weight_decay"])),
        "--val_cell_frac",
        str(args.val_cell_frac),
        "--early_stop_patience",
        str(args.early_stop_patience),
        "--min_epochs_before_early_stop",
        str(final_source_min_early_stop),
        "--ft_min_epochs_before_early_stop",
        str(final_ft_min_early_stop),
        "--seed",
        str(args.seed),
        "--ft_lr",
        str(best["ft_lr"]),
        "--ft_weight_decay",
        str(best["ft_weight_decay"]),
        "--ft_epochs",
        str(final_ft_epochs),
        "--ft_freeze_hidden_layers",
        str(int(best["ft_freeze_hidden_layers"])),
        "--target_support_ratio",
        str(best["target_support_ratio"]),
        "--support_subset_mode",
        str(args.support_subset_mode),
        "--support_subset_seed",
        str(args.support_subset_seed),
        "--min_support_cells",
        str(args.min_support_cells),
        "--transfer_replay_weight",
        str(best["transfer_replay_weight"]),
        "--min_target_val_cells",
        str(args.min_target_val_cells),
        "--ft_batch_mode",
        str(args.ft_batch_mode),
        "--ft_selection_mode",
        str(args.ft_selection_mode),
        "--ft_smooth_window",
        str(args.ft_smooth_window),
        "--ft_swa_window",
        str(args.ft_swa_window),
        "--ft_l2sp_weight",
        str(args.ft_l2sp_weight),
        "--log_every",
        "100",
    ]
    subprocess.run(cmd, check=True, env=ensure_child_env())

    transfer_config = {
        "runner": str(args.runner),
        "features": str(best["features"]),
        "hidden_dims": str(best["hidden_dims"]),
        "source_checkpoint": str(best["source_checkpoint"]),
        "source_lr": float(source_summary["lr"]),
        "source_weight_decay": float(source_summary["weight_decay"]),
        "source_epochs": int(final_source_epochs),
        "source_min_epochs_before_early_stop": int(final_source_min_early_stop),
        "ft_lr": float(best["ft_lr"]),
        "ft_weight_decay": float(best["ft_weight_decay"]),
        "ft_epochs": int(final_ft_epochs),
        "ft_min_epochs_before_early_stop": int(final_ft_min_early_stop),
        "ft_freeze_hidden_layers": int(best["ft_freeze_hidden_layers"]),
        "target_support_ratio": float(best["target_support_ratio"]),
        "support_subset_mode": str(args.support_subset_mode),
        "support_subset_seed": int(args.support_subset_seed),
        "min_support_cells": int(args.min_support_cells),
        "transfer_replay_weight": float(best["transfer_replay_weight"]),
        "min_target_val_cells": int(args.min_target_val_cells),
        "ft_batch_mode": str(args.ft_batch_mode),
        "ft_selection_mode": str(args.ft_selection_mode),
        "ft_smooth_window": int(args.ft_smooth_window),
        "ft_swa_window": int(args.ft_swa_window),
        "ft_l2sp_weight": float(args.ft_l2sp_weight),
        "dropout": float(final_dropout),
        "activation": final_activation,
        "batch_size": int(args.batch_size),
        "y_col": args.y_col,
        "data_csv": str(args.data_csv),
        "split_csv": str(args.split_csv),
        "group_cond_csv": str(args.group_cond_csv),
    }
    pd.DataFrame([best_row]).to_csv(out_dir / "selected_stage2_config.csv", index=False, encoding="utf-8-sig")
    save_json(
        out_dir / "final_selection.json",
        {
            "stage2_best_csv": str(args.stage2_best_csv),
            "selected_stage2_row": best_row,
            "selected_stage1_trial_dir": str(source_trial_dir),
            "selected_stage1_summary": source_summary,
            "final_runner_config": transfer_config,
        },
    )
    if (out_dir / "transfer_model" / "target_finetune_cells.csv").exists():
        shutil.copy2(out_dir / "transfer_model" / "target_finetune_cells.csv", out_dir / "support_cells.csv")
    transfer_test = read_single_row_csv(out_dir / "transfer_model" / "test_overall_metrics.csv")
    transfer_ft_val = read_single_row_csv(out_dir / "transfer_model" / "target_finetune_val_overall_metrics.csv")
    benchmark_test = read_single_row_csv(out_dir / "benchmark" / "test_overall_metrics.csv")
    summary_report = {
        "selected_stage2_row": best_row,
        "selected_stage1_summary": source_summary,
        "final_package_root": str(out_dir),
        "final_source_epochs": int(final_source_epochs),
        "final_ft_epochs": int(final_ft_epochs),
        "final_source_min_epochs_before_early_stop": int(final_source_min_early_stop),
        "final_ft_min_epochs_before_early_stop": int(final_ft_min_early_stop),
        "benchmark_test_overall": benchmark_test,
        "transfer_target_finetune_val_overall": transfer_ft_val,
        "transfer_test_overall": transfer_test,
    }
    save_json(out_dir / "stage3_final_report.json", summary_report)
    print(json.dumps(summary_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
