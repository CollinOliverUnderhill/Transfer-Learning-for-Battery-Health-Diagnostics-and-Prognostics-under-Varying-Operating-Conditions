#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Dict, List

import optuna
import pandas as pd

from three_step_common import (
    DEFAULT_DATA_CSV,
    DEFAULT_GROUP_CSV,
    DEFAULT_PYTHON,
    DEFAULT_RUNNER,
    DEFAULT_SPLIT_CSV,
    ensure_child_env,
    save_json,
    sort_frame_for_stage,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage 2 fine-tune search with Optuna TPE.")
    ap.add_argument("--python_exe", type=str, default=DEFAULT_PYTHON)
    ap.add_argument("--runner", type=str, default=str(DEFAULT_RUNNER))
    ap.add_argument("--data_csv", type=str, default=str(DEFAULT_DATA_CSV))
    ap.add_argument("--split_csv", type=str, default=str(DEFAULT_SPLIT_CSV))
    ap.add_argument("--group_cond_csv", type=str, default=str(DEFAULT_GROUP_CSV))
    ap.add_argument("--stage1_top_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--study_name", type=str, default="stage2_finetune_search_w5")
    ap.add_argument("--n_trials", type=int, default=160)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--y_col", type=str, default="lifetime_weeks_EOL70")
    ap.add_argument("--dropout", type=float, default=-1.0)
    ap.add_argument("--activation", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_cell_frac", type=float, default=0.2)
    ap.add_argument("--early_stop_patience", type=int, default=25)
    ap.add_argument("--ft_min_epochs_before_early_stop", type=int, default=0)
    ap.add_argument("--stage2_finetune_selection", type=str, default="fixed_budget", choices=["fixed_budget", "early_stop"])
    ap.add_argument("--min_target_val_cells", type=int, default=3)
    ap.add_argument("--support_subset_mode", type=str, default="quantile", choices=["quantile", "random", "high_tail"])
    ap.add_argument("--support_subset_seed", type=int, default=17)
    ap.add_argument("--min_support_cells", type=int, default=6)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--support_ratios", type=str, default="0.5,0.67,0.8,1.0")
    ap.add_argument("--ft_epoch_choices", type=str, default="200,400,800,1200,1600")
    ap.add_argument("--replay_weight_choices", type=str, default="0.0,0.05,0.1,0.3,0.5,1.0,2.0")
    ap.add_argument("--ft_freeze_hidden_layers_min", type=int, default=0)
    ap.add_argument("--ft_lr_min", type=float, default=1e-5)
    ap.add_argument("--ft_lr_max", type=float, default=1e-2)
    ap.add_argument("--ft_weight_decay_min", type=float, default=1e-7)
    ap.add_argument("--ft_weight_decay_max", type=float, default=1e-3)
    ap.add_argument("--stage2_repeats", type=int, default=1)
    ap.add_argument("--ft_batch_mode", type=str, default="mini", choices=["mini", "full"])
    ap.add_argument("--ft_selection_mode", type=str, default="raw_best", choices=["raw_best", "smooth_best", "last_window_swa", "final"])
    ap.add_argument("--ft_smooth_window", type=int, default=25)
    ap.add_argument("--ft_swa_window", type=int, default=50)
    ap.add_argument("--ft_l2sp_weight", type=float, default=0.0)
    ap.add_argument("--source_only_gate_margin", type=float, default=0.0)
    ap.add_argument("--source_only_gate_penalty", type=float, default=0.0)
    ap.add_argument("--source_only_gate_hard", action="store_true")
    return ap


def resolve_source_checkpoint(path_text: str, stage1_top_csv: Path) -> str:
    raw_path = Path(str(path_text))
    if raw_path.exists():
        return str(raw_path)
    local_candidate = stage1_top_csv.parent / "stage1_runs" / raw_path.parent.name / raw_path.name
    if local_candidate.exists():
        return str(local_candidate)
    parts = raw_path.parts
    if "stage1_runs" in parts:
        idx = parts.index("stage1_runs")
        trial_rel = Path(*parts[idx:])
        local_candidate = stage1_top_csv.parent / trial_rel
        if local_candidate.exists():
            return str(local_candidate)
    return str(raw_path)


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "stage2_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    study_db = out_dir / "stage2_optuna.db"

    source_df = pd.read_csv(args.stage1_top_csv)
    if len(source_df) == 0:
        raise ValueError(f"No source checkpoints found in {args.stage1_top_csv}")
    source_df = source_df.reset_index(drop=True)
    source_df["source_checkpoint"] = [
        resolve_source_checkpoint(value, Path(args.stage1_top_csv)) for value in source_df["source_checkpoint"].astype(str)
    ]
    source_choices = [str(i) for i in range(len(source_df))]
    max_hidden_layers = max(len(str(v).split(",")) for v in source_df["hidden_dims"].astype(str))
    max_freeze_hidden_layers = max(0, int(max_hidden_layers) - 1)

    support_ratios = [float(x.strip()) for x in str(args.support_ratios).split(",") if x.strip()]
    ft_epoch_choices = [int(x.strip()) for x in str(args.ft_epoch_choices).split(",") if x.strip()]
    replay_weight_choices = [float(x.strip()) for x in str(args.replay_weight_choices).split(",") if x.strip()]

    sampler = optuna.samplers.TPESampler(seed=int(args.seed), multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///{study_db}",
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        source_idx = int(trial.suggest_categorical("source_idx", source_choices))
        source_row = source_df.iloc[source_idx]
        ft_lr = trial.suggest_float("ft_lr", float(args.ft_lr_min), float(args.ft_lr_max), log=True)
        ft_weight_decay = trial.suggest_float("ft_weight_decay", float(args.ft_weight_decay_min), float(args.ft_weight_decay_max), log=True)
        support_ratio = float(trial.suggest_categorical("target_support_ratio", support_ratios))
        ft_epochs = int(trial.suggest_categorical("ft_epochs", ft_epoch_choices))
        replay_weight = float(trial.suggest_categorical("transfer_replay_weight", replay_weight_choices))
        n_hidden_layers = len(str(source_row["hidden_dims"]).split(","))
        freeze_layers_min = min(max(0, int(args.ft_freeze_hidden_layers_min)), int(max_freeze_hidden_layers))
        freeze_layers_raw = int(trial.suggest_int("ft_freeze_hidden_layers_raw", freeze_layers_min, int(max_freeze_hidden_layers)))
        freeze_layers = min(freeze_layers_raw, max(0, n_hidden_layers - 1))
        ft_min_epochs_before_early_stop = int(ft_epochs) if str(args.stage2_finetune_selection) == "fixed_budget" else int(args.ft_min_epochs_before_early_stop)
        source_dropout = float(source_row["dropout"]) if "dropout" in source_row and pd.notna(source_row["dropout"]) else float(args.dropout if float(args.dropout) >= 0.0 else 0.0)
        source_activation = str(source_row["activation"]) if "activation" in source_row and str(source_row["activation"]).strip() else (str(args.activation).strip() or "relu")

        trial_dir = runs_dir / f"trial_{trial.number:04d}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)

        repeat_summaries: List[Dict[str, object]] = []
        repeat_dirs: List[str] = []
        n_repeats = max(1, int(args.stage2_repeats))
        for repeat_idx in range(n_repeats):
            repeat_dir = trial_dir if n_repeats == 1 else trial_dir / f"repeat_{repeat_idx:02d}"
            repeat_dirs.append(str(repeat_dir))
            cmd = [
                str(args.python_exe),
                str(args.runner),
                "--mode",
                "transfer",
                "--data_csv",
                str(args.data_csv),
                "--split_csv",
                str(args.split_csv),
                "--group_cond_csv",
                str(args.group_cond_csv),
                "--out_root",
                str(repeat_dir),
                "--features",
                str(source_row["features"]),
                "--y_col",
                args.y_col,
                "--hidden_dims",
                str(source_row["hidden_dims"]),
                "--dropout",
                str(source_dropout),
                "--activation",
                source_activation,
                "--batch_size",
                str(args.batch_size),
                "--val_cell_frac",
                str(args.val_cell_frac),
                "--early_stop_patience",
                str(args.early_stop_patience),
                "--ft_min_epochs_before_early_stop",
                str(ft_min_epochs_before_early_stop),
                "--min_target_val_cells",
                str(args.min_target_val_cells),
                "--support_subset_mode",
                str(args.support_subset_mode),
                "--support_subset_seed",
                str(args.support_subset_seed),
                "--min_support_cells",
                str(args.min_support_cells),
                "--seed",
                str(int(args.seed) + int(trial.number) * 100 + repeat_idx),
                "--source_checkpoint",
                str(source_row["source_checkpoint"]),
                "--ft_lr",
                str(ft_lr),
                "--ft_weight_decay",
                str(ft_weight_decay),
                "--ft_epochs",
                str(ft_epochs),
                "--ft_freeze_hidden_layers",
                str(freeze_layers),
                "--target_support_ratio",
                str(support_ratio),
                "--transfer_replay_weight",
                str(replay_weight),
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
            try:
                subprocess.run(cmd, check=True, env=ensure_child_env())
            except subprocess.CalledProcessError as exc:
                invalid_path = repeat_dir / "trial_invalid.json"
                if invalid_path.exists():
                    invalid = json.loads(invalid_path.read_text(encoding="utf-8"))
                    raise optuna.TrialPruned(invalid.get("reason", "invalid trial")) from exc
                raise
            repeat_summaries.append(json.loads((repeat_dir / "trial_summary.json").read_text(encoding="utf-8")))

        summary = dict(repeat_summaries[0])
        for key in [
            "target_ft_val_mae",
            "target_ft_val_rmse",
            "target_ft_val_mape_percent_mean",
            "target_ft_val_smape_percent_mean",
            "target_ft_val_wmape_percent",
            "target_ft_val_source_only_mae",
            "target_ft_val_source_only_rmse",
            "target_ft_val_source_only_mape_percent_mean",
            "target_ft_val_vs_source_only_mae_improve_percent",
            "target_ft_val_vs_source_only_mape_improve_percent",
        ]:
            if key in repeat_summaries[0]:
                summary[key] = float(statistics.median(float(item[key]) for item in repeat_summaries))
        source_only_val_mae = float(summary.get("target_ft_val_source_only_mae", float("nan")))
        ft_val_mae = float(summary["target_ft_val_mae"])
        required_mae = source_only_val_mae * (1.0 - float(args.source_only_gate_margin)) if source_only_val_mae > 0 else float("nan")
        gate_deficit_mae = max(0.0, ft_val_mae - required_mae) if source_only_val_mae > 0 else 0.0
        objective_score = ft_val_mae + float(args.source_only_gate_penalty) * gate_deficit_mae
        if bool(args.source_only_gate_hard) and gate_deficit_mae > 0.0:
            objective_score += 1_000_000.0 + gate_deficit_mae
        summary["source_only_gate_required_mae"] = float(required_mae)
        summary["source_only_gate_deficit_mae"] = float(gate_deficit_mae)
        summary["source_only_gate_pass"] = bool(gate_deficit_mae <= 0.0)
        summary["stage2_objective_score"] = float(objective_score)
        summary["stage2_repeats"] = int(n_repeats)
        summary["repeat_trial_dirs"] = repeat_dirs
        if n_repeats > 1:
            save_json(trial_dir / "trial_summary.json", summary)
        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("repeat_trial_dirs", repeat_dirs)
        trial.set_user_attr("source_idx", source_idx)
        trial.set_user_attr("source_checkpoint", str(source_row["source_checkpoint"]))
        trial.set_user_attr("features", str(source_row["features"]))
        trial.set_user_attr("hidden_dims", str(source_row["hidden_dims"]))
        trial.set_user_attr("dropout", source_dropout)
        trial.set_user_attr("activation", source_activation)
        trial.set_user_attr("ft_freeze_hidden_layers", freeze_layers)
        trial.set_user_attr("ft_min_epochs_before_early_stop", ft_min_epochs_before_early_stop)
        trial.set_user_attr("stage2_repeats", int(n_repeats))
        trial.set_user_attr("ft_batch_mode", str(args.ft_batch_mode))
        trial.set_user_attr("ft_selection_mode", str(args.ft_selection_mode))
        trial.set_user_attr("ft_smooth_window", int(args.ft_smooth_window))
        trial.set_user_attr("ft_swa_window", int(args.ft_swa_window))
        trial.set_user_attr("ft_l2sp_weight", float(args.ft_l2sp_weight))
        trial.set_user_attr("target_ft_val_mae", summary["target_ft_val_mae"])
        trial.set_user_attr("target_ft_val_rmse", summary["target_ft_val_rmse"])
        trial.set_user_attr("target_ft_val_mape_percent_mean", summary["target_ft_val_mape_percent_mean"])
        trial.set_user_attr("target_ft_val_smape_percent_mean", summary["target_ft_val_smape_percent_mean"])
        trial.set_user_attr("target_ft_val_wmape_percent", summary["target_ft_val_wmape_percent"])
        trial.set_user_attr("target_ft_val_cell_count", int(summary["target_ft_val_cell_count"]))
        trial.set_user_attr("target_ft_val_source_only_mae", float(summary.get("target_ft_val_source_only_mae", float("nan"))))
        trial.set_user_attr("target_ft_val_source_only_rmse", float(summary.get("target_ft_val_source_only_rmse", float("nan"))))
        trial.set_user_attr("target_ft_val_source_only_mape_percent_mean", float(summary.get("target_ft_val_source_only_mape_percent_mean", float("nan"))))
        trial.set_user_attr("target_ft_val_vs_source_only_mae_improve_percent", float(summary.get("target_ft_val_vs_source_only_mae_improve_percent", float("nan"))))
        trial.set_user_attr("target_ft_val_vs_source_only_mape_improve_percent", float(summary.get("target_ft_val_vs_source_only_mape_improve_percent", float("nan"))))
        trial.set_user_attr("source_only_gate_required_mae", float(summary["source_only_gate_required_mae"]))
        trial.set_user_attr("source_only_gate_deficit_mae", float(summary["source_only_gate_deficit_mae"]))
        trial.set_user_attr("source_only_gate_pass", bool(summary["source_only_gate_pass"]))
        trial.set_user_attr("stage2_objective_score", float(objective_score))
        return float(objective_score)

    study.optimize(objective, n_trials=int(args.n_trials))

    rows: List[Dict[str, object]] = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        source_idx = int(trial.user_attrs["source_idx"])
        source_row = source_df.iloc[source_idx]
        rows.append(
            {
                "trial_number": int(trial.number),
                "objective": float(trial.value),
                "trial_dir": trial.user_attrs["trial_dir"],
                "repeat_trial_dirs": json.dumps(trial.user_attrs.get("repeat_trial_dirs", []), ensure_ascii=False),
                "source_rank_idx": source_idx,
                "source_checkpoint": trial.user_attrs["source_checkpoint"],
                "features": trial.user_attrs["features"],
                "hidden_dims": trial.user_attrs["hidden_dims"],
                "dropout": float(trial.user_attrs["dropout"]),
                "activation": str(trial.user_attrs["activation"]),
                "source_stage1_val_mae": float(source_row["source_val_mae"]),
                "ft_lr": float(trial.params["ft_lr"]),
                "ft_weight_decay": float(trial.params["ft_weight_decay"]),
                "ft_epochs": int(trial.params["ft_epochs"]),
                "ft_min_epochs_before_early_stop": int(trial.user_attrs["ft_min_epochs_before_early_stop"]),
                "ft_freeze_hidden_layers_raw": int(trial.params["ft_freeze_hidden_layers_raw"]),
                "ft_freeze_hidden_layers": int(trial.user_attrs["ft_freeze_hidden_layers"]),
                "stage2_repeats": int(trial.user_attrs.get("stage2_repeats", 1)),
                "ft_batch_mode": str(trial.user_attrs.get("ft_batch_mode", "mini")),
                "ft_selection_mode": str(trial.user_attrs.get("ft_selection_mode", "raw_best")),
                "ft_smooth_window": int(trial.user_attrs.get("ft_smooth_window", 25)),
                "ft_swa_window": int(trial.user_attrs.get("ft_swa_window", 50)),
                "ft_l2sp_weight": float(trial.user_attrs.get("ft_l2sp_weight", 0.0)),
                "target_support_ratio": float(trial.params["target_support_ratio"]),
                "transfer_replay_weight": float(trial.params["transfer_replay_weight"]),
                "stage2_objective_score": float(trial.user_attrs.get("stage2_objective_score", trial.value)),
                "source_only_gate_required_mae": float(trial.user_attrs.get("source_only_gate_required_mae", float("nan"))),
                "source_only_gate_deficit_mae": float(trial.user_attrs.get("source_only_gate_deficit_mae", float("nan"))),
                "source_only_gate_pass": bool(trial.user_attrs.get("source_only_gate_pass", True)),
                "target_ft_val_mae": float(trial.user_attrs["target_ft_val_mae"]),
                "target_ft_val_rmse": float(trial.user_attrs["target_ft_val_rmse"]),
                "target_ft_val_mape_percent_mean": float(trial.user_attrs["target_ft_val_mape_percent_mean"]),
                "target_ft_val_smape_percent_mean": float(trial.user_attrs["target_ft_val_smape_percent_mean"]),
                "target_ft_val_wmape_percent": float(trial.user_attrs["target_ft_val_wmape_percent"]),
                "target_ft_val_source_only_mae": float(trial.user_attrs.get("target_ft_val_source_only_mae", float("nan"))),
                "target_ft_val_source_only_rmse": float(trial.user_attrs.get("target_ft_val_source_only_rmse", float("nan"))),
                "target_ft_val_source_only_mape_percent_mean": float(trial.user_attrs.get("target_ft_val_source_only_mape_percent_mean", float("nan"))),
                "target_ft_val_vs_source_only_mae_improve_percent": float(trial.user_attrs.get("target_ft_val_vs_source_only_mae_improve_percent", float("nan"))),
                "target_ft_val_vs_source_only_mape_improve_percent": float(trial.user_attrs.get("target_ft_val_vs_source_only_mape_improve_percent", float("nan"))),
                "target_ft_val_cell_count": int(trial.user_attrs["target_ft_val_cell_count"]),
            }
        )

    trials_df = pd.DataFrame(rows).sort_values(
        ["objective", "target_ft_val_mae", "target_ft_val_rmse", "target_ft_val_mape_percent_mean"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    trials_path = out_dir / "stage2_trials.csv"
    trials_df.to_csv(trials_path, index=False, encoding="utf-8-sig")

    top_df = trials_df.head(int(args.top_k)).copy()
    top_path = out_dir / "stage2_best_configs.csv"
    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")

    save_json(
        out_dir / "stage2_manifest.json",
        {
            "study_name": args.study_name,
            "study_db": str(study_db),
            "trials_csv": str(trials_path),
            "best_configs_csv": str(top_path),
            "n_trials": int(args.n_trials),
            "source_checkpoint_pool": int(len(source_df)),
            "max_hidden_layers": int(max_hidden_layers),
            "max_freeze_hidden_layers": int(max_freeze_hidden_layers),
            "min_target_val_cells": int(args.min_target_val_cells),
            "support_subset_mode": str(args.support_subset_mode),
            "support_subset_seed": int(args.support_subset_seed),
            "min_support_cells": int(args.min_support_cells),
            "stage2_finetune_selection": str(args.stage2_finetune_selection),
            "ft_lr_min": float(args.ft_lr_min),
            "ft_lr_max": float(args.ft_lr_max),
            "ft_weight_decay_min": float(args.ft_weight_decay_min),
            "ft_weight_decay_max": float(args.ft_weight_decay_max),
            "ft_freeze_hidden_layers_min": int(args.ft_freeze_hidden_layers_min),
            "support_ratios": support_ratios,
            "ft_epoch_choices": ft_epoch_choices,
            "replay_weight_choices": replay_weight_choices,
            "stage2_repeats": int(args.stage2_repeats),
            "ft_batch_mode": str(args.ft_batch_mode),
            "ft_selection_mode": str(args.ft_selection_mode),
            "ft_smooth_window": int(args.ft_smooth_window),
            "ft_swa_window": int(args.ft_swa_window),
            "ft_l2sp_weight": float(args.ft_l2sp_weight),
            "source_only_gate_margin": float(args.source_only_gate_margin),
            "source_only_gate_penalty": float(args.source_only_gate_penalty),
            "source_only_gate_hard": bool(args.source_only_gate_hard),
        },
    )
    print(trials_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
