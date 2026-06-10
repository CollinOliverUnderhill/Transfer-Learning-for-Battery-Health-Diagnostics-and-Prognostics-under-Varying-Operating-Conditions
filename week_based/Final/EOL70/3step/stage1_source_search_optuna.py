#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import optuna
import pandas as pd

from three_step_common import (
    DEFAULT_CANDIDATE_CSV,
    DEFAULT_DATA_CSV,
    DEFAULT_GROUP_CSV,
    DEFAULT_PYTHON,
    DEFAULT_RUNNER,
    DEFAULT_SPLIT_CSV,
    architecture_candidates,
    ensure_child_env,
    load_feature_candidates,
    save_json,
    sort_frame_for_stage,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage 1 source-domain search with Optuna TPE.")
    ap.add_argument("--python_exe", type=str, default=DEFAULT_PYTHON)
    ap.add_argument("--runner", type=str, default=str(DEFAULT_RUNNER))
    ap.add_argument("--data_csv", type=str, default=str(DEFAULT_DATA_CSV))
    ap.add_argument("--split_csv", type=str, default=str(DEFAULT_SPLIT_CSV))
    ap.add_argument("--group_cond_csv", type=str, default=str(DEFAULT_GROUP_CSV))
    ap.add_argument("--candidate_csv", type=str, default=str(DEFAULT_CANDIDATE_CSV))
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--study_name", type=str, default="stage1_source_search_w5")
    ap.add_argument("--n_trials", type=int, default=160)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--max_feature_candidates", type=int, default=120)
    ap.add_argument("--width_candidates", type=str, default="8,16,32,64,96,128")
    ap.add_argument("--max_depth", type=int, default=5)
    ap.add_argument("--y_col", type=str, default="lifetime_weeks_EOL70")
    ap.add_argument("--dropout_choices", type=str, default="0.0,0.05,0.1,0.2")
    ap.add_argument("--activation_choices", type=str, default="relu,gelu")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_cell_frac", type=float, default=0.2)
    ap.add_argument("--early_stop_patience", type=int, default=25)
    ap.add_argument("--min_epochs_before_early_stop", type=int, default=0)
    ap.add_argument("--lr_min", type=float, default=1e-4)
    ap.add_argument("--lr_max", type=float, default=3e-3)
    ap.add_argument("--weight_decay_min", type=float, default=1e-7)
    ap.add_argument("--weight_decay_max", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=2)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "stage1_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    study_db = out_dir / "stage1_optuna.db"

    feature_candidates = load_feature_candidates(Path(args.candidate_csv), max_candidates=args.max_feature_candidates)
    width_candidates = [int(x.strip()) for x in str(args.width_candidates).split(",") if x.strip()]
    dropout_choices = [float(x.strip()) for x in str(args.dropout_choices).split(",") if x.strip()]
    activation_choices = [x.strip() for x in str(args.activation_choices).split(",") if x.strip()]
    arch_candidates = architecture_candidates(width_candidates, int(args.max_depth))

    sampler = optuna.samplers.TPESampler(seed=int(args.seed), multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///{study_db}",
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        feature_set = trial.suggest_categorical("features", feature_candidates)
        hidden_dims = trial.suggest_categorical("hidden_dims", arch_candidates)
        dropout = float(trial.suggest_categorical("dropout", dropout_choices))
        activation = trial.suggest_categorical("activation", activation_choices)
        lr = trial.suggest_float("lr", float(args.lr_min), float(args.lr_max), log=True)
        weight_decay = trial.suggest_float("weight_decay", float(args.weight_decay_min), float(args.weight_decay_max), log=True)

        trial_dir = runs_dir / f"trial_{trial.number:04d}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(args.python_exe),
            str(args.runner),
            "--mode",
            "source_only",
            "--data_csv",
            str(args.data_csv),
            "--split_csv",
            str(args.split_csv),
            "--group_cond_csv",
            str(args.group_cond_csv),
            "--out_root",
            str(trial_dir),
            "--features",
            feature_set,
            "--y_col",
            args.y_col,
            "--hidden_dims",
            hidden_dims,
            "--dropout",
            str(dropout),
            "--activation",
            activation,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(lr),
            "--weight_decay",
            str(weight_decay),
            "--val_cell_frac",
            str(args.val_cell_frac),
            "--early_stop_patience",
            str(args.early_stop_patience),
            "--min_epochs_before_early_stop",
            str(args.min_epochs_before_early_stop),
            "--seed",
            str(args.seed + trial.number),
            "--log_every",
            "100",
        ]
        subprocess.run(cmd, check=True, env=ensure_child_env())

        summary = json.loads((trial_dir / "trial_summary.json").read_text(encoding="utf-8"))
        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("source_checkpoint", summary["source_checkpoint"])
        trial.set_user_attr("features", summary["features"])
        trial.set_user_attr("hidden_dims", summary["hidden_dims"])
        trial.set_user_attr("dropout", summary["dropout"])
        trial.set_user_attr("activation", summary["activation"])
        trial.set_user_attr("source_val_mae", summary["source_val_mae"])
        trial.set_user_attr("source_val_rmse", summary["source_val_rmse"])
        trial.set_user_attr("source_val_mape_percent_mean", summary["source_val_mape_percent_mean"])
        trial.set_user_attr("source_val_smape_percent_mean", summary["source_val_smape_percent_mean"])
        trial.set_user_attr("source_val_wmape_percent", summary["source_val_wmape_percent"])
        trial.set_user_attr("train_val_mae_gap", summary["train_val_mae_gap"])
        return float(summary["source_val_mae"])

    study.optimize(objective, n_trials=int(args.n_trials))

    rows: List[Dict[str, object]] = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        rows.append(
            {
                "trial_number": int(trial.number),
                "objective": float(trial.value),
                "trial_dir": trial.user_attrs.get("trial_dir", ""),
                "source_checkpoint": trial.user_attrs.get("source_checkpoint", ""),
                "features": trial.user_attrs.get("features", trial.params.get("features", "")),
                "hidden_dims": trial.user_attrs.get("hidden_dims", trial.params.get("hidden_dims", "")),
                "dropout": float(trial.user_attrs["dropout"]),
                "activation": str(trial.user_attrs["activation"]),
                "lr": float(trial.params["lr"]),
                "weight_decay": float(trial.params["weight_decay"]),
                "source_val_mae": float(trial.user_attrs["source_val_mae"]),
                "source_val_rmse": float(trial.user_attrs["source_val_rmse"]),
                "source_val_mape_percent_mean": float(trial.user_attrs["source_val_mape_percent_mean"]),
                "source_val_smape_percent_mean": float(trial.user_attrs["source_val_smape_percent_mean"]),
                "source_val_wmape_percent": float(trial.user_attrs["source_val_wmape_percent"]),
                "train_val_mae_gap": float(trial.user_attrs["train_val_mae_gap"]),
            }
        )

    trials_df = sort_frame_for_stage(pd.DataFrame(rows), "stage1")
    trials_path = out_dir / "stage1_trials.csv"
    trials_df.to_csv(trials_path, index=False, encoding="utf-8-sig")

    top_df = trials_df.head(int(args.top_k)).copy()
    top_path = out_dir / "stage1_top_source_checkpoints.csv"
    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")

    save_json(
        out_dir / "stage1_manifest.json",
        {
            "study_name": args.study_name,
            "study_db": str(study_db),
            "trials_csv": str(trials_path),
            "top_source_csv": str(top_path),
            "n_trials": int(args.n_trials),
            "top_k": int(args.top_k),
            "feature_candidate_count": len(feature_candidates),
            "architecture_candidate_count": len(arch_candidates),
            "dropout_choices": dropout_choices,
            "activation_choices": activation_choices,
            "lr_min": float(args.lr_min),
            "lr_max": float(args.lr_max),
            "weight_decay_min": float(args.weight_decay_min),
            "weight_decay_max": float(args.weight_decay_max),
        },
    )
    print(trials_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
