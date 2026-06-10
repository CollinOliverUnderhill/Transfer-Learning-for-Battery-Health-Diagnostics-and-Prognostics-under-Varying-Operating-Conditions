#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from three_step_common import DEFAULT_CANDIDATE_CSV, DEFAULT_DATA_CSV, DEFAULT_GROUP_CSV, DEFAULT_PYTHON, DEFAULT_SPLIT_CSV, ensure_child_env


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Legacy Stage 1 wrapper preserving the original narrow week5 search.")
    ap.add_argument("--python_exe", type=str, default=DEFAULT_PYTHON)
    ap.add_argument("--script", type=str, default=str(Path(__file__).with_name("stage1_source_search_optuna.py")))
    ap.add_argument("--data_csv", type=str, default=str(DEFAULT_DATA_CSV))
    ap.add_argument("--split_csv", type=str, default=str(DEFAULT_SPLIT_CSV))
    ap.add_argument("--group_cond_csv", type=str, default=str(DEFAULT_GROUP_CSV))
    ap.add_argument("--candidate_csv", type=str, default=str(DEFAULT_CANDIDATE_CSV))
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--study_name", type=str, default="stage1_source_search_w5_legacy")
    ap.add_argument("--n_trials", type=int, default=80)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_feature_candidates", type=int, default=20)
    ap.add_argument("--width_candidates", type=str, default="8,16,32,64")
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--y_col", type=str, default="lifetime_weeks_EOL70")
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
    cmd = [
        str(args.python_exe),
        str(args.script),
        "--data_csv",
        str(args.data_csv),
        "--split_csv",
        str(args.split_csv),
        "--group_cond_csv",
        str(args.group_cond_csv),
        "--candidate_csv",
        str(args.candidate_csv),
        "--out_dir",
        str(args.out_dir),
        "--study_name",
        str(args.study_name),
        "--n_trials",
        str(args.n_trials),
        "--top_k",
        str(args.top_k),
        "--max_feature_candidates",
        str(args.max_feature_candidates),
        "--width_candidates",
        str(args.width_candidates),
        "--max_depth",
        str(args.max_depth),
        "--y_col",
        str(args.y_col),
        "--dropout_choices",
        "0.0",
        "--activation_choices",
        "relu",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--val_cell_frac",
        str(args.val_cell_frac),
        "--early_stop_patience",
        str(args.early_stop_patience),
        "--min_epochs_before_early_stop",
        str(args.min_epochs_before_early_stop),
        "--lr_min",
        str(args.lr_min),
        "--lr_max",
        str(args.lr_max),
        "--weight_decay_min",
        str(args.weight_decay_min),
        "--weight_decay_max",
        str(args.weight_decay_max),
        "--seed",
        str(args.seed),
    ]
    subprocess.run(cmd, check=True, env=ensure_child_env())


if __name__ == "__main__":
    main()
