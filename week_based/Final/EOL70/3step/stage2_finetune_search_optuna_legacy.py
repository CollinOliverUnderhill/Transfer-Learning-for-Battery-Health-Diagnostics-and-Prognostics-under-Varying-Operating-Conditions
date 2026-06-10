#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from three_step_common import DEFAULT_DATA_CSV, DEFAULT_GROUP_CSV, DEFAULT_PYTHON, DEFAULT_SPLIT_CSV, ensure_child_env


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Legacy Stage 2 wrapper preserving the original fine-tune search protocol.")
    ap.add_argument("--python_exe", type=str, default=DEFAULT_PYTHON)
    ap.add_argument("--script", type=str, default=str(Path(__file__).with_name("stage2_finetune_search_optuna.py")))
    ap.add_argument("--data_csv", type=str, default=str(DEFAULT_DATA_CSV))
    ap.add_argument("--split_csv", type=str, default=str(DEFAULT_SPLIT_CSV))
    ap.add_argument("--group_cond_csv", type=str, default=str(DEFAULT_GROUP_CSV))
    ap.add_argument("--stage1_top_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--study_name", type=str, default="stage2_finetune_search_w5_legacy")
    ap.add_argument("--n_trials", type=int, default=80)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--y_col", type=str, default="lifetime_weeks_EOL70")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--activation", type=str, default="relu")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_cell_frac", type=float, default=0.2)
    ap.add_argument("--early_stop_patience", type=int, default=25)
    ap.add_argument("--ft_min_epochs_before_early_stop", type=int, default=0)
    ap.add_argument("--stage2_finetune_selection", type=str, default="early_stop", choices=["fixed_budget", "early_stop"])
    ap.add_argument("--min_target_val_cells", type=int, default=3)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--support_ratios", type=str, default="0.67,1.0")
    ap.add_argument("--ft_epoch_choices", type=str, default="200,400,800")
    ap.add_argument("--replay_weight_choices", type=str, default="0.0,0.1,0.3,1.0")
    ap.add_argument("--ft_freeze_hidden_layers_min", type=int, default=2)
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
        "--stage1_top_csv",
        str(args.stage1_top_csv),
        "--out_dir",
        str(args.out_dir),
        "--study_name",
        str(args.study_name),
        "--n_trials",
        str(args.n_trials),
        "--top_k",
        str(args.top_k),
        "--y_col",
        str(args.y_col),
        "--dropout",
        str(args.dropout),
        "--activation",
        str(args.activation),
        "--batch_size",
        str(args.batch_size),
        "--val_cell_frac",
        str(args.val_cell_frac),
        "--early_stop_patience",
        str(args.early_stop_patience),
        "--ft_min_epochs_before_early_stop",
        str(args.ft_min_epochs_before_early_stop),
        "--stage2_finetune_selection",
        str(args.stage2_finetune_selection),
        "--min_target_val_cells",
        str(args.min_target_val_cells),
        "--seed",
        str(args.seed),
        "--support_ratios",
        str(args.support_ratios),
        "--ft_epoch_choices",
        str(args.ft_epoch_choices),
        "--replay_weight_choices",
        str(args.replay_weight_choices),
        "--ft_freeze_hidden_layers_min",
        str(args.ft_freeze_hidden_layers_min),
        "--ft_lr_min",
        str(args.ft_lr_min),
        "--ft_lr_max",
        str(args.ft_lr_max),
        "--ft_weight_decay_min",
        str(args.ft_weight_decay_min),
        "--ft_weight_decay_max",
        str(args.ft_weight_decay_max),
        "--stage2_repeats",
        str(args.stage2_repeats),
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
        "--source_only_gate_margin",
        str(args.source_only_gate_margin),
        "--source_only_gate_penalty",
        str(args.source_only_gate_penalty),
    ]
    if args.source_only_gate_hard:
        cmd.append("--source_only_gate_hard")
    subprocess.run(cmd, check=True, env=ensure_child_env())


if __name__ == "__main__":
    main()
