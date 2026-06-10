#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from three_step_common import DEFAULT_DATA_CSV, DEFAULT_GROUP_CSV, DEFAULT_PYTHON, DEFAULT_SPLIT_CSV, WORKSPACE_ROOT, ensure_child_env


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Legacy Stage 3 wrapper preserving the earlier final-evaluation defaults.")
    ap.add_argument("--python_exe", type=str, default=DEFAULT_PYTHON)
    ap.add_argument("--script", type=str, default=str(Path(__file__).with_name("stage3_final_evaluate.py")))
    ap.add_argument("--data_csv", type=str, default=str(DEFAULT_DATA_CSV))
    ap.add_argument("--split_csv", type=str, default=str(DEFAULT_SPLIT_CSV))
    ap.add_argument("--group_cond_csv", type=str, default=str(DEFAULT_GROUP_CSV))
    ap.add_argument("--runner", type=str, default=str(WORKSPACE_ROOT / "codes" / "run_lifetime_transfer_mlp.py"))
    ap.add_argument("--stage2_best_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--y_col", type=str, default="lifetime_weeks_EOL70")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--activation", type=str, default="relu")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_cell_frac", type=float, default=0.2)
    ap.add_argument("--early_stop_patience", type=int, default=25)
    ap.add_argument("--min_source_epochs", type=int, default=0)
    ap.add_argument("--min_ft_epochs", type=int, default=0)
    ap.add_argument("--min_epochs_before_early_stop", type=int, default=0)
    ap.add_argument("--ft_min_epochs_before_early_stop", type=int, default=0)
    ap.add_argument("--support_subset_mode", type=str, default="quantile")
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


def main() -> None:
    args = build_arg_parser().parse_args()
    cmd = [
        str(args.python_exe),
        str(args.script),
        "--runner",
        str(args.runner),
        "--data_csv",
        str(args.data_csv),
        "--split_csv",
        str(args.split_csv),
        "--group_cond_csv",
        str(args.group_cond_csv),
        "--stage2_best_csv",
        str(args.stage2_best_csv),
        "--out_dir",
        str(args.out_dir),
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
        "--min_source_epochs",
        str(args.min_source_epochs),
        "--min_ft_epochs",
        str(args.min_ft_epochs),
        "--min_epochs_before_early_stop",
        str(args.min_epochs_before_early_stop),
        "--ft_min_epochs_before_early_stop",
        str(args.ft_min_epochs_before_early_stop),
        "--support_subset_mode",
        str(args.support_subset_mode),
        "--support_subset_seed",
        str(args.support_subset_seed),
        "--min_support_cells",
        str(args.min_support_cells),
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
        "--seed",
        str(args.seed),
    ]
    subprocess.run(cmd, check=True, env=ensure_child_env())


if __name__ == "__main__":
    main()
