#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
from itertools import product
from pathlib import Path
from typing import Dict, List

import pandas as pd

from three_step_common import (
    DEFAULT_DATA_CSV,
    DEFAULT_GROUP_CSV,
    DEFAULT_PYTHON,
    DEFAULT_RUNNER,
    DEFAULT_SPLIT_CSV,
    ensure_child_env,
    parse_csv_list,
    save_json,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Low-capacity Stage2 fine-tune grid. Reuses Stage1 source selection, "
            "keeps a small pre-defined fine-tune search space, and ranks by robust repeated target-val score."
        )
    )
    ap.add_argument("--python_exe", type=str, default=DEFAULT_PYTHON)
    ap.add_argument("--runner", type=str, default=str(DEFAULT_RUNNER))
    ap.add_argument("--data_csv", type=str, default=str(DEFAULT_DATA_CSV))
    ap.add_argument("--split_csv", type=str, default=str(DEFAULT_SPLIT_CSV))
    ap.add_argument("--group_cond_csv", type=str, default=str(DEFAULT_GROUP_CSV))
    ap.add_argument("--stage1_top_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--source_top_k", type=int, default=5)
    ap.add_argument("--y_col", type=str, default="lifetime_weeks_EOL70")
    ap.add_argument("--dropout", type=float, default=-1.0)
    ap.add_argument("--activation", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_cell_frac", type=float, default=0.2)
    ap.add_argument("--early_stop_patience", type=int, default=25)
    ap.add_argument("--min_target_val_cells", type=int, default=3)
    ap.add_argument("--support_subset_mode", type=str, default="quantile", choices=["quantile", "random", "high_tail"])
    ap.add_argument("--support_subset_seed", type=int, default=17)
    ap.add_argument("--min_support_cells", type=int, default=6)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--stage2_repeats", type=int, default=5)
    ap.add_argument("--ft_epochs", type=int, default=400)
    ap.add_argument("--ft_lrs", type=str, default="1e-5,3e-5,1e-4")
    ap.add_argument("--ft_weight_decays", type=str, default="1e-4")
    ap.add_argument("--replay_weights", type=str, default="0.3,1.0")
    ap.add_argument("--support_ratios", type=str, default="0.67")
    ap.add_argument("--freeze_mode", type=str, default="all_hidden", choices=["all_hidden", "all_but_last_hidden"])
    ap.add_argument("--ft_batch_mode", type=str, default="mini", choices=["mini", "full"])
    ap.add_argument("--ft_selection_mode", type=str, default="raw_best", choices=["raw_best", "smooth_best", "last_window_swa", "final"])
    ap.add_argument("--ft_smooth_window", type=int, default=25)
    ap.add_argument("--ft_swa_window", type=int, default=50)
    ap.add_argument("--ft_l2sp_weight", type=float, default=0.0)
    ap.add_argument("--robust_iqr_weight", type=float, default=0.5)
    ap.add_argument("--source_only_gate_margin", type=float, default=0.0)
    ap.add_argument("--source_only_gate_penalty", type=float, default=10.0)
    ap.add_argument("--source_only_gate_hard", action="store_true")
    return ap


def resolve_source_checkpoint(path_text: str, stage1_top_csv: Path) -> str:
    raw_path = Path(str(path_text))
    if raw_path.exists():
        return str(raw_path)
    parts = raw_path.parts
    if "stage1_runs" in parts:
        idx = parts.index("stage1_runs")
        trial_rel = Path(*parts[idx:])
        local_candidate = stage1_top_csv.parent / trial_rel
        if local_candidate.exists():
            return str(local_candidate)
    local_candidate = stage1_top_csv.parent / "stage1_runs" / raw_path.parent.name / raw_path.name
    if local_candidate.exists():
        return str(local_candidate)
    return str(raw_path)


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(float(v) for v in values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * float(q)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def iqr(values: List[float]) -> float:
    return percentile(values, 0.75) - percentile(values, 0.25)


def median(values: List[float]) -> float:
    return float(statistics.median(float(v) for v in values))


def stdev(values: List[float]) -> float:
    return float(statistics.stdev(float(v) for v in values)) if len(values) > 1 else 0.0


def freeze_layers_for(hidden_dims: str, freeze_mode: str) -> int:
    n_hidden = len([v for v in str(hidden_dims).split(",") if str(v).strip()])
    if str(freeze_mode) == "all_but_last_hidden":
        return max(0, n_hidden - 1)
    return max(0, n_hidden)


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "stage2_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    stage1_top_csv = Path(args.stage1_top_csv)
    source_df = pd.read_csv(stage1_top_csv).reset_index(drop=True)
    if len(source_df) == 0:
        raise ValueError(f"No source checkpoints found in {stage1_top_csv}")
    source_df["source_checkpoint"] = [
        resolve_source_checkpoint(value, stage1_top_csv) for value in source_df["source_checkpoint"].astype(str)
    ]
    source_df = source_df.head(max(1, int(args.source_top_k))).reset_index(drop=True)

    ft_lrs = parse_csv_list(args.ft_lrs, float)
    ft_weight_decays = parse_csv_list(args.ft_weight_decays, float)
    replay_weights = parse_csv_list(args.replay_weights, float)
    support_ratios = parse_csv_list(args.support_ratios, float)

    rows: List[Dict[str, object]] = []
    trial_number = 0
    for source_idx, source_row in source_df.iterrows():
        source_dropout = (
            float(source_row["dropout"])
            if "dropout" in source_row and pd.notna(source_row["dropout"])
            else float(args.dropout if float(args.dropout) >= 0.0 else 0.0)
        )
        source_activation = (
            str(source_row["activation"])
            if "activation" in source_row and str(source_row["activation"]).strip()
            else (str(args.activation).strip() or "relu")
        )
        freeze_layers = freeze_layers_for(str(source_row["hidden_dims"]), str(args.freeze_mode))

        for ft_lr, ft_weight_decay, replay_weight, support_ratio in product(
            ft_lrs,
            ft_weight_decays,
            replay_weights,
            support_ratios,
        ):
            trial_dir = runs_dir / f"trial_{trial_number:04d}"
            if trial_dir.exists():
                shutil.rmtree(trial_dir)
            trial_dir.mkdir(parents=True, exist_ok=True)

            repeat_summaries: List[Dict[str, object]] = []
            repeat_dirs: List[str] = []
            n_repeats = max(1, int(args.stage2_repeats))
            for repeat_idx in range(n_repeats):
                repeat_dir = trial_dir / f"repeat_{repeat_idx:02d}"
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
                    str(args.ft_epochs),
                    "--min_target_val_cells",
                    str(args.min_target_val_cells),
                    "--support_subset_mode",
                    str(args.support_subset_mode),
                    "--support_subset_seed",
                    str(args.support_subset_seed),
                    "--min_support_cells",
                    str(args.min_support_cells),
                    "--seed",
                    str(int(args.seed) + trial_number * 100 + repeat_idx),
                    "--source_checkpoint",
                    str(source_row["source_checkpoint"]),
                    "--ft_lr",
                    str(ft_lr),
                    "--ft_weight_decay",
                    str(ft_weight_decay),
                    "--ft_epochs",
                    str(args.ft_epochs),
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
                        raise RuntimeError(invalid.get("reason", "invalid low-capacity trial")) from exc
                    raise
                repeat_summaries.append(json.loads((repeat_dir / "trial_summary.json").read_text(encoding="utf-8")))

            ft_val_maes = [float(item["target_ft_val_mae"]) for item in repeat_summaries]
            ft_val_rmses = [float(item["target_ft_val_rmse"]) for item in repeat_summaries]
            ft_val_mapes = [float(item["target_ft_val_mape_percent_mean"]) for item in repeat_summaries]
            source_only_maes = [float(item["target_ft_val_source_only_mae"]) for item in repeat_summaries]
            source_only_mapes = [float(item["target_ft_val_source_only_mape_percent_mean"]) for item in repeat_summaries]
            improve_maes = [float(item["target_ft_val_vs_source_only_mae_improve_percent"]) for item in repeat_summaries]
            improve_mapes = [float(item["target_ft_val_vs_source_only_mape_improve_percent"]) for item in repeat_summaries]

            ft_val_mae_median = median(ft_val_maes)
            source_only_mae_median = median(source_only_maes)
            required_mae = source_only_mae_median * (1.0 - float(args.source_only_gate_margin))
            gate_deficit_mae = max(0.0, ft_val_mae_median - required_mae)
            robust_iqr = iqr(ft_val_maes)
            objective = (
                ft_val_mae_median
                + float(args.robust_iqr_weight) * robust_iqr
                + float(args.source_only_gate_penalty) * gate_deficit_mae
            )
            if bool(args.source_only_gate_hard) and gate_deficit_mae > 0.0:
                objective += 1_000_000.0 + gate_deficit_mae

            row = {
                "trial_number": int(trial_number),
                "objective": float(objective),
                "trial_dir": str(trial_dir),
                "repeat_trial_dirs": json.dumps(repeat_dirs, ensure_ascii=False),
                "source_rank_idx": int(source_idx),
                "source_checkpoint": str(source_row["source_checkpoint"]),
                "features": str(source_row["features"]),
                "hidden_dims": str(source_row["hidden_dims"]),
                "dropout": float(source_dropout),
                "activation": str(source_activation),
                "source_stage1_val_mae": float(source_row["source_val_mae"]),
                "ft_lr": float(ft_lr),
                "ft_weight_decay": float(ft_weight_decay),
                "ft_epochs": int(args.ft_epochs),
                "ft_min_epochs_before_early_stop": int(args.ft_epochs),
                "ft_freeze_hidden_layers_raw": int(freeze_layers),
                "ft_freeze_hidden_layers": int(freeze_layers),
                "stage2_repeats": int(n_repeats),
                "ft_batch_mode": str(args.ft_batch_mode),
                "ft_selection_mode": str(args.ft_selection_mode),
                "ft_smooth_window": int(args.ft_smooth_window),
                "ft_swa_window": int(args.ft_swa_window),
                "ft_l2sp_weight": float(args.ft_l2sp_weight),
                "target_support_ratio": float(support_ratio),
                "transfer_replay_weight": float(replay_weight),
                "stage2_objective_score": float(objective),
                "target_ft_val_mae": float(ft_val_mae_median),
                "target_ft_val_mae_mean": float(statistics.mean(ft_val_maes)),
                "target_ft_val_mae_std": float(stdev(ft_val_maes)),
                "target_ft_val_mae_iqr": float(robust_iqr),
                "target_ft_val_rmse": float(median(ft_val_rmses)),
                "target_ft_val_mape_percent_mean": float(median(ft_val_mapes)),
                "target_ft_val_smape_percent_mean": float(median([float(item["target_ft_val_smape_percent_mean"]) for item in repeat_summaries])),
                "target_ft_val_wmape_percent": float(median([float(item["target_ft_val_wmape_percent"]) for item in repeat_summaries])),
                "target_ft_val_cell_count": int(statistics.median(int(item["target_ft_val_cell_count"]) for item in repeat_summaries)),
                "target_ft_val_source_only_mae": float(source_only_mae_median),
                "target_ft_val_source_only_mae_mean": float(statistics.mean(source_only_maes)),
                "target_ft_val_source_only_mape_percent_mean": float(median(source_only_mapes)),
                "target_ft_val_vs_source_only_mae_improve_percent": float(median(improve_maes)),
                "target_ft_val_vs_source_only_mape_improve_percent": float(median(improve_mapes)),
                "source_only_gate_required_mae": float(required_mae),
                "source_only_gate_deficit_mae": float(gate_deficit_mae),
                "source_only_gate_pass": bool(gate_deficit_mae <= 0.0),
            }
            rows.append(row)
            save_json(trial_dir / "trial_summary.json", {**row, "repeat_summaries": repeat_summaries})
            trial_number += 1

    trials_df = pd.DataFrame(rows).sort_values(
        [
            "objective",
            "target_ft_val_mae",
            "target_ft_val_mae_iqr",
            "target_ft_val_vs_source_only_mae_improve_percent",
        ],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    trials_path = out_dir / "stage2_trials.csv"
    trials_df.to_csv(trials_path, index=False, encoding="utf-8-sig")

    top_df = trials_df.head(max(1, int(args.top_k))).copy()
    top_path = out_dir / "stage2_best_configs.csv"
    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")

    save_json(
        out_dir / "stage2_manifest.json",
        {
            "mode": "lowcapacity_grid",
            "trials_csv": str(trials_path),
            "best_configs_csv": str(top_path),
            "n_trials": int(len(trials_df)),
            "source_top_k": int(args.source_top_k),
            "top_k": int(args.top_k),
            "stage2_repeats": int(args.stage2_repeats),
            "ft_epochs": int(args.ft_epochs),
            "ft_lrs": ft_lrs,
            "ft_weight_decays": ft_weight_decays,
            "replay_weights": replay_weights,
            "support_ratios": support_ratios,
            "freeze_mode": str(args.freeze_mode),
            "robust_iqr_weight": float(args.robust_iqr_weight),
            "source_only_gate_margin": float(args.source_only_gate_margin),
            "source_only_gate_penalty": float(args.source_only_gate_penalty),
            "source_only_gate_hard": bool(args.source_only_gate_hard),
        },
    )
    print(trials_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
