#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(r"E:\Datasets\IVAS\Lifetime_RUL_prediction\EOL60")
RUNNER = Path(r"E:\Datasets\IVAS\Codes\chunqiu_codes\Lifetimepre\run_lifetime_transfer_mlp.py")
DATA_CSV = Path(r"E:\Datasets\IVAS\Processing_Data\Lifetime_prediction\ivas_lifetime_10features_multiweek_per_cell.csv")
GROUP_COND_CSV = Path(r"E:\Datasets\IVAS\Groupcondi.csv")
DOMAIN_SPLIT_DIR = ROOT / "domain_split"
RESULTS_DIR = ROOT / "results"
WEEKS = (3, 5, 10, 15)


def run_one(week: int) -> Path:
    out_root = RESULTS_DIR / f"fixed_8x8x8_freeze2_w{week}"
    split_csv = DOMAIN_SPLIT_DIR / f"cell_split_targetspread_w{week}_EOL60.csv"
    features = f"f1_w{week},f3_w{week},f5_w{week},f8_w{week},f2_w{week}"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--data_csv",
        str(DATA_CSV),
        "--split_csv",
        str(split_csv),
        "--group_cond_csv",
        str(GROUP_COND_CSV),
        "--out_root",
        str(out_root),
        "--features",
        features,
        "--y_col",
        "lifetime_weeks_EOL60",
        "--hidden_dims",
        "8,8,8",
        "--activation",
        "relu",
        "--epochs",
        "800",
        "--ft_epochs",
        "800",
        "--ft_freeze_hidden_layers",
        "2",
        "--batch_size",
        "16",
        "--lr",
        "7e-4",
        "--ft_lr",
        "5e-5",
        "--weight_decay",
        "1e-4",
        "--ft_weight_decay",
        "1e-4",
        "--val_cell_frac",
        "0.2",
        "--early_stop_patience",
        "25",
        "--min_epochs_before_early_stop",
        "0",
        "--ft_min_epochs_before_early_stop",
        "0",
        "--dropout",
        "0.0",
        "--tail_q",
        "0.95",
        "--log_every",
        "10",
        "--seed",
        "2",
        "--device",
        "auto",
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    return out_root


def read_one(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return next(csv.DictReader(f))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for week in WEEKS:
        out_root = run_one(week)
        final_test = read_one(out_root / "transfer_model" / "test_overall_metrics.csv")
        source_only = read_one(out_root / "transfer_model" / "test_overall_metrics_source_only.csv")
        ft_val = read_one(out_root / "transfer_model" / "target_finetune_val_overall_metrics.csv")
        ft_combined = read_one(out_root / "transfer_model" / "target_finetune_overall_metrics.csv")
        rows.append(
            {
                "week": f"w{week}",
                "out_root": str(out_root),
                "test_final_mae": float(final_test["mae_mean"]),
                "test_final_rmse": float(final_test["rmse"]),
                "test_final_r2": float(final_test["r2"]),
                "test_source_only_mae": float(source_only["mae_mean"]),
                "test_source_only_rmse": float(source_only["rmse"]),
                "test_source_only_r2": float(source_only["r2"]),
                "ft_val_mae": float(ft_val["mae_mean"]),
                "ft_val_rmse": float(ft_val["rmse"]),
                "ft_val_r2": float(ft_val["r2"]),
                "ft_combined_mae": float(ft_combined["mae_mean"]),
                "ft_combined_rmse": float(ft_combined["rmse"]),
                "ft_combined_r2": float(ft_combined["r2"]),
            }
        )

    out_csv = RESULTS_DIR / "week_feature_transfer_comparison_repartitioned_EOL60.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
