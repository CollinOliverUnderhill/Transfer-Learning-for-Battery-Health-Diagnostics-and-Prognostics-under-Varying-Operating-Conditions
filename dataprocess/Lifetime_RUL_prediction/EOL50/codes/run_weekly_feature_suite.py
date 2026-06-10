#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(r"E:\Datasets\IVAS\Lifetime_RUL_prediction\EOL50")
RUNNER = ROOT / "codes" / "run_lifetime_transfer_mlp.py"
CONFIG_JSON = ROOT / "codes" / "fixed_transfer_config.json"
DATA_CSV = Path(r"E:\Datasets\IVAS\Processing_Data\Lifetime_prediction\ivas_lifetime_10features_multiweek_per_cell.csv")
GROUP_COND_CSV = Path(r"E:\Datasets\IVAS\Groupcondi.csv")
DOMAIN_SPLIT_DIR = ROOT / "domain_split"
RESULTS_DIR = ROOT / "results"
WEEKS = (3, 5, 10, 15)


def load_config() -> Dict[str, object]:
    with CONFIG_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_one(week: int, cfg: Dict[str, object]) -> Path:
    out_root = RESULTS_DIR / f"{cfg['experiment_name']}_w{week}"
    split_csv = DOMAIN_SPLIT_DIR / f"cell_split_targetspread_w{week}_EOL50.csv"
    features = ",".join(f"{alias}_w{week}" for alias in cfg["feature_aliases"])
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
        str(cfg["y_col"]),
        "--hidden_dims",
        ",".join(str(v) for v in cfg["hidden_dims"]),
        "--activation",
        str(cfg["activation"]),
        "--epochs",
        str(cfg["epochs"]),
        "--ft_epochs",
        str(cfg["ft_epochs"]),
        "--ft_freeze_hidden_layers",
        str(cfg["ft_freeze_hidden_layers"]),
        "--batch_size",
        str(cfg["batch_size"]),
        "--lr",
        str(cfg["lr"]),
        "--ft_lr",
        str(cfg["ft_lr"]),
        "--weight_decay",
        str(cfg["weight_decay"]),
        "--ft_weight_decay",
        str(cfg["ft_weight_decay"]),
        "--val_cell_frac",
        str(cfg["val_cell_frac"]),
        "--early_stop_patience",
        str(cfg["early_stop_patience"]),
        "--min_epochs_before_early_stop",
        str(cfg["min_epochs_before_early_stop"]),
        "--ft_min_epochs_before_early_stop",
        str(cfg["ft_min_epochs_before_early_stop"]),
        "--dropout",
        str(cfg["dropout"]),
        "--tail_q",
        str(cfg["tail_q"]),
        "--log_every",
        "10",
        "--seed",
        str(cfg["seed"]),
        "--device",
        str(cfg["device"]),
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    return out_root


def read_one(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return next(csv.DictReader(f))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    rows: List[Dict[str, object]] = []

    for week in WEEKS:
        out_root = run_one(week, cfg)
        final_test = read_one(out_root / "transfer_model" / "test_overall_metrics.csv")
        source_only = read_one(out_root / "transfer_model" / "test_overall_metrics_source_only.csv")
        ft_val = read_one(out_root / "transfer_model" / "target_finetune_val_overall_metrics.csv")
        ft_combined = read_one(out_root / "transfer_model" / "target_finetune_overall_metrics.csv")
        rows.append(
            {
                "eol": "EOL50",
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

    out_csv = RESULTS_DIR / "week_feature_transfer_comparison_repartitioned_EOL50.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
