#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


IVAS_ROOT = Path(r"E:\Datasets\IVAS")
GROUP_COND_CSV = IVAS_ROOT / "Groupcondi.csv"
LEGACY_WEEKLY_DATA_CSV = IVAS_ROOT / "Processing_Data" / "Lifetime_prediction" / "ivas_lifetime_10features_per_cell.csv"
MULTIWEEK_DATA_CSV = IVAS_ROOT / "Processing_Data" / "Lifetime_prediction" / "ivas_lifetime_10features_multiweek_per_cell.csv"
MULTICYCLE_DATA_CSV = IVAS_ROOT / "Cycle_version_baoshou" / "data" / "ivas_lifetime_10features_multicycle_per_cell.csv"

LEGACY_WEEKLY_RUNNER = IVAS_ROOT / "Codes" / "chunqiu_codes" / "Lifetimepre" / "run_lifetime_transfer_mlp.py"
CYCLE_RUNNER = IVAS_ROOT / "Cycle_version_baoshou" / "codes" / "run_lifetime_transfer_mlp_cycle.py"

ARG_DEFAULTS = {
    "dropout": 0.0,
    "activation": "relu",
    "epochs": 250,
    "ft_epochs": 60,
    "ft_freeze_hidden_layers": 3,
    "batch_size": 64,
    "lr": 1e-3,
    "ft_lr": 5e-5,
    "weight_decay": 1e-4,
    "ft_weight_decay": 1e-4,
    "val_cell_frac": 0.2,
    "early_stop_patience": 25,
    "min_epochs_before_early_stop": 0,
    "ft_min_epochs_before_early_stop": 0,
    "tail_q": 0.95,
}

FEATURE_WEEK_RE = re.compile(r"_w(\d+)$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Refresh all IVAS benchmark outputs with train+fine-tune cells merged into the benchmark train split.")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on the number of benchmark roots to refresh.")
    ap.add_argument("--filter", type=str, default="", help="Optional substring filter applied to benchmark root paths.")
    ap.add_argument("--dry_run", action="store_true", help="Print the commands without executing them.")
    return ap.parse_args()


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_single_row_csv(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, got {len(rows)}")
    return rows[0]


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def parse_report_summary(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def classify_root(root: Path) -> Optional[str]:
    path_text = str(root)
    if "Cycle_version_baoshou" in path_text:
        return "cycle"
    if "Lifetime_RUL_prediction" in path_text:
        return "weekly_rul"
    if "Lifetime_Prediction" in path_text:
        return "weekly_legacy"
    return None


def find_benchmark_roots(path_filter: str) -> List[Path]:
    roots: List[Path] = []
    for benchmark_dir in IVAS_ROOT.rglob("benchmark"):
        root = benchmark_dir.parent
        family = classify_root(root)
        if family is None:
            continue
        if path_filter and path_filter not in str(root):
            continue
        if not (benchmark_dir / "config.json").exists():
            continue
        if not (root / "report_summary.txt").exists():
            continue
        roots.append(root)
    return sorted(set(roots), key=lambda p: str(p))


def find_seed_from_ancestors(root: Path) -> Optional[int]:
    for current in [root, *root.parents]:
        sweep_spec = current / "sweep_spec.json"
        if sweep_spec.exists():
            data = read_json(sweep_spec)
            if "seed" in data:
                return int(data["seed"])
    return None


def infer_seed(root: Path, family: str) -> int:
    seed = find_seed_from_ancestors(root)
    if seed is not None:
        return int(seed)
    if family in {"weekly_rul", "cycle"}:
        return 2
    return 42


def infer_device(root: Path) -> str:
    model_json = root / "benchmark" / "model.json"
    if not model_json.exists():
        return "auto"
    device = str(read_json(model_json).get("device", "auto")).strip().lower()
    if device == "cpu":
        return "cpu"
    return "auto"


def infer_runner(root: Path, family: str) -> Path:
    if family == "cycle":
        return CYCLE_RUNNER
    if family == "weekly_rul":
        return root.parents[1] / "codes" / "run_lifetime_transfer_mlp.py"
    if family == "weekly_legacy":
        return LEGACY_WEEKLY_RUNNER
    raise ValueError(f"Unsupported family: {family}")


def infer_feature_week(benchmark_cfg: Dict[str, object]) -> int:
    if "feature_week" in benchmark_cfg:
        return int(benchmark_cfg["feature_week"])

    weeks = set()
    for alias in benchmark_cfg.get("feature_aliases", []):
        match = FEATURE_WEEK_RE.search(str(alias).strip().lower())
        if match is not None:
            weeks.add(int(match.group(1)))
    if len(weeks) > 1:
        raise ValueError(f"Mixed feature weeks in benchmark config: {sorted(weeks)}")
    return next(iter(weeks), 3)


def infer_data_csv(family: str, benchmark_cfg: Dict[str, object]) -> Path:
    if family == "cycle":
        return MULTICYCLE_DATA_CSV
    if family == "weekly_rul":
        return MULTIWEEK_DATA_CSV
    if family == "weekly_legacy":
        if infer_feature_week(benchmark_cfg) != 3:
            return MULTIWEEK_DATA_CSV
        feature_aliases = [str(v).strip().lower() for v in benchmark_cfg.get("feature_aliases", [])]
        if any(FEATURE_WEEK_RE.search(alias) is not None for alias in feature_aliases):
            return MULTIWEEK_DATA_CSV
        return LEGACY_WEEKLY_DATA_CSV
    raise ValueError(f"Unsupported family: {family}")


def normalize_existing_path(path_text: str, family: str) -> str:
    candidate = Path(path_text)
    if candidate.exists():
        return str(candidate)
    if family == "cycle":
        remapped = Path(str(candidate).replace("\\Cycle_version\\", "\\Cycle_version_baoshou\\"))
        if remapped.exists():
            return str(remapped)
    return str(candidate)


def build_refresh_command(root: Path) -> List[str]:
    family = classify_root(root)
    if family is None:
        raise ValueError(f"Cannot classify benchmark root: {root}")

    benchmark_cfg = read_json(root / "benchmark" / "config.json")
    report_summary = parse_report_summary(root / "report_summary.txt")
    split_csv = normalize_existing_path(report_summary.get("split_csv", ""), family)
    if not split_csv:
        raise ValueError(f"split_csv not found in report summary: {root / 'report_summary.txt'}")

    runner = infer_runner(root, family)
    data_csv = infer_data_csv(family, benchmark_cfg)
    seed = infer_seed(root, family)
    device = infer_device(root)

    hidden_dims = ",".join(str(v) for v in benchmark_cfg.get("hidden_dims", [128, 128, 128, 128]))
    feature_aliases = ",".join(str(v) for v in benchmark_cfg.get("feature_aliases", ["f1", "f3", "f5"]))
    return [
        sys.executable,
        str(runner),
        "--out_root",
        str(root),
        "--data_csv",
        str(data_csv),
        "--split_csv",
        split_csv,
        "--group_cond_csv",
        str(GROUP_COND_CSV),
        "--features",
        feature_aliases,
        "--y_col",
        str(benchmark_cfg["y_col"]),
        "--hidden_dims",
        hidden_dims,
        "--dropout",
        str(benchmark_cfg.get("dropout", ARG_DEFAULTS["dropout"])),
        "--activation",
        str(benchmark_cfg.get("activation", ARG_DEFAULTS["activation"])),
        "--epochs",
        str(benchmark_cfg.get("epochs", ARG_DEFAULTS["epochs"])),
        "--ft_epochs",
        str(benchmark_cfg.get("ft_epochs", ARG_DEFAULTS["ft_epochs"])),
        "--ft_freeze_hidden_layers",
        str(benchmark_cfg.get("ft_freeze_hidden_layers", ARG_DEFAULTS["ft_freeze_hidden_layers"])),
        "--batch_size",
        str(benchmark_cfg.get("batch_size", ARG_DEFAULTS["batch_size"])),
        "--lr",
        str(benchmark_cfg.get("lr", ARG_DEFAULTS["lr"])),
        "--ft_lr",
        str(benchmark_cfg.get("ft_lr", ARG_DEFAULTS["ft_lr"])),
        "--weight_decay",
        str(benchmark_cfg.get("weight_decay", ARG_DEFAULTS["weight_decay"])),
        "--ft_weight_decay",
        str(benchmark_cfg.get("ft_weight_decay", ARG_DEFAULTS["ft_weight_decay"])),
        "--val_cell_frac",
        str(benchmark_cfg.get("val_cell_frac", ARG_DEFAULTS["val_cell_frac"])),
        "--early_stop_patience",
        str(benchmark_cfg.get("early_stop_patience", ARG_DEFAULTS["early_stop_patience"])),
        "--min_epochs_before_early_stop",
        str(benchmark_cfg.get("min_epochs_before_early_stop", ARG_DEFAULTS["min_epochs_before_early_stop"])),
        "--ft_min_epochs_before_early_stop",
        str(benchmark_cfg.get("ft_min_epochs_before_early_stop", ARG_DEFAULTS["ft_min_epochs_before_early_stop"])),
        "--tail_q",
        str(benchmark_cfg.get("tail_q", ARG_DEFAULTS["tail_q"])),
        "--seed",
        str(seed),
        "--device",
        device,
        "--run_parts",
        "benchmark",
    ]


def refresh_benchmark_root(root: Path, dry_run: bool) -> None:
    cmd = build_refresh_command(root)
    print(f"[RUN ] {root}")
    if dry_run:
        print(" ".join(cmd))
        return

    log_path = root / "run_stdout.log"
    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
            env=env,
        )


def collect_trial_value_map(trial_dir: Path) -> Dict[str, object]:
    values: Dict[str, object] = {}

    benchmark_dir = trial_dir / "benchmark"
    if benchmark_dir.exists():
        benchmark_cfg = read_json(benchmark_dir / "config.json")
        source_inner = read_single_row_csv(benchmark_dir / "source_inner_train_overall_metrics.csv")
        source_val = read_single_row_csv(benchmark_dir / "source_val_overall_metrics.csv")
        values.update(
            {
                "source_best_epoch": benchmark_cfg.get("source_best_epoch", ""),
                "inner_mae": source_inner.get("mae_mean", ""),
                "inner_rmse": source_inner.get("rmse", ""),
                "inner_r2": source_inner.get("r2", ""),
                "val_mae": source_val.get("mae_mean", ""),
                "val_rmse": source_val.get("rmse", ""),
                "val_r2": source_val.get("r2", ""),
                "val_mape": source_val.get("mape_percent_mean", ""),
                "source_inner_mae": source_inner.get("mae_mean", ""),
                "source_inner_rmse": source_inner.get("rmse", ""),
                "source_inner_r2": source_inner.get("r2", ""),
                "source_val_mae": source_val.get("mae_mean", ""),
                "source_val_rmse": source_val.get("rmse", ""),
                "source_val_r2": source_val.get("r2", ""),
                "source_val_mape": source_val.get("mape_percent_mean", ""),
            }
        )

    transfer_dir = trial_dir / "transfer_model"
    if transfer_dir.exists():
        transfer_cfg = read_json(transfer_dir / "config.json")
        test_final = read_single_row_csv(transfer_dir / "test_overall_metrics.csv")
        values.update(
            {
                "ft_best_epoch": transfer_cfg.get("active_best_epoch", ""),
                "active_best_epoch": transfer_cfg.get("active_best_epoch", ""),
                "test_mae": test_final.get("mae_mean", ""),
                "test_rmse": test_final.get("rmse", ""),
                "test_r2": test_final.get("r2", ""),
            }
        )

        for split_name, prefix in (
            ("target_finetune_inner_train_overall_metrics.csv", "ft_inner"),
            ("target_finetune_val_overall_metrics.csv", "ft_val"),
            ("target_finetune_overall_metrics.csv", "ft_combined"),
        ):
            split_path = transfer_dir / split_name
            if split_path.exists():
                split_row = read_single_row_csv(split_path)
                values[f"{prefix}_mae"] = split_row.get("mae_mean", "")
                values[f"{prefix}_rmse"] = split_row.get("rmse", "")
                values[f"{prefix}_r2"] = split_row.get("r2", "")

    return values


def refresh_summary_file(summary_path: Path) -> None:
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []
    if not rows or "trial" not in fieldnames:
        return

    updated_rows: List[Dict[str, object]] = []
    for row in rows:
        trial_name = str(row.get("trial", "")).strip()
        trial_dir = summary_path.parent / trial_name
        if not trial_name or not trial_dir.exists():
            updated_rows.append(row)
            continue
        value_map = collect_trial_value_map(trial_dir)
        for key, value in value_map.items():
            if key in row:
                row[key] = value
        updated_rows.append(row)

    write_csv(summary_path, updated_rows, fieldnames)
    print(f"[SYNC] {summary_path}")


def refresh_lifetime_prediction_summaries() -> None:
    eol60_root = IVAS_ROOT / "Lifetime_Prediction" / "EOL60"
    if not eol60_root.exists():
        return
    for pattern in ("sweep_summary.csv", "beam_summary.csv"):
        for summary_path in sorted(eol60_root.rglob(pattern), key=lambda p: str(p)):
            refresh_summary_file(summary_path)


def main() -> None:
    args = parse_args()
    roots = find_benchmark_roots(args.filter)
    if args.limit is not None:
        roots = roots[: int(args.limit)]

    print(f"[INFO] benchmark roots found: {len(roots)}")
    failures: List[str] = []
    for index, root in enumerate(roots, start=1):
        print(f"[INFO] [{index}/{len(roots)}] {root}")
        try:
            refresh_benchmark_root(root, dry_run=bool(args.dry_run))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{root}: {exc}")
            print(f"[FAIL] {root}: {exc}")

    if not args.dry_run:
        refresh_lifetime_prediction_summaries()

    if failures:
        failure_path = IVAS_ROOT / "benchmark_refresh_failures.txt"
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise SystemExit(f"Benchmark refresh completed with failures. See: {failure_path}")

    print("[INFO] Benchmark refresh completed.")


if __name__ == "__main__":
    main()
