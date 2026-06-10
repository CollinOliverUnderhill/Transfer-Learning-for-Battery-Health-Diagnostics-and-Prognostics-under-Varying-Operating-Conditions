#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean, pstdev


WORK_ROOT = Path(__file__).resolve().parent
REFERENCE_SUMMARY = WORK_ROOT / "outputs" / "BasicModel" / "stage3_full_benchmark_transfer_summary_all_splits.csv"
PROTOCOL_ROOT = WORK_ROOT / "outputs" / "protocol_w6_10_from_stage3_final_rerun_400_legacy400"
RANDOM_ROOT = WORK_ROOT / "outputs" / "random_w5_EOL70_10seeds_legacy400"

COUNT_METRICS = ("n_rows", "n_cells", "n_groups")
VALUE_METRICS = ("mae", "rmse", "mape", "smape", "wmape", "r2")
ALL_METRICS = COUNT_METRICS + VALUE_METRICS

CSV_METRIC_MAP = {
    "mae": "mae",
    "rmse": "rmse",
    "mape": "mape_percent",
    "smape": "smape_percent",
    "wmape": "wmape_percent",
    "r2": "r2",
}

BENCH_SPLITS = {
    "bench_source_inner_train": ("benchmark", "source_inner_train_overall_metrics.csv"),
    "bench_source_val": ("benchmark", "source_val_overall_metrics.csv"),
    "bench_test": ("benchmark", "test_overall_metrics.csv"),
}

TRANSFER_SPLITS = {
    "transfer_source_inner_train": ("transfer_model", "source_inner_train_overall_metrics.csv"),
    "transfer_source_val": ("transfer_model", "source_val_overall_metrics.csv"),
    "transfer_target_finetune_inner_train": ("transfer_model", "target_finetune_inner_train_overall_metrics.csv"),
    "transfer_target_finetune_val": ("transfer_model", "target_finetune_val_overall_metrics.csv"),
    "transfer_target_finetune": ("transfer_model", "target_finetune_overall_metrics.csv"),
    "transfer_test": ("transfer_model", "test_overall_metrics.csv"),
}

IMPROVEMENT_SPECS = (
    ("source_inner_train", "bench_source_inner_train", "transfer_source_inner_train"),
    ("source_val", "bench_source_val", "transfer_source_val"),
    ("test", "bench_test", "transfer_test"),
)


def read_reference_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def read_single_row_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, got {len(rows)}")
    return rows[0]


def to_float(value: object) -> float:
    try:
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return float("nan")
        return float(text)
    except Exception:
        return float("nan")


def is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def format_number(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    try:
        v = float(value)
    except Exception:
        return ""
    if not is_finite(v):
        return ""
    rounded = round(v, digits)
    text = f"{rounded:.{digits}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def add_metric_block(out: dict[str, object], prefix: str, csv_path: Path) -> None:
    metrics = read_single_row_csv(csv_path)
    for metric in COUNT_METRICS:
        value = to_float(metrics.get(metric, ""))
        out[f"{prefix}_{metric}"] = int(value) if is_finite(value) else ""
    for metric in VALUE_METRICS:
        source_col = CSV_METRIC_MAP[metric]
        value = to_float(metrics.get(source_col, ""))
        out[f"{prefix}_{metric}"] = value if is_finite(value) else ""


def improve_percent(bench: float, transfer: float) -> float:
    if not is_finite(bench) or not is_finite(transfer) or bench == 0:
        return float("nan")
    return (bench - transfer) / bench * 100.0


def add_improvement_columns(out: dict[str, object]) -> None:
    for label, bench_prefix, transfer_prefix in IMPROVEMENT_SPECS:
        for metric in ("mae", "rmse", "mape"):
            out[f"{label}_transfer_vs_bench_{metric}_improve_percent"] = improve_percent(
                to_float(out.get(f"{bench_prefix}_{metric}", "")),
                to_float(out.get(f"{transfer_prefix}_{metric}", "")),
            )
        out[f"{label}_transfer_vs_bench_r2_delta"] = (
            to_float(out.get(f"{transfer_prefix}_r2", "")) - to_float(out.get(f"{bench_prefix}_r2", ""))
        )


def summarize_stage3_dir(stage3_dir: Path, display_name: str) -> dict[str, object]:
    out: dict[str, object] = {"stage3_dir": display_name}

    for prefix, (subdir, filename) in BENCH_SPLITS.items():
        path = stage3_dir / subdir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        add_metric_block(out, prefix, path)

    for prefix, (subdir, filename) in TRANSFER_SPLITS.items():
        path = stage3_dir / subdir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        add_metric_block(out, prefix, path)

    add_improvement_columns(out)
    return out


def discover_stage3_dirs(root: Path) -> list[tuple[Path, str]]:
    dirs: list[tuple[Path, str]] = []
    def natural_key(path: Path) -> tuple[str, int, str]:
        match = re.match(r"^([A-Za-z_]+)(\d+)$", path.name)
        if match:
            return match.group(1), int(match.group(2)), path.name
        return path.name, -1, path.name

    for child in sorted((p for p in root.iterdir() if p.is_dir() and p.name != "logs"), key=natural_key):
        stage3 = child / "stage3_final"
        if stage3.exists():
            dirs.append((stage3, f"{child.name}/stage3_final"))
    return dirs


def write_wide_summary(rows: list[dict[str, object]], header: list[str], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: format_number(row.get(col, "")) for col in header})


def write_numeric_aggregate(rows: list[dict[str, object]], header: list[str], out_csv: Path) -> None:
    numeric_cols = [col for col in header if col != "stage3_dir"]
    stat_fields = ["metric", "n", "mean", "std", "min", "max"]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stat_fields)
        writer.writeheader()
        for col in numeric_cols:
            values = [to_float(row.get(col, "")) for row in rows]
            values = [v for v in values if is_finite(v)]
            if not values:
                continue
            writer.writerow(
                {
                    "metric": col,
                    "n": len(values),
                    "mean": format_number(mean(values)),
                    "std": format_number(pstdev(values)) if len(values) > 1 else "0",
                    "min": format_number(min(values)),
                    "max": format_number(max(values)),
                }
            )


def summarize_root(root: Path, reference_header: list[str]) -> tuple[Path, Path, int]:
    stage3_dirs = discover_stage3_dirs(root)
    if not stage3_dirs:
        raise ValueError(f"No stage3_final directories found under {root}")

    rows = [summarize_stage3_dir(stage3, display_name) for stage3, display_name in stage3_dirs]
    out_csv = root / f"{root.name}_benchmark_transfer_summary_all_splits.csv"
    agg_csv = root / f"{root.name}_benchmark_transfer_summary_numeric_aggregate.csv"
    write_wide_summary(rows, reference_header, out_csv)
    write_numeric_aggregate(rows, reference_header, agg_csv)
    return out_csv, agg_csv, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize legacy400 benchmark/transfer metrics with BasicModel-compatible columns."
    )
    parser.add_argument("--reference_summary", type=Path, default=REFERENCE_SUMMARY)
    parser.add_argument("--roots", nargs="*", type=Path, default=[PROTOCOL_ROOT, RANDOM_ROOT])
    args = parser.parse_args()

    reference_header = read_reference_header(args.reference_summary)
    for root in args.roots:
        out_csv, agg_csv, n_rows = summarize_root(root, reference_header)
        print(f"[ok] {root} -> {out_csv} ({n_rows} experiments)")
        print(f"[ok] aggregate -> {agg_csv}")


if __name__ == "__main__":
    main()
