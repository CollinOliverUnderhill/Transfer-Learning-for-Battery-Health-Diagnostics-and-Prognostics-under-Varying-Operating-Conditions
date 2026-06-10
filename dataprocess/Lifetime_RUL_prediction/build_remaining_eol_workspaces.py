#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import Dict, Iterable, List


ROOT = Path(r"E:\Datasets\IVAS\Lifetime_RUL_prediction")
IVAS_ROOT = Path(r"E:\Datasets\IVAS")
EXCLUDE_ROOT = IVAS_ROOT / "Processing_Data_dd_exclude"
MULTIWEEK_DATA_CSV = IVAS_ROOT / "Processing_Data" / "Lifetime_prediction" / "ivas_lifetime_10features_multiweek_per_cell.csv"
GROUP_COND_CSV = IVAS_ROOT / "Groupcondi.csv"
CODE_ROOT = IVAS_ROOT / "Codes" / "chunqiu_codes" / "Lifetimepre"
EOLS = (50, 55, 65, 70, 75, 80)
WEEKS = (3, 5, 10, 15)

FIXED_CONFIG_TEMPLATE = {
    "task_type": "lifetime_prediction",
    "feature_aliases": ["f1", "f3", "f5", "f8", "f2"],
    "hidden_dims": [8, 8, 8],
    "dropout": 0.0,
    "activation": "relu",
    "epochs": 800,
    "ft_epochs": 800,
    "batch_size": 16,
    "lr": 7e-4,
    "ft_lr": 5e-5,
    "weight_decay": 1e-4,
    "ft_weight_decay": 1e-4,
    "val_cell_frac": 0.2,
    "early_stop_patience": 25,
    "min_epochs_before_early_stop": 0,
    "ft_min_epochs_before_early_stop": 0,
    "ft_freeze_hidden_layers": 2,
    "tail_q": 0.95,
    "seed": 2,
    "device": "auto",
    "experiment_name": "fixed_8x8x8_freeze2",
    "config_source": r"E:\Datasets\IVAS\Lifetime_Prediction\EOL60\Final_result\r02_8x8x8_relu_lr7e4_wd1e4_bs16_ft800_splitft_freeze2\transfer_model\config.json",
}

COPIED_CODE_FILES = (
    CODE_ROOT / "run_lifetime_transfer_mlp.py",
    CODE_ROOT / "Feature_extraction" / "extract_ivas_lifetime_multiweek_and_augment_eol.py",
    CODE_ROOT / "Feature_engineering" / "analyze_feature_lifetime_correlations_multiweek.py",
)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def copy_tree_contents(src: Path, dst: Path, include_dirs: Iterable[str], include_file_predicate) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    wanted_dirs = set(include_dirs)
    for item in src.iterdir():
        if item.is_dir() and item.name in wanted_dirs:
            shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
        elif item.is_file() and include_file_predicate(item.name):
            shutil.copy2(item, dst / item.name)


def render_split_script(eol: int) -> str:
    return dedent(
        f"""\
        #!/usr/bin/env python3
        # -*- coding: utf-8 -*-

        from __future__ import annotations

        import csv
        import math
        from collections import defaultdict
        from pathlib import Path
        from typing import Dict, List, Sequence


        ROOT = Path(r"E:\\Datasets\\IVAS\\Lifetime_RUL_prediction\\EOL{eol}")
        SOURCE_CSV = Path(r"E:\\Datasets\\IVAS\\Processing_Data_dd_exclude\\EOL{eol}\\cell_split_by_lifetime_EOL{eol}_augmented_weeks.csv")
        WEEKS = (3, 5, 10, 15)
        TARGET_COL = "lifetime_weeks_EOL{eol}"
        TRAIN_GROUP_FRAC = 0.70
        FT_CELL_FRAC = 0.15


        def to_float(value: object) -> float:
            text = str(value).strip()
            if text == "" or text.lower() == "nan":
                return float("nan")
            try:
                return float(text)
            except Exception:
                return float("nan")


        def round_half_up(value: float) -> int:
            return int(math.floor(value + 0.5))


        def read_rows(path: Path) -> List[Dict[str, str]]:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f))


        def write_rows(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)


        def usable_for_week(row: Dict[str, str], week: int) -> bool:
            for idx in range(1, 11):
                if not math.isfinite(to_float(row.get(f"f{{idx}}_w{{week}}", ""))):
                    return False
            return True


        def evenly_spaced_indices(n: int, k: int) -> List[int]:
            if k <= 0 or n <= 0:
                return []
            if k >= n:
                return list(range(n))
            if k == 1:
                return [n // 2]
            idxs = [int(round(i * (n - 1) / float(k - 1))) for i in range(k)]
            deduped: List[int] = []
            seen = set()
            for idx in idxs:
                idx = max(0, min(n - 1, idx))
                if idx not in seen:
                    seen.add(idx)
                    deduped.append(idx)
            cur = 0
            while len(deduped) < k:
                if cur not in seen:
                    seen.add(cur)
                    deduped.append(cur)
                cur += 1
            return sorted(deduped)


        def pick_ft_cells_spread(target_groups: Sequence[Dict[str, object]], target_rows: Sequence[Dict[str, str]], ft_quota: int) -> List[str]:
            rows_by_group: Dict[int, List[Dict[str, str]]] = defaultdict(list)
            for row in target_rows:
                rows_by_group[int(to_float(row["group_num"]))].append(row)

            ordered_group_rows: List[List[Dict[str, str]]] = []
            for group in target_groups:
                group_num = int(group["group_num"])
                cell_rows = sorted(
                    rows_by_group[group_num],
                    key=lambda r: (int(to_float(r["cell_idx"])), str(r["cell"])),
                )
                if cell_rows:
                    ordered_group_rows.append(cell_rows)

            total_target_cells = sum(len(items) for items in ordered_group_rows)
            if total_target_cells == 0:
                return []
            if total_target_cells == 1:
                return [str(ordered_group_rows[0][0]["cell"])]

            ft_quota = max(1, min(int(ft_quota), total_target_cells - 1))
            selected: List[str] = []
            selected_set = set()

            first_pass = evenly_spaced_indices(len(ordered_group_rows), min(ft_quota, len(ordered_group_rows)))
            for group_idx in first_pass:
                row = ordered_group_rows[group_idx][0]
                cell = str(row["cell"])
                selected.append(cell)
                selected_set.add(cell)

            while len(selected) < ft_quota:
                added = False
                for group_rows in ordered_group_rows:
                    for row in group_rows:
                        cell = str(row["cell"])
                        if cell in selected_set:
                            continue
                        selected.append(cell)
                        selected_set.add(cell)
                        added = True
                        break
                    if len(selected) >= ft_quota:
                        break
                if not added:
                    break

            return selected


        def build_group_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
            grouped: Dict[int, List[Dict[str, str]]] = defaultdict(list)
            for row in rows:
                grouped[int(to_float(row["group_num"]))].append(row)

            group_rows: List[Dict[str, object]] = []
            for group_num, items in grouped.items():
                lifetimes = [to_float(r[TARGET_COL]) for r in items if math.isfinite(to_float(r[TARGET_COL]))]
                if not lifetimes:
                    continue
                sorted_lifetimes = sorted(lifetimes)
                if len(sorted_lifetimes) % 2 == 1:
                    lifetime_median = sorted_lifetimes[len(sorted_lifetimes) // 2]
                else:
                    right = len(sorted_lifetimes) // 2
                    lifetime_median = (sorted_lifetimes[right - 1] + sorted_lifetimes[right]) / 2.0
                group_rows.append(
                    {{
                        "group_num": group_num,
                        "group_label": f"G{{group_num}}",
                        "valid_cell_count": len(items),
                        "lifetime_min": min(lifetimes),
                        "lifetime_median": lifetime_median,
                        "lifetime_mean": sum(lifetimes) / len(lifetimes),
                        "lifetime_max": max(lifetimes),
                        "cells": ",".join(sorted(str(r["cell"]) for r in items)),
                    }}
                )
            group_rows.sort(key=lambda row: (float(row["lifetime_median"]), float(row["lifetime_mean"]), int(row["group_num"])))
            return group_rows


        def main() -> None:
            domain_dir = ROOT / "domain_split"
            features_dir = ROOT / "features"
            rows = read_rows(SOURCE_CSV)

            write_rows(
                features_dir / "feature_table_all_cells_multiweek_EOL{eol}.csv",
                rows,
                list(rows[0].keys()),
            )

            manifest_rows: List[Dict[str, object]] = []

            for week in WEEKS:
                usable_rows = [dict(row) for row in rows if usable_for_week(row, week)]
                group_rows = build_group_rows(usable_rows)
                train_end = round_half_up(len(group_rows) * TRAIN_GROUP_FRAC)
                train_groups = group_rows[:train_end]
                target_groups = group_rows[train_end:]
                train_group_set = {{int(row["group_num"]) for row in train_groups}}
                target_group_set = {{int(row["group_num"]) for row in target_groups}}

                target_rows = [row for row in usable_rows if int(to_float(row["group_num"])) in target_group_set]
                ft_quota = round_half_up(len(usable_rows) * FT_CELL_FRAC)
                ft_cells = set(pick_ft_cells_spread(target_groups, target_rows, ft_quota))

                split_rows: List[Dict[str, object]] = []
                for row in usable_rows:
                    group_num = int(to_float(row["group_num"]))
                    cell = str(row["cell"])
                    if group_num in train_group_set:
                        split = "train"
                        split_label = "Train"
                        domain = "source"
                    elif cell in ft_cells:
                        split = "fine_tune"
                        split_label = "Fine-tune"
                        domain = "target"
                    else:
                        split = "test"
                        split_label = "Test"
                        domain = "target"
                    split_rows.append(
                        {{
                            **row,
                            "feature_week": f"w{{week}}",
                            "target_domain": domain,
                            "split": split,
                            "split_label": split_label,
                        }}
                    )

                split_rows.sort(key=lambda r: (str(r["split"]), int(to_float(r["group_num"])), int(to_float(r["cell_idx"])), str(r["cell"])))

                ft_counts = defaultdict(int)
                test_counts = defaultdict(int)
                for row in split_rows:
                    g = int(to_float(row["group_num"]))
                    if row["split"] == "fine_tune":
                        ft_counts[g] += 1
                    elif row["split"] == "test":
                        test_counts[g] += 1

                group_export: List[Dict[str, object]] = []
                for rank, group in enumerate(group_rows, start=1):
                    group_num = int(group["group_num"])
                    domain = "source" if group_num in train_group_set else "target"
                    group_export.append(
                        {{
                            "feature_week": f"w{{week}}",
                            "rank_by_lifetime": rank,
                            "domain": domain,
                            "group_num": group_num,
                            "group_label": group["group_label"],
                            "valid_cell_count": group["valid_cell_count"],
                            "ft_cell_count": ft_counts[group_num],
                            "test_cell_count": test_counts[group_num],
                            "lifetime_min": group["lifetime_min"],
                            "lifetime_median": group["lifetime_median"],
                            "lifetime_mean": group["lifetime_mean"],
                            "lifetime_max": group["lifetime_max"],
                            "cells": group["cells"],
                        }}
                    )

                split_summary = [
                    {{
                        "feature_week": f"w{{week}}",
                        "usable_cells": len(usable_rows),
                        "train_cells": sum(1 for r in split_rows if r["split"] == "train"),
                        "fine_tune_cells": sum(1 for r in split_rows if r["split"] == "fine_tune"),
                        "test_cells": sum(1 for r in split_rows if r["split"] == "test"),
                        "train_groups": len(train_groups),
                        "target_groups": len(target_groups),
                    }}
                ]

                split_csv = domain_dir / f"cell_split_targetspread_w{{week}}_EOL{eol}.csv"
                group_csv = domain_dir / f"group_split_targetspread_w{{week}}_EOL{eol}.csv"
                summary_csv = domain_dir / f"split_summary_targetspread_w{{week}}_EOL{eol}.csv"
                write_rows(split_csv, split_rows, list(split_rows[0].keys()))
                write_rows(group_csv, group_export, list(group_export[0].keys()))
                write_rows(summary_csv, split_summary, list(split_summary[0].keys()))

                manifest_rows.append(
                    {{
                        "feature_week": f"w{{week}}",
                        "split_csv": str(split_csv),
                        "group_csv": str(group_csv),
                        "summary_csv": str(summary_csv),
                        "usable_cells": len(usable_rows),
                    }}
                )

            write_rows(
                domain_dir / "weekly_split_manifest_EOL{eol}.csv",
                manifest_rows,
                list(manifest_rows[0].keys()),
            )


        if __name__ == "__main__":
            main()
        """
    )


def render_run_script(eol: int) -> str:
    return dedent(
        f"""\
        #!/usr/bin/env python3
        # -*- coding: utf-8 -*-

        from __future__ import annotations

        import csv
        import json
        import subprocess
        import sys
        from pathlib import Path
        from typing import Dict, List


        ROOT = Path(r"E:\\Datasets\\IVAS\\Lifetime_RUL_prediction\\EOL{eol}")
        RUNNER = ROOT / "codes" / "run_lifetime_transfer_mlp.py"
        CONFIG_JSON = ROOT / "codes" / "fixed_transfer_config.json"
        DATA_CSV = Path(r"E:\\Datasets\\IVAS\\Processing_Data\\Lifetime_prediction\\ivas_lifetime_10features_multiweek_per_cell.csv")
        GROUP_COND_CSV = Path(r"E:\\Datasets\\IVAS\\Groupcondi.csv")
        DOMAIN_SPLIT_DIR = ROOT / "domain_split"
        RESULTS_DIR = ROOT / "results"
        WEEKS = (3, 5, 10, 15)


        def load_config() -> Dict[str, object]:
            with CONFIG_JSON.open("r", encoding="utf-8") as f:
                return json.load(f)


        def run_one(week: int, cfg: Dict[str, object]) -> Path:
            out_root = RESULTS_DIR / f"{{cfg['experiment_name']}}_w{{week}}"
            split_csv = DOMAIN_SPLIT_DIR / f"cell_split_targetspread_w{{week}}_EOL{eol}.csv"
            features = ",".join(f"{{alias}}_w{{week}}" for alias in cfg["feature_aliases"])
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
                    {{
                        "eol": "EOL{eol}",
                        "week": f"w{{week}}",
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
                    }}
                )

            out_csv = RESULTS_DIR / "week_feature_transfer_comparison_repartitioned_EOL{eol}.csv"
            with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)


        if __name__ == "__main__":
            main()
        """
    )


def render_readme(eol: int) -> str:
    return dedent(
        f"""\
        # EOL{eol} RUL Workspace

        This folder is a self-contained workspace for the revised EOL{eol} setup.

        - `codes/`
          - split generation with target-domain fine-tune spread
          - fixed transfer-model config copied from the EOL60 baseline
          - training runner used for the final experiments
        - `domain_split/`
          - week-specific split files
          - group-level split summaries
          - manifest for all generated split files
        - `features/`
          - multi-week feature table for EOL{eol}
          - original exclude-based augmented split table
          - week availability summary
        - `feature_engineering/`
          - week-specific correlation csv / md / svg / html
          - scatter plot folders
        - `results/`
          - actual benchmark / transfer-model outputs for `w3 / w5 / w10 / w15`
          - comparison csv for the repartitioned experiments

        Current split logic:
        - source/train groups are taken from the lower-lifetime side after sorting by group lifetime median
        - target groups are the remaining groups
        - `fine_tune` cells are sampled inside the target groups so they are spread across the target domain
        - `test` cells are the remaining target-domain cells

        Current model config:
        - temporarily fixed to the validated EOL60 baseline (`8x8x8 / relu / lr=7e-4 / wd=1e-4 / bs=16 / ft_epochs=800 / freeze_hidden_layers=2`)
        - stored explicitly in `codes/fixed_transfer_config.json` so EOL-specific tuning can replace it later without changing other folders
        """
    )


def prepare_workspace(eol: int) -> None:
    eol_name = f"EOL{eol}"
    root = ROOT / eol_name
    codes_dir = root / "codes"
    domain_dir = root / "domain_split"
    features_dir = root / "features"
    fe_dir = root / "feature_engineering"
    results_dir = root / "results"
    for path in (codes_dir, domain_dir, features_dir, fe_dir, results_dir):
        path.mkdir(parents=True, exist_ok=True)

    cfg = dict(FIXED_CONFIG_TEMPLATE)
    cfg["y_col"] = f"lifetime_weeks_EOL{eol}"
    write_text(codes_dir / "fixed_transfer_config.json", json.dumps(cfg, indent=2))
    write_text(codes_dir / "build_weekly_target_spread_splits.py", render_split_script(eol))
    write_text(codes_dir / "run_weekly_feature_suite.py", render_run_script(eol))
    write_text(root / "README.md", render_readme(eol))

    for src in COPIED_CODE_FILES:
        shutil.copy2(src, codes_dir / src.name)

    exclude_dir = EXCLUDE_ROOT / eol_name
    shutil.copy2(
        exclude_dir / f"cell_split_by_lifetime_{eol_name}_augmented_weeks.csv",
        features_dir / f"cell_split_by_lifetime_{eol_name}_augmented_weeks.csv",
    )
    shutil.copy2(
        exclude_dir / f"week_availability_summary_{eol_name}.csv",
        features_dir / f"week_availability_summary_{eol_name}.csv",
    )

    copy_tree_contents(
        exclude_dir,
        fe_dir,
        include_dirs=[f"feature_plots_w{week}_{eol_name}" for week in WEEKS],
        include_file_predicate=lambda name: name.startswith("feature_lifetime_correlations") and eol_name in name,
    )


def run_workspace_pipeline(eol: int) -> None:
    eol_name = f"EOL{eol}"
    root = ROOT / eol_name
    subprocess.run(
        [sys.executable, str(root / "codes" / "build_weekly_target_spread_splits.py")],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(root / "codes" / "run_weekly_feature_suite.py")],
        check=True,
    )


def collect_root_summaries() -> None:
    result_rows: List[Dict[str, object]] = []
    split_rows: List[Dict[str, object]] = []
    workspace_rows: List[Dict[str, object]] = []

    for eol in EOLS:
        eol_name = f"EOL{eol}"
        root = ROOT / eol_name
        comparison_csv = root / "results" / f"week_feature_transfer_comparison_repartitioned_{eol_name}.csv"
        manifest_csv = root / "domain_split" / f"weekly_split_manifest_{eol_name}.csv"
        if comparison_csv.exists():
            for row in read_csv(comparison_csv):
                result_rows.append(row)
        if manifest_csv.exists():
            for row in read_csv(manifest_csv):
                split_rows.append({"eol": eol_name, **row})
        workspace_rows.append(
            {
                "eol": eol_name,
                "workspace_root": str(root),
                "readme": str(root / "README.md"),
                "config_json": str(root / "codes" / "fixed_transfer_config.json"),
                "comparison_csv": str(comparison_csv),
            }
        )

    write_csv(ROOT / "all_eols_week_feature_transfer_comparison.csv", result_rows)
    write_csv(ROOT / "all_eols_week_split_manifest.csv", split_rows)
    write_csv(ROOT / "workspace_manifest.csv", workspace_rows)


def main() -> None:
    if not MULTIWEEK_DATA_CSV.exists():
        raise FileNotFoundError(MULTIWEEK_DATA_CSV)
    if not GROUP_COND_CSV.exists():
        raise FileNotFoundError(GROUP_COND_CSV)

    for eol in EOLS:
        prepare_workspace(eol)
        run_workspace_pipeline(eol)

    collect_root_summaries()


if __name__ == "__main__":
    main()
