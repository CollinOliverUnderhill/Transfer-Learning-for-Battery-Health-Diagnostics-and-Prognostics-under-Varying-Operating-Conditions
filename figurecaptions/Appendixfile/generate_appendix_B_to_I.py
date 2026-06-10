from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "Figurecaption" / "Appendixfile"
APP_MD = ROOT / "Figurecaption" / "Appendix.md"

EOL = ROOT / "week_based" / "Final" / "EOL70"
FEATURE_DIR = EOL / "feature_engineering"
FEATURE_TABLE_DIR = EOL / "features"
DOMAIN_DIR = EOL / "domain_split"
MAIN_STAGE3 = EOL / "3step" / "outputs_400" / "BasicModel" / "stage3_final_rerun_400"
PROTOCOL_DIR = EOL / "3step" / "outputs_400" / "protocol_w6_10_from_stage3_final_rerun_400_legacy400"
RANDOM_DIR = EOL / "3step" / "outputs_400" / "random_w5_EOL70_10seeds_legacy400"
LOWCAP_DIR = EOL / "3step" / "outputs_lowcapacity"
SOH_INDEX = ROOT / "week_based" / "SOHest" / "results" / "soh_results_index.csv"
EOL70_TERM = "the end-of-life (EOL) threshold defined at 70% state of health (SOH)"
EOL70_SHORT = "70% SOH EOL threshold"


COLORS = {
    "benchmark": "#7A7A7A",
    "source_only": "#6BB7B2",
    "transfer": "#F28E2B",
    "positive": "#73B66B",
    "negative": "#D95F5F",
    "grid": "#D8D8D8",
    "text": "#222222",
}


def ensure_dir(name: str) -> Path:
    out = APP_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.7,
            "axes.edgecolor": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def save_fig(fig: plt.Figure, out_no_ext: Path) -> None:
    fig.savefig(out_no_ext.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_no_ext.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def read_metrics(path: Path, label: str, week: int | None = None) -> dict:
    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()
    out = {"model": label}
    if week is not None:
        out["week"] = week
    for key in [
        "mae",
        "rmse",
        "r2",
        "mape_percent",
        "smape_percent",
        "wmape_percent",
        "n_cells",
        "n_groups",
    ]:
        if key in row:
            out[key] = row[key]
    return out


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    shown = df.copy()
    if max_rows is not None:
        shown = shown.head(max_rows)
    for col in shown.columns:
        if shown[col].dtype == object:
            shown[col] = shown[col].map(lambda x: str(x).replace("|", r"\|") if pd.notna(x) else x)
    try:
        return shown.to_markdown(index=False)
    except Exception:
        return "```csv\n" + shown.to_csv(index=False) + "```"


def feature_catalog_as_numbered_text(df: pd.DataFrame) -> str:
    lines: list[str] = []
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        lines.append(f"{idx}. `{row.feature}` ({row.category})")
        lines.append("")
        lines.append(f"   {row.mathematical_expression}")
        lines.append("")
        lines.append(f"   {row.interpretation}")
        lines.append("")
    return "\n".join(lines).rstrip()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def short_float_frame(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: round(float(x), digits) if pd.notna(x) else x)
    return out


def sanitize_eol_terms_for_appendix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(
        columns={
            "lifetime_weeks_EOL70": "lifetime_weeks_at_70pct_soh_eol_threshold",
            "lifetime_weeks_EOL80": "lifetime_weeks_at_80pct_soh_eol_threshold",
        }
    )
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(
                lambda x: str(x)
                .replace("EOL70", "70pct_soh_eol_threshold")
                .replace("EOL80", "80pct_soh_eol_threshold")
                if pd.notna(x)
                else x
            )
    return out


def appendix_b() -> dict:
    out = ensure_dir("AppendixB")
    feature_catalog = pd.DataFrame(
        [
            {
                "feature": "f1_w5",
                "category": "IC-curve feature",
                "mathematical_expression": r"$\log(|\mathrm{mean}(\Delta(dQ/dV)_{w_5-w_0}^{3.6-3.9V})|)$",
                "interpretation": "Average early-life change in the incremental-capacity curve within the 3.6-3.9 V window.",
            },
            {
                "feature": "f2_w5",
                "category": "CV-time feature",
                "mathematical_expression": r"$\log(|\Delta CV\ time_{w_5-w_0}|)$",
                "interpretation": "Absolute early-life change in constant-voltage charging time.",
            },
            {
                "feature": "f3_w5",
                "category": "Usage-condition feature",
                "mathematical_expression": r"$DoD$",
                "interpretation": "Depth of discharge, representing the fraction of available capacity used during cycling.",
            },
            {
                "feature": "f4_w5",
                "category": "Capacity-response feature",
                "mathematical_expression": r"$\Delta Q^1_{w_5-w_0}$",
                "interpretation": "Early-life change in the first capacity-related descriptor.",
            },
            {
                "feature": "f5_w5",
                "category": "Usage-condition feature",
                "mathematical_expression": r"$C_{chg}^{0.5}DoD^{0.5}$",
                "interpretation": "Interaction between charge C-rate and depth of discharge.",
            },
            {
                "feature": "f6_w5",
                "category": "Usage-condition feature",
                "mathematical_expression": r"$C_{chg}$",
                "interpretation": "Charge C-rate, describing charging current relative to nominal cell capacity.",
            },
            {
                "feature": "f7_w5",
                "category": "IC-curve feature",
                "mathematical_expression": r"$\log(\mathrm{var}(\Delta(dQ/dV)_{w_5-w_0}^{3.0-3.6V}))$",
                "interpretation": "Variance of early-life incremental-capacity curve change within the 3.0-3.6 V window.",
            },
            {
                "feature": "f8_w5",
                "category": "Capacity-response feature",
                "mathematical_expression": r"$\Delta Q^3_{w_5-w_0}$",
                "interpretation": "Early-life change in the third capacity-related descriptor, complementary to the first descriptor.",
            },
            {
                "feature": "f9_w5",
                "category": "IC-curve feature",
                "mathematical_expression": r"$\log(|\mathrm{mean}(\Delta(dQ/dV)_{w_5-w_0}^{3.0-3.6V})|)$",
                "interpretation": "Average early-life change in the incremental-capacity curve within the lower-voltage 3.0-3.6 V window.",
            },
            {
                "feature": "f10_w5",
                "category": "CV-time feature",
                "mathematical_expression": r"$\log(|CV\ time_{w_0}|)$",
                "interpretation": "Initial constant-voltage charging time before the diagnostic ageing window.",
            },
        ]
    )
    feature_catalog.to_csv(out / "appendix_B_feature_notation_categories.csv", index=False)

    corr = pd.read_csv(FEATURE_DIR / "feature_lifetime_correlations_w5_EOL70.csv")
    corr = corr.rename(
        columns={
            "feature_key": "feature",
            "feature_short": "feature_description",
            "valid_n": "n",
        }
    )
    keep = ["feature", "feature_description", "pearson_r", "spearman_r", "n"]
    corr_out = corr[[c for c in keep if c in corr.columns]].copy()
    corr_out.to_csv(out / "appendix_B_w5_feature_correlations.csv", index=False)

    definitions = corr_out[[c for c in ["feature", "feature_description"] if c in corr_out.columns]].drop_duplicates()
    definitions.to_csv(out / "appendix_B_feature_definitions.csv", index=False)

    feature_table = pd.read_csv(FEATURE_TABLE_DIR / "feature_table_all_cells_multiweek_EOL70.csv")
    value_rows = []
    corr_lookup = corr_out.set_index("feature")
    for feature in feature_catalog["feature"]:
        values = pd.to_numeric(feature_table[feature], errors="coerce") if feature in feature_table.columns else pd.Series(dtype=float)
        row = {
            "feature": feature,
            "n_non_nan": int(values.notna().sum()),
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "median": values.median(),
            "max": values.max(),
        }
        if feature in corr_lookup.index:
            row["pearson_r_with_lifetime"] = corr_lookup.loc[feature, "pearson_r"]
            row["spearman_r_with_lifetime"] = corr_lookup.loc[feature, "spearman_r"]
        value_rows.append(row)
    value_summary = pd.DataFrame(value_rows)
    value_summary.to_csv(out / "appendix_B_w5_feature_value_summary.csv", index=False)

    matrix = pd.read_csv(FEATURE_TABLE_DIR / "correlation_matrix_w5_EOL70.csv", index_col=0)
    matrix.to_csv(out / "appendix_B_w5_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(4.9, 4.2))
    im = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(matrix.index, fontsize=6)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation", fontsize=8)
    ax.set_title("Feature correlation matrix, week 5", fontsize=9, pad=6)
    save_fig(fig, out / "appendix_B_w5_correlation_heatmap")

    return {
        "corr": corr_out,
        "definitions": definitions,
        "catalog": feature_catalog,
        "value_summary": value_summary,
        "files": sorted(out.glob("*")),
    }


def appendix_c() -> dict:
    out = ensure_dir("AppendixC")
    split_summary = pd.read_csv(DOMAIN_DIR / "split_summary_targetspread_w5_EOL70.csv")
    split_summary.to_csv(out / "appendix_C_w5_split_summary.csv", index=False)

    group_split = pd.read_csv(DOMAIN_DIR / "group_split_targetspread_w5_EOL70.csv")
    group_cols = [
        "feature_week",
        "rank_by_lifetime",
        "domain",
        "group_num",
        "valid_cell_count",
        "ft_cell_count",
        "test_cell_count",
        "lifetime_min",
        "lifetime_median",
        "lifetime_mean",
        "lifetime_max",
        "cells",
    ]
    group_split[[c for c in group_cols if c in group_split.columns]].to_csv(
        out / "appendix_C_w5_group_split.csv", index=False
    )

    cell_split = pd.read_csv(DOMAIN_DIR / "cell_split_targetspread_w5_EOL70.csv")
    cell_cols = [
        "split",
        "split_label",
        "target_domain",
        "group_num",
        "cell",
        "lifetime_weeks_EOL70",
        "chg_c_rate",
        "dchg_c_rate",
        "f3_w5",
        "feature_status_w5",
    ]
    cell_split_out = cell_split[[c for c in cell_cols if c in cell_split.columns]]
    sanitize_eol_terms_for_appendix(cell_split_out).to_csv(
        out / "appendix_C_w5_cell_split.csv", index=False
    )

    cell_summary = (
        cell_split.groupby(["split", "split_label"], dropna=False)
        .agg(
            cells=("cell", "nunique"),
            groups=("group_num", "nunique"),
            lifetime_min=("lifetime_weeks_EOL70", "min"),
            lifetime_median=("lifetime_weeks_EOL70", "median"),
            lifetime_max=("lifetime_weeks_EOL70", "max"),
        )
        .reset_index()
    )
    cell_summary.to_csv(out / "appendix_C_w5_cell_split_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.9, 3.35))
    plot_df = cell_split.copy()
    jitter = {"train": -0.012, "fine_tune": 0.0, "test": 0.012}
    color_map = {"train": COLORS["source_only"], "fine_tune": COLORS["transfer"], "test": "#4E79A7"}
    label_map = {"train": "Source train", "fine_tune": "Target fine-tune", "test": "Target test"}
    for split, sub in plot_df.groupby("split"):
        x = sub["chg_c_rate"].astype(float) + jitter.get(split, 0)
        y = sub["dchg_c_rate"].astype(float) + jitter.get(split, 0)
        ax.scatter(
            x,
            y,
            s=28,
            alpha=0.78,
            color=color_map.get(split, "#999999"),
            edgecolor="white",
            linewidth=0.35,
            label=label_map.get(split, split),
        )
    ax.set_xlabel("Charge C-rate")
    ax.set_ylabel("Discharge C-rate")
    ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.8)
    ax.legend(loc="best", fontsize=7)
    ax.set_title("w5 split distribution in operating-condition space", fontsize=9, pad=6)
    save_fig(fig, out / "appendix_C_w5_condition_split_scatter")

    for stem in ["plot_condition_split_selection_3d_cell_jitter_w5_EOL70"]:
        for ext in [".png", ".pdf"]:
            src = DOMAIN_DIR / f"{stem}{ext}"
            if src.exists():
                shutil.copy2(
                    src,
                    out / f"appendix_C_condition_split_selection_3d_cell_jitter_w5_70pct_soh_eol_threshold{ext}",
                )

    return {"split_summary": split_summary, "cell_summary": cell_summary, "files": sorted(out.glob("*"))}


def appendix_c_model_config() -> dict:
    out = ensure_dir("AppendixC")
    selected = sanitize_eol_terms_for_appendix(pd.read_csv(MAIN_STAGE3 / "selected_stage2_config.csv"))
    selected.to_csv(out / "appendix_C_selected_stage2_config.csv", index=False)

    with open(MAIN_STAGE3 / "final_selection.json", "r", encoding="utf-8") as f:
        selection = json.load(f)

    rows = []
    stage1 = selection.get("selected_stage1_summary", {})
    stage2 = selection.get("selected_stage2_row", {})
    for key in [
        "features",
        "hidden_dims",
        "dropout",
        "activation",
        "epochs",
        "lr",
        "weight_decay",
        "source_stage1_val_mae",
    ]:
        if key in stage1 or key in stage2:
            rows.append({"section": "source pretraining", "parameter": key, "value": stage1.get(key, stage2.get(key))})
    for key in [
        "ft_lr",
        "ft_weight_decay",
        "ft_epochs",
        "ft_freeze_hidden_layers",
        "target_support_ratio",
        "transfer_replay_weight",
        "target_ft_val_mae",
    ]:
        if key in stage2:
            rows.append({"section": "target fine-tuning", "parameter": key, "value": stage2.get(key)})
    rows.append({"section": "final evaluation", "parameter": "feature_week", "value": "w5"})
    rows.append({"section": "final evaluation", "parameter": "EOL threshold", "value": EOL70_TERM})
    config = pd.DataFrame(rows)
    config.to_csv(out / "appendix_C_final_model_config.csv", index=False)

    hpo = pd.DataFrame(
        [
            {"stage": "Stage 1 source search", "parameter": "feature subset", "search_space_or_setting": "candidate engineered features"},
            {"stage": "Stage 1 source search", "parameter": "hidden dimensions", "search_space_or_setting": "candidate MLP architectures"},
            {"stage": "Stage 1 source search", "parameter": "dropout", "search_space_or_setting": "categorical candidates"},
            {"stage": "Stage 1 source search", "parameter": "activation", "search_space_or_setting": "categorical candidates"},
            {"stage": "Stage 1 source search", "parameter": "learning rate", "search_space_or_setting": "log-uniform range from script arguments"},
            {"stage": "Stage 1 source search", "parameter": "weight decay", "search_space_or_setting": "log-uniform range from script arguments"},
            {"stage": "Stage 2 fine-tuning search", "parameter": "source checkpoint", "search_space_or_setting": "top-ranked Stage 1 candidates"},
            {"stage": "Stage 2 fine-tuning search", "parameter": "fine-tuning learning rate", "search_space_or_setting": "log-uniform range from script arguments"},
            {"stage": "Stage 2 fine-tuning search", "parameter": "fine-tuning weight decay", "search_space_or_setting": "log-uniform range from script arguments"},
            {"stage": "Stage 2 fine-tuning search", "parameter": "support-cell ratio", "search_space_or_setting": "categorical candidates"},
            {"stage": "Stage 2 fine-tuning search", "parameter": "fine-tuning epochs", "search_space_or_setting": "categorical candidates"},
            {"stage": "Stage 2 fine-tuning search", "parameter": "frozen hidden layers", "search_space_or_setting": "integer candidates"},
            {"stage": "Stage 2 fine-tuning search", "parameter": "source replay weight", "search_space_or_setting": "categorical candidates"},
        ]
    )
    hpo.to_csv(out / "appendix_C_hpo_search_space.csv", index=False)
    return {"config": config, "selected": selected, "hpo": hpo, "files": sorted(out.glob("*"))}


def appendix_e() -> dict:
    out = ensure_dir("AppendixE")
    soh = pd.read_csv(SOH_INDEX)
    soh.to_csv(out / "appendix_E_soh_results_index.csv", index=False)

    compact_cols = [
        "category",
        "model",
        "setting",
        "split",
        "n_cells",
        "n_groups",
        "mae",
        "rmse",
        "r2",
        "mape_percent",
        "source_only_mae",
        "target",
    ]
    compact = soh[[c for c in compact_cols if c in soh.columns]].copy()
    compact = compact.sort_values(["category", "model", "setting"]).reset_index(drop=True)
    compact.to_csv(out / "appendix_E_soh_results_summary.csv", index=False)

    plot_df = compact.dropna(subset=["mae"]).copy()
    plot_df["label"] = plot_df["category"].astype(str) + "\n" + plot_df["setting"].astype(str)
    plot_df = plot_df.sort_values("mae", ascending=False).head(14).sort_values("mae", ascending=True)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.barh(np.arange(len(plot_df)), plot_df["mae"], color=COLORS["source_only"], edgecolor="none")
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["label"], fontsize=6.5)
    ax.set_xlabel("SOH estimation MAE")
    ax.grid(True, axis="x", color=COLORS["grid"], linewidth=0.5, alpha=0.8)
    ax.set_title("Supplementary SOH estimation results", fontsize=9, pad=6)
    save_fig(fig, out / "appendix_E_soh_mae_overview")

    return {"compact": compact, "files": sorted(out.glob("*"))}


def appendix_f() -> dict:
    out = ensure_dir("AppendixF")
    rows = [
        read_metrics(MAIN_STAGE3 / "benchmark" / "test_overall_metrics.csv", "benchmark"),
        read_metrics(MAIN_STAGE3 / "transfer_model" / "test_overall_metrics_source_only.csv", "source_only"),
        read_metrics(MAIN_STAGE3 / "transfer_model" / "test_overall_metrics.csv", "fine_tuned_transfer"),
    ]
    overall = pd.DataFrame(rows)
    overall.to_csv(out / "appendix_F_main_rul_overall_metrics.csv", index=False)

    for name, rel_path in [
        ("benchmark_group", MAIN_STAGE3 / "benchmark" / "test_group_metrics.csv"),
        ("benchmark_cell", MAIN_STAGE3 / "benchmark" / "test_cell_metrics.csv"),
        ("source_only_group", MAIN_STAGE3 / "transfer_model" / "test_group_metrics_source_only.csv"),
        ("source_only_cell", MAIN_STAGE3 / "transfer_model" / "test_cell_metrics_source_only.csv"),
        ("transfer_group", MAIN_STAGE3 / "transfer_model" / "test_group_metrics.csv"),
        ("transfer_cell", MAIN_STAGE3 / "transfer_model" / "test_cell_metrics.csv"),
    ]:
        if rel_path.exists():
            pd.read_csv(rel_path).to_csv(out / f"appendix_F_{name}_metrics.csv", index=False)

    lowcap_file = LOWCAP_DIR / "outputs_lowcapacity_benchmark_transfer_summary_all_splits.csv"
    lowcap = pd.DataFrame()
    if lowcap_file.exists():
        lowcap = pd.read_csv(lowcap_file)
        lowcap_cols = [
            "stage3_dir",
            "bench_test_n_cells",
            "bench_test_mae",
            "bench_test_rmse",
            "bench_test_mape",
            "bench_test_r2",
            "transfer_test_n_cells",
            "transfer_test_mae",
            "transfer_test_rmse",
            "transfer_test_mape",
            "transfer_test_r2",
            "test_transfer_vs_bench_mae_improve_percent",
        ]
        lowcap[[c for c in lowcap_cols if c in lowcap.columns]].to_csv(
            out / "appendix_F_low_capacity_stress_test.csv", index=False
        )

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6))
    x = np.arange(len(overall))
    labels = ["Benchmark", "Source-only", "Fine-tuned"]
    bar_colors = [COLORS["benchmark"], COLORS["source_only"], COLORS["transfer"]]
    axes[0].bar(x, overall["mae"], color=bar_colors, edgecolor="none")
    axes[0].set_ylabel("MAE (weeks)")
    axes[1].bar(x, overall["mape_percent"], color=bar_colors, edgecolor="none")
    axes[1].set_ylabel("MAPE (%)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(True, axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.8)
    fig.suptitle("Main RUL test performance at the 70% SOH EOL threshold, w5", fontsize=9, y=1.03)
    save_fig(fig, out / "appendix_F_main_rul_test_metrics")

    return {"overall": overall, "lowcap": lowcap, "files": sorted(out.glob("*"))}


def appendix_g() -> dict:
    out = ensure_dir("AppendixG")
    rows = []
    rows.extend(
        [
            read_metrics(MAIN_STAGE3 / "benchmark" / "test_overall_metrics.csv", "benchmark", 5),
            read_metrics(MAIN_STAGE3 / "transfer_model" / "test_overall_metrics_source_only.csv", "source_only", 5),
            read_metrics(MAIN_STAGE3 / "transfer_model" / "test_overall_metrics.csv", "fine_tuned_transfer", 5),
        ]
    )
    for week in [6, 7, 8, 9, 10]:
        stage = PROTOCOL_DIR / f"week{week}" / "stage3_final"
        rows.extend(
            [
                read_metrics(stage / "benchmark" / "test_overall_metrics.csv", "benchmark", week),
                read_metrics(stage / "transfer_model" / "test_overall_metrics_source_only.csv", "source_only", week),
                read_metrics(stage / "transfer_model" / "test_overall_metrics.csv", "fine_tuned_transfer", week),
            ]
        )
    week_metrics = pd.DataFrame(rows)
    week_metrics.to_csv(out / "appendix_G_week_sensitivity_overall_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.7), sharex=True)
    color_map = {
        "benchmark": COLORS["benchmark"],
        "source_only": COLORS["source_only"],
        "fine_tuned_transfer": COLORS["transfer"],
    }
    label_map = {
        "benchmark": "Benchmark",
        "source_only": "Source-only",
        "fine_tuned_transfer": "Fine-tuned",
    }
    for model, sub in week_metrics.groupby("model"):
        sub = sub.sort_values("week")
        axes[0].plot(sub["week"], sub["mae"], marker="o", linewidth=1.3, markersize=3.5, color=color_map[model], label=label_map[model])
        axes[1].plot(sub["week"], sub["mape_percent"], marker="o", linewidth=1.3, markersize=3.5, color=color_map[model], label=label_map[model])
    axes[0].set_ylabel("MAE (weeks)")
    axes[1].set_ylabel("MAPE (%)")
    for ax in axes:
        ax.set_xlabel("Feature week")
        ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.8)
        ax.set_xticks([5, 6, 7, 8, 9, 10])
    axes[1].legend(loc="best", fontsize=7)
    fig.suptitle("Week-based sensitivity under the 70% SOH EOL threshold", fontsize=9, y=1.03)
    save_fig(fig, out / "appendix_G_week_sensitivity_metrics")

    return {"week_metrics": week_metrics, "files": sorted(out.glob("*"))}


def appendix_h() -> dict:
    out = ensure_dir("AppendixH")
    random_file = RANDOM_DIR / "random_w5_EOL70_10seeds_legacy400_benchmark_transfer_summary_all_splits.csv"
    rnd = pd.read_csv(random_file)
    rnd["seed"] = rnd["stage3_dir"].astype(str).str.extract(r"(seed\d+)")[0]
    keep = [
        "seed",
        "stage3_dir",
        "bench_test_n_cells",
        "bench_test_mae",
        "bench_test_rmse",
        "bench_test_mape",
        "bench_test_r2",
        "transfer_test_n_cells",
        "transfer_test_mae",
        "transfer_test_rmse",
        "transfer_test_mape",
        "transfer_test_r2",
        "test_transfer_vs_bench_mae_improve_percent",
        "test_transfer_vs_bench_mape_improve_percent",
        "test_transfer_vs_bench_r2_delta",
    ]
    compact = rnd[[c for c in keep if c in rnd.columns]].copy()
    compact.to_csv(out / "appendix_H_random_target_selection_seed_results.csv", index=False)

    agg_file = RANDOM_DIR / "random_w5_EOL70_10seeds_legacy400_benchmark_transfer_summary_numeric_aggregate.csv"
    if agg_file.exists():
        shutil.copy2(agg_file, out / "appendix_H_random_target_selection_numeric_aggregate.csv")

    plot_df = compact.sort_values("seed").copy()
    vals = plot_df["test_transfer_vs_bench_mae_improve_percent"].astype(float)
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in vals]
    fig, ax = plt.subplots(figsize=(6.1, 2.9))
    x = np.arange(len(plot_df))
    ax.bar(x, vals, color=colors, edgecolor="none")
    ax.axhline(0, color=COLORS["text"], linewidth=0.8)
    ax.set_ylabel("MAE improvement vs. benchmark (%)")
    ax.set_xlabel("Random target-cell selection seed")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["seed"], rotation=45, ha="right", fontsize=6.5)
    ax.grid(True, axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.8)
    ax.set_title("Sensitivity to random target-cell selection, w5, 70% SOH EOL threshold", fontsize=9, pad=6)
    save_fig(fig, out / "appendix_H_random_target_selection_mae_improvement")

    return {"compact": compact, "files": sorted(out.glob("*"))}


def appendix_d_sensitivity() -> dict:
    out = ensure_dir("AppendixD")

    rows = []
    rows.extend(
        [
            read_metrics(MAIN_STAGE3 / "benchmark" / "test_overall_metrics.csv", "benchmark", 5),
            read_metrics(MAIN_STAGE3 / "transfer_model" / "test_overall_metrics_source_only.csv", "source_only", 5),
            read_metrics(MAIN_STAGE3 / "transfer_model" / "test_overall_metrics.csv", "fine_tuned_transfer", 5),
        ]
    )
    for week in [6, 7, 8, 9, 10]:
        stage = PROTOCOL_DIR / f"week{week}" / "stage3_final"
        rows.extend(
            [
                read_metrics(stage / "benchmark" / "test_overall_metrics.csv", "benchmark", week),
                read_metrics(stage / "transfer_model" / "test_overall_metrics_source_only.csv", "source_only", week),
                read_metrics(stage / "transfer_model" / "test_overall_metrics.csv", "fine_tuned_transfer", week),
            ]
        )
    week_metrics = pd.DataFrame(rows)
    week_metrics.to_csv(out / "appendix_D_week_sensitivity_overall_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.7), sharex=True)
    color_map = {
        "benchmark": COLORS["benchmark"],
        "source_only": COLORS["source_only"],
        "fine_tuned_transfer": COLORS["transfer"],
    }
    label_map = {
        "benchmark": "Benchmark",
        "source_only": "Source-only",
        "fine_tuned_transfer": "Fine-tuned",
    }
    for model, sub in week_metrics.groupby("model"):
        sub = sub.sort_values("week")
        axes[0].plot(sub["week"], sub["mae"], marker="o", linewidth=1.3, markersize=3.5, color=color_map[model], label=label_map[model])
        axes[1].plot(sub["week"], sub["mape_percent"], marker="o", linewidth=1.3, markersize=3.5, color=color_map[model], label=label_map[model])
    axes[0].set_ylabel("MAE (weeks)")
    axes[1].set_ylabel("MAPE (%)")
    for ax in axes:
        ax.set_xlabel("Feature week")
        ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.8)
        ax.set_xticks([5, 6, 7, 8, 9, 10])
    axes[1].legend(loc="best", fontsize=7)
    fig.suptitle("Week-based sensitivity under the 70% SOH EOL threshold", fontsize=9, y=1.03)
    save_fig(fig, out / "appendix_D_week_sensitivity_metrics")

    random_file = RANDOM_DIR / "random_w5_EOL70_10seeds_legacy400_benchmark_transfer_summary_all_splits.csv"
    rnd = pd.read_csv(random_file)
    rnd["seed"] = rnd["stage3_dir"].astype(str).str.extract(r"(seed\d+)")[0]
    keep = [
        "seed",
        "stage3_dir",
        "bench_test_n_cells",
        "bench_test_mae",
        "bench_test_rmse",
        "bench_test_mape",
        "bench_test_r2",
        "transfer_test_n_cells",
        "transfer_test_mae",
        "transfer_test_rmse",
        "transfer_test_mape",
        "transfer_test_r2",
        "test_transfer_vs_bench_mae_improve_percent",
        "test_transfer_vs_bench_mape_improve_percent",
        "test_transfer_vs_bench_r2_delta",
    ]
    random_compact = sanitize_eol_terms_for_appendix(rnd[[c for c in keep if c in rnd.columns]].copy())
    random_compact.to_csv(out / "appendix_D_random_target_selection_seed_results.csv", index=False)

    agg_file = RANDOM_DIR / "random_w5_EOL70_10seeds_legacy400_benchmark_transfer_summary_numeric_aggregate.csv"
    if agg_file.exists():
        agg = sanitize_eol_terms_for_appendix(pd.read_csv(agg_file))
        agg.to_csv(out / "appendix_D_random_target_selection_numeric_aggregate.csv", index=False)

    plot_df = random_compact.sort_values("seed").copy()
    vals = plot_df["test_transfer_vs_bench_mae_improve_percent"].astype(float)
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in vals]
    fig, ax = plt.subplots(figsize=(6.1, 2.9))
    x = np.arange(len(plot_df))
    ax.bar(x, vals, color=colors, edgecolor="none")
    ax.axhline(0, color=COLORS["text"], linewidth=0.8)
    ax.set_ylabel("MAE improvement vs. benchmark (%)")
    ax.set_xlabel("Random target-cell selection seed")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["seed"], rotation=45, ha="right", fontsize=6.5)
    ax.grid(True, axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.8)
    ax.set_title("Sensitivity to random target-cell selection, w5, 70% SOH EOL threshold", fontsize=9, pad=6)
    save_fig(fig, out / "appendix_D_random_target_selection_mae_improvement")

    return {
        "week_metrics": week_metrics,
        "random_compact": random_compact,
        "files": sorted(out.glob("*")),
    }


def appendix_e_reproducibility() -> dict:
    out = ensure_dir("AppendixE")
    code_map = pd.DataFrame(
        [
            {"purpose": f"{EOL70_SHORT} feature table and week availability", "path": "week_based/Final/EOL70/features"},
            {"purpose": "Feature-lifetime correlation analysis", "path": "week_based/Final/EOL70/feature_engineering"},
            {"purpose": "Hyperparameter search and final RUL model configuration", "path": "week_based/Final/EOL70/3step"},
            {"purpose": "Main w5 final RUL run", "path": "week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400"},
            {"purpose": "Week-sensitivity runs", "path": "week_based/Final/EOL70/3step/outputs_400/protocol_w6_10_from_stage3_final_rerun_400_legacy400"},
            {"purpose": "Random target-cell selection runs", "path": "week_based/Final/EOL70/3step/outputs_400/random_w5_EOL70_10seeds_legacy400"},
            {"purpose": "Appendix generation scripts and assets", "path": "Figurecaption/Appendixfile"},
        ]
    )
    code_map = sanitize_eol_terms_for_appendix(code_map)
    code_map.to_csv(out / "appendix_E_code_map.csv", index=False)

    checklist = pd.DataFrame(
        [
            {"item": "Python environment", "value": "local conda base environment"},
            {"item": "Figure format", "value": "PNG and PDF; no SVG generated for appendix figures"},
            {"item": "Main endpoint", "value": f"RUL label under {EOL70_TERM}"},
            {"item": "Main feature week", "value": "w5"},
            {"item": "Supplementary week sensitivity", "value": "w5-w10"},
            {"item": "Random selection analysis", "value": "reported as seed-level sensitivity only; no new metric defined"},
            {"item": "Appendix script", "value": "Figurecaption/Appendixfile/generate_appendix_B_to_I.py"},
        ]
    )
    checklist.to_csv(out / "appendix_E_reproducibility_checklist.csv", index=False)
    return {"code_map": code_map, "checklist": checklist, "files": sorted(out.glob("*"))}


def build_markdown(results: dict) -> str:
    a_dir = APP_DIR / "AppendixA"
    a_tables = a_dir / "appendix_A_generated_tables.md"
    a_extra = ""
    if a_tables.exists():
        a_extra = a_tables.read_text(encoding="utf-8")

    b = results["B"]
    c = results["C"]
    d = results["D"]
    e = results["E"]
    f = results["F"]
    g = results["G"]
    h = results["H"]
    i = results["I"]

    b_corr = short_float_frame(b["corr"], 3)
    b_catalog = b["catalog"].copy()
    b_values = short_float_frame(b["value_summary"], 4)
    c_summary = short_float_frame(c["cell_summary"], 3)
    d_config = short_float_frame(d["config"], 4)
    e_summary = short_float_frame(e["compact"], 4)
    f_overall = short_float_frame(f["overall"], 3)
    g_metrics = short_float_frame(g["week_metrics"], 3)
    h_summary = short_float_frame(h["compact"], 3)

    code_map_display = i["code_map"].copy()
    if "path" in code_map_display.columns:
        code_map_display["path"] = code_map_display["path"].str.replace(
            "week_based/Final/EOL70",
            "week_based/Final/[70% SOH EOL-threshold results]",
            regex=False,
        )
        code_map_display["path"] = code_map_display["path"].str.replace(
            "random_w5_EOL70_10seeds_legacy400",
            "random_w5_[70% SOH EOL threshold]_10seeds_legacy400",
            regex=False,
        )

    lines = [
        "# Thesis Appendix",
        "",
        "This file collects the proposed appendix material for the thesis. The generated tables and figures are stored under `Figurecaption/Appendixfile` in section-specific folders. Appendix figures are provided as PNG and PDF files.",
        "",
        "Important boundary: this appendix reports data availability, engineered features, split details, model configuration, and supplementary results. It does not define a new target-cell selection metric; random target-cell selection is reported only as an empirical sensitivity result.",
        "",
        "## Appendix A: Dataset and Cell Availability",
        "",
        f"Purpose: document cell-level availability and operating conditions used by the RUL experiments under {EOL70_TERM}. This section supports the statement that week-based sensitivity is affected by changing sample availability.",
        "",
        a_extra.strip(),
        "",
        "Suggested figures:",
        "",
        f"- `Figurecaption/Appendixfile/AppendixA/appendix_A_week_availability.png` / `.pdf`",
        f"- `Figurecaption/Appendixfile/AppendixA/appendix_A_eol_label_availability.png` / `.pdf`",
        "",
        "## Appendix B: Engineered Feature Definitions",
        "",
        "Purpose: list the engineered early-cycle features used for RUL prediction and provide the feature-correlation context for the main w5 experiments.",
        "",
        "Notation:",
        "",
        "- Let `w0` denote the baseline diagnostic week and `wk` denote a later early-life diagnostic week. In the main experiment reported here, `k = 5`.",
        "- For any descriptor `x`, `Delta x_{wk-w0} = x_{wk} - x_{w0}`.",
        "- `dQ/dV` denotes the incremental-capacity curve.",
        "- `CV time` denotes the constant-voltage charging time.",
        "- `C_chg` denotes charge C-rate.",
        "- `DoD` denotes depth of discharge.",
        "",
        "Basic feature categories:",
        "",
        "- IC-curve features quantify early electrochemical response changes from `dQ/dV` curves in selected voltage windows.",
        "- CV-time features describe changes in constant-voltage charging behaviour.",
        "- Usage-condition features include `DoD`, `C_chg`, and their interaction term.",
        "- Capacity-response features describe early capacity-related changes through `Delta Q^1` and `Delta Q^3` descriptors.",
        "",
        "Files:",
        "",
        "- `Figurecaption/Appendixfile/AppendixB/appendix_B_feature_notation_categories.csv`",
        "- `Figurecaption/Appendixfile/AppendixB/appendix_B_feature_definitions.csv`",
        "- `Figurecaption/Appendixfile/AppendixB/appendix_B_w5_feature_correlations.csv`",
        "- `Figurecaption/Appendixfile/AppendixB/appendix_B_w5_feature_value_summary.csv`",
        "- `Figurecaption/Appendixfile/AppendixB/appendix_B_w5_correlation_heatmap.png` / `.pdf`",
        "",
        "Feature notation, category, and interpretation:",
        "",
        feature_catalog_as_numbered_text(b_catalog),
        "",
        "Current numerical values for the week-5 feature table:",
        "",
        md_table(b_values),
        "",
        "Feature-lifetime correlations for w5:",
        "",
        md_table(b_corr),
        "",
        f"Suggested caption: Supplementary feature correlation matrix for the week-5 RUL feature set under {EOL70_TERM}. The heatmap is used only to document the degree of feature redundancy and does not introduce an additional selection metric.",
        "",
        "## Appendix C: Domain Split Details",
        "",
        "Purpose: make the source-domain, target fine-tuning, and target-test composition transparent at the group and cell levels.",
        "",
        "Files:",
        "",
        "- `Figurecaption/Appendixfile/AppendixC/appendix_C_w5_split_summary.csv`",
        "- `Figurecaption/Appendixfile/AppendixC/appendix_C_w5_group_split.csv`",
        "- `Figurecaption/Appendixfile/AppendixC/appendix_C_w5_cell_split.csv`",
        "- `Figurecaption/Appendixfile/AppendixC/appendix_C_w5_condition_split_scatter.png` / `.pdf`",
        "",
        "Cell-level split summary:",
        "",
        md_table(c_summary),
        "",
        f"Suggested caption: Source, fine-tuning, and target-test cells under the w5 split for {EOL70_TERM}, shown in the operating-condition space. The plot is a transparency check for the split composition rather than a new method.",
        "",
        "## Appendix D: Hyperparameter Search and Final Model Configuration",
        "",
        "Purpose: document the Stage 1 source search, Stage 2 fine-tuning search, and the final selected model configuration used in the main w5 RUL experiment.",
        "",
        "Files:",
        "",
        "- `Figurecaption/Appendixfile/AppendixD/appendix_D_hpo_search_space.csv`",
        "- `Figurecaption/Appendixfile/AppendixD/appendix_D_selected_stage2_config.csv`",
        "- `Figurecaption/Appendixfile/AppendixD/appendix_D_final_model_config.csv`",
        "",
        "Final selected configuration:",
        "",
        md_table(d_config),
        "",
        "## Appendix E: Supplementary SOH Estimation Results",
        "",
        "Purpose: provide the supplementary SOH estimation results that support the data-processing and representation-learning context of the thesis.",
        "",
        "Files:",
        "",
        "- `Figurecaption/Appendixfile/AppendixE/appendix_E_soh_results_index.csv`",
        "- `Figurecaption/Appendixfile/AppendixE/appendix_E_soh_results_summary.csv`",
        "- `Figurecaption/Appendixfile/AppendixE/appendix_E_soh_mae_overview.png` / `.pdf`",
        "",
        "Compact SOH result table:",
        "",
        md_table(e_summary, max_rows=18),
        "",
        "Suggested caption: Supplementary SOH estimation MAE across model and split settings. The complete table is provided as CSV.",
        "",
        "## Appendix F: Supplementary Main RUL Prediction Results",
        "",
        f"Purpose: provide additional test metrics for the main week-5 RUL experiment under {EOL70_TERM}, including benchmark, source-only transfer, and fine-tuned transfer models.",
        "",
        "Files:",
        "",
        "- `Figurecaption/Appendixfile/AppendixF/appendix_F_main_rul_overall_metrics.csv`",
        "- `Figurecaption/Appendixfile/AppendixF/appendix_F_*_group_metrics.csv`",
        "- `Figurecaption/Appendixfile/AppendixF/appendix_F_*_cell_metrics.csv`",
        "- `Figurecaption/Appendixfile/AppendixF/appendix_F_low_capacity_stress_test.csv`",
        "- `Figurecaption/Appendixfile/AppendixF/appendix_F_main_rul_test_metrics.png` / `.pdf`",
        "",
        "Overall test metrics:",
        "",
        md_table(f_overall),
        "",
        f"Suggested caption: Overall RUL test performance for the main week-5 experiment under {EOL70_TERM}. The low-capacity stress-test table is included as a supplementary robustness check.",
        "",
        "## Appendix G: Week-Based Sensitivity Results",
        "",
        "Purpose: report how the RUL prediction results change when the observation week changes. This section should be interpreted together with Appendix A because later weeks reduce the number of usable cells.",
        "",
        "Files:",
        "",
        "- `Figurecaption/Appendixfile/AppendixG/appendix_G_week_sensitivity_overall_metrics.csv`",
        "- `Figurecaption/Appendixfile/AppendixG/appendix_G_week_sensitivity_metrics.png` / `.pdf`",
        "",
        "Week-level overall metrics:",
        "",
        md_table(g_metrics),
        "",
        f"Suggested caption: Week-based sensitivity of RUL prediction from w5 to w10 under {EOL70_TERM}. The figure reports observed model performance under each available split and does not define a separate quantitative score.",
        "",
        "## Appendix H: Random Target-Cell Selection Sensitivity",
        "",
        "Purpose: report the seed-level sensitivity caused by changing which target cells are selected for fine-tuning. This section is purely empirical and should not introduce a new target-cell selection metric.",
        "",
        "Files:",
        "",
        "- `Figurecaption/Appendixfile/AppendixH/appendix_H_random_target_selection_seed_results.csv`",
        "- `Figurecaption/Appendixfile/AppendixH/appendix_H_random_target_selection_numeric_aggregate.csv`",
        "- `Figurecaption/Appendixfile/AppendixH/appendix_H_random_target_selection_mae_improvement.png` / `.pdf`",
        "",
        "Seed-level results:",
        "",
        md_table(h_summary),
        "",
        "Suggested caption: Seed-level test MAE improvement of fine-tuned transfer relative to the benchmark under random target-cell selection. Positive bars indicate improvement over the benchmark; negative bars indicate degradation.",
        "",
        "## Appendix I: Code and Reproducibility Notes",
        "",
        "Purpose: provide a concise code map and reproducibility checklist for locating the inputs, generated appendix files, and experiment outputs.",
        "",
        "Files:",
        "",
        "- `Figurecaption/Appendixfile/AppendixI/appendix_I_code_map.csv`",
        "- `Figurecaption/Appendixfile/AppendixI/appendix_I_reproducibility_checklist.csv`",
        "",
        "Code map:",
        "",
        md_table(code_map_display),
        "",
        "Reproducibility checklist:",
        "",
        md_table(i["checklist"]),
        "",
    ]
    return "\n".join(lines)


def build_markdown_reduced(results: dict) -> str:
    a_dir = APP_DIR / "AppendixA"
    a_tables = a_dir / "appendix_A_generated_tables.md"
    a_extra = a_tables.read_text(encoding="utf-8") if a_tables.exists() else ""

    b = results["B"]
    c = results["C"]
    d = results["D"]
    e = results["E"]

    b_corr = short_float_frame(b["corr"], 3)
    b_catalog = b["catalog"].copy()
    b_values = short_float_frame(b["value_summary"], 4)
    c_config = short_float_frame(c["config"], 4)
    d_week = short_float_frame(d["week_metrics"], 3)
    d_random = short_float_frame(d["random_compact"], 3)

    code_map_display = e["code_map"].copy()
    if "path" in code_map_display.columns:
        code_map_display["path"] = code_map_display["path"].str.replace(
            "week_based/Final/70pct_soh_eol_threshold",
            "week_based/Final/[70% SOH EOL-threshold results]",
            regex=False,
        )
        code_map_display["path"] = code_map_display["path"].str.replace(
            "random_w5_70pct_soh_eol_threshold_10seeds_legacy400",
            "random_w5_[70% SOH EOL threshold]_10seeds_legacy400",
            regex=False,
        )

    lines = [
        "# Thesis Appendix",
        "",
        "This file collects the proposed appendix material for the thesis. The generated tables and figures are stored under `Figurecaption/Appendixfile` in section-specific folders. Appendix figures are provided as PNG and PDF files.",
        "",
        "Important boundary: this appendix reports data availability, engineered features, model configuration, sensitivity checks, and reproducibility notes. It does not define a new target-cell selection metric; random target-cell selection is reported only as an empirical sensitivity result.",
        "",
        "## Appendix A: Dataset and Cell Availability",
        "",
        f"Purpose: document cell-level availability and operating conditions used by the RUL experiments under {EOL70_TERM}. This section supports the statement that week-based sensitivity is affected by changing sample availability.",
        "",
        a_extra.strip(),
        "",
        "Suggested figures:",
        "",
        "- `Figurecaption/Appendixfile/AppendixA/appendix_A_week_availability.png` / `.pdf`",
        "- `Figurecaption/Appendixfile/AppendixA/appendix_A_eol_label_availability.png` / `.pdf`",
        "",
        "## Appendix B: Engineered Feature Definitions",
        "",
        "Purpose: list the engineered early-cycle features used for RUL prediction and provide the feature-correlation context for the main w5 experiments.",
        "",
        "Notation:",
        "",
        "- Let `w0` denote the baseline diagnostic week and `wk` denote a later early-life diagnostic week. In the main experiment reported here, `k = 5`.",
        "- For any descriptor `x`, `Delta x_{wk-w0} = x_{wk} - x_{w0}`.",
        "- `dQ/dV` denotes the incremental-capacity curve.",
        "- `CV time` denotes the constant-voltage charging time.",
        "- `C_chg` denotes charge C-rate.",
        "- `DoD` denotes depth of discharge.",
        "",
        "Basic feature categories:",
        "",
        "- IC-curve features quantify early electrochemical response changes from `dQ/dV` curves in selected voltage windows.",
        "- CV-time features describe changes in constant-voltage charging behaviour.",
        "- Usage-condition features include `DoD`, `C_chg`, and their interaction term.",
        "- Capacity-response features describe early capacity-related changes through `Delta Q^1` and `Delta Q^3` descriptors.",
        "",
        "Files:",
        "",
        "- `Figurecaption/Appendixfile/AppendixB/appendix_B_feature_notation_categories.csv`",
        "- `Figurecaption/Appendixfile/AppendixB/appendix_B_feature_definitions.csv`",
        "- `Figurecaption/Appendixfile/AppendixB/appendix_B_w5_feature_correlations.csv`",
        "- `Figurecaption/Appendixfile/AppendixB/appendix_B_w5_feature_value_summary.csv`",
        "- `Figurecaption/Appendixfile/AppendixB/appendix_B_w5_correlation_heatmap.png` / `.pdf`",
        "",
        "Feature notation, category, and interpretation:",
        "",
        feature_catalog_as_numbered_text(b_catalog),
        "",
        "Current numerical values for the week-5 feature table:",
        "",
        md_table(b_values),
        "",
        "Feature-lifetime correlations for w5:",
        "",
        md_table(b_corr),
        "",
        f"Suggested caption: Supplementary feature correlation matrix for the week-5 RUL feature set under {EOL70_TERM}. The heatmap is used only to document the degree of feature redundancy and does not introduce an additional selection metric.",
        "",
        "## Appendix C: Hyperparameter Search and Final Model Configuration",
        "",
        "Purpose: document the Stage 1 source search, Stage 2 fine-tuning search, and the final selected model configuration used in the main w5 RUL experiment.",
        "",
        "Files:",
        "",
        "- `Figurecaption/Appendixfile/AppendixC/appendix_C_hpo_search_space.csv`",
        "- `Figurecaption/Appendixfile/AppendixC/appendix_C_selected_stage2_config.csv`",
        "- `Figurecaption/Appendixfile/AppendixC/appendix_C_final_model_config.csv`",
        "",
        "Final selected configuration:",
        "",
        md_table(c_config),
        "",
        "## Appendix D: Sensitivity Check Results",
        "",
        "Purpose: collect the two sensitivity checks retained in the appendix: observation-week sensitivity and random target-cell selection sensitivity. These results are empirical checks only and do not introduce a new selection metric.",
        "",
        "Files:",
        "",
        "- `Figurecaption/Appendixfile/AppendixD/appendix_D_week_sensitivity_overall_metrics.csv`",
        "- `Figurecaption/Appendixfile/AppendixD/appendix_D_week_sensitivity_metrics.png` / `.pdf`",
        "- `Figurecaption/Appendixfile/AppendixD/appendix_D_random_target_selection_seed_results.csv`",
        "- `Figurecaption/Appendixfile/AppendixD/appendix_D_random_target_selection_numeric_aggregate.csv`",
        "- `Figurecaption/Appendixfile/AppendixD/appendix_D_random_target_selection_mae_improvement.png` / `.pdf`",
        "",
        "Week-level overall metrics:",
        "",
        md_table(d_week),
        "",
        "Random target-cell selection seed-level results:",
        "",
        md_table(d_random),
        "",
        f"Suggested caption: Sensitivity checks for RUL prediction under {EOL70_TERM}. The week-based plot reports performance changes from w5 to w10, while the random-selection plot reports seed-level changes caused by changing target fine-tuning cells.",
        "",
        "## Appendix E: Code and Reproducibility Notes",
        "",
        "Purpose: provide a concise code map and reproducibility checklist for locating the inputs, generated appendix files, and experiment outputs.",
        "",
        "Files:",
        "",
        "- `Figurecaption/Appendixfile/AppendixE/appendix_E_code_map.csv`",
        "- `Figurecaption/Appendixfile/AppendixE/appendix_E_reproducibility_checklist.csv`",
        "",
        "Code map:",
        "",
        md_table(code_map_display),
        "",
        "Reproducibility checklist:",
        "",
        md_table(e["checklist"]),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    set_style()
    results = {
        "B": appendix_b(),
        "C": appendix_c_model_config(),
        "D": appendix_d_sensitivity(),
        "E": appendix_e_reproducibility(),
    }
    APP_MD.write_text(build_markdown_reduced(results), encoding="utf-8")
    print(f"Wrote appendix markdown: {APP_MD}")
    for key, res in results.items():
        print(f"Appendix {key}: {len(res['files'])} files")


if __name__ == "__main__":
    main()
