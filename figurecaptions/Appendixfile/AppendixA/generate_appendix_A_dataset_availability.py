#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Appendix A dataset and cell-availability tables/figures.

The visual style follows the PulseBat-combo thesis figures: compact dimensions,
Arial/DejaVu sans fonts, 600 dpi raster export, PDF fonttype 42, thin axes, and a
restrained multi-color palette.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent

VALID_CELLS_CSV = ROOT / "Valid_cells.csv"
GROUP_CONDI_CSV = ROOT / "Groupcondi.csv"
EOL_AVAILABILITY_CSV = ROOT / "Data" / "Processing_Data" / "Lifetime_prediction" / "ivas_lifetime_eol_availability.csv"
MASTER_MULTI_WEEK_CSV = ROOT / "Data" / "Processing_Data" / "Lifetime_prediction" / "ivas_lifetime_10features_multiweek_per_cell.csv"
FINAL_FEATURE_TABLE_CSV = ROOT / "week_based" / "Final" / "EOL70" / "features" / "feature_table_all_cells_multiweek_EOL70.csv"
WEEK_AVAILABILITY_CSV = ROOT / "week_based" / "Final" / "EOL70" / "features" / "week_availability_summary_EOL70.csv"
EOL70_TERM = "the end-of-life (EOL) threshold defined at 70% state of health (SOH)"
EOL70_SHORT = "70% SOH EOL threshold"

DPI = 600
CM = 1 / 2.54

COLORS = {
    "usable": "#6BB7B2",
    "status": "#73B66B",
    "missing": "#D2D6DA",
    "usable_highlight": "#F28E2B",
    "missing_highlight": "#F4C8A6",
    "label": "#F28E2B",
    "label_other": "#8C6BB1",
    "label_highlight": "#D95F5F",
    "label_missing": "#E6E1EF",
    "label_missing_highlight": "#F4C8C8",
    "edge": "#111111",
    "grid": "#E8E8E8",
    "total": "#7F7F7F",
    "trend": "#222222",
}


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.75,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.unicode_minus": False,
    })


def group_from_cell(cell: str) -> int | float:
    match = re.search(r"G(\d+)C\d+", str(cell))
    return int(match.group(1)) if match else np.nan


def cell_idx_from_cell(cell: str) -> int | float:
    match = re.search(r"G\d+C(\d+)", str(cell))
    return int(match.group(1)) if match else np.nan


def sort_cells(df: pd.DataFrame, cell_col: str = "cell") -> pd.DataFrame:
    out = df.copy()
    out["_group_sort"] = out[cell_col].map(group_from_cell)
    out["_cell_sort"] = out[cell_col].map(cell_idx_from_cell)
    out = out.sort_values(["_group_sort", "_cell_sort", cell_col], kind="mergesort")
    return out.drop(columns=["_group_sort", "_cell_sort"])


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = pd.read_csv(VALID_CELLS_CSV)
    group_cond = pd.read_csv(GROUP_CONDI_CSV)
    eol_avail = pd.read_csv(EOL_AVAILABILITY_CSV)
    master = pd.read_csv(MASTER_MULTI_WEEK_CSV)
    final = pd.read_csv(FINAL_FEATURE_TABLE_CSV)
    week_avail = pd.read_csv(WEEK_AVAILABILITY_CSV)
    return valid, group_cond, eol_avail, master, final, week_avail


def build_operating_conditions(group_cond: pd.DataFrame, final: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    out = group_cond.copy()
    out["Mean DoD (%)"] = out["Mean DoD"].astype(str).str.replace("%", "", regex=False).astype(float)
    valid_counts = valid.assign(Group=valid["Cell"].map(group_from_cell)).groupby("Group").size().rename("valid_cells_in_list")
    final_counts = final.groupby("group_num").size().rename("final_70pct_soh_eol_feature_cells")
    out = out.merge(valid_counts, on="Group", how="left")
    out = out.merge(final_counts, left_on="Group", right_index=True, how="left")
    out["valid_cells_in_list"] = out["valid_cells_in_list"].fillna(0).astype(int)
    out["final_70pct_soh_eol_feature_cells"] = out["final_70pct_soh_eol_feature_cells"].fillna(0).astype(int)
    cols = [
        "Group",
        "Charging C-rate",
        "Discharging C-rate",
        "Mean DoD",
        "Mean DoD (%)",
        "Mean Lifetime [weeks]",
        "valid_cells_in_list",
        "final_70pct_soh_eol_feature_cells",
    ]
    return out[cols]


def build_cell_availability(valid: pd.DataFrame, master: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    base = valid.rename(columns={"Cell": "cell"}).copy()
    base["group_num"] = base["cell"].map(group_from_cell)
    base["cell_idx"] = base["cell"].map(cell_idx_from_cell)

    master_cols = [
        "cell",
        "release",
        "lifetime_weeks_EOL70",
        "feature_status_w3",
        "feature_status_w5",
        "feature_status_w10",
        "feature_status_w15",
    ]
    available_master_cols = [c for c in master_cols if c in master.columns]
    out = base.merge(master[available_master_cols], on="cell", how="left")

    final_cols = [
        "cell",
        "split",
        "split_label",
        "feature_status_w3",
        "feature_status_w5",
        "feature_status_w6",
        "feature_status_w7",
        "feature_status_w8",
        "feature_status_w9",
        "feature_status_w10",
        "feature_status_w15",
    ]
    final_avail = final[[c for c in final_cols if c in final.columns]].copy()
    final_avail = final_avail.rename(columns={c: f"final_{c}" for c in final_avail.columns if c != "cell"})
    out = out.merge(final_avail, on="cell", how="left")

    out["label_available_at_70pct_soh_eol_threshold"] = out["lifetime_weeks_EOL70"].notna()
    out["in_final_70pct_soh_eol_feature_table"] = out["final_split"].notna()

    def reason(row: pd.Series) -> str:
        if bool(row["in_final_70pct_soh_eol_feature_table"]):
            return "retained_final_70pct_soh_eol_feature_table"
        if not bool(row["label_available_at_70pct_soh_eol_threshold"]):
            status_text = " ".join(
                str(row.get(c, ""))
                for c in ["feature_status_w3", "feature_status_w5", "feature_status_w10", "feature_status_w15"]
            )
            if "missing" in status_text:
                return "missing_70pct_soh_eol_label_and_week_features"
            return "missing_70pct_soh_eol_label"
        return "not_retained_final_feature_table_reason_not_encoded"

    out["availability_status"] = out.apply(reason, axis=1)
    return sort_cells(out)


def build_exclusion_summary(cell_availability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_valid = len(cell_availability)
    for status, sub in cell_availability.groupby("availability_status", sort=False):
        rows.append({
            "availability_status": status,
            "cell_count": int(len(sub)),
            "fraction_of_valid_cells": len(sub) / total_valid if total_valid else np.nan,
            "example_cells": ", ".join(sub["cell"].head(8).tolist()),
        })
    return pd.DataFrame(rows)


def build_week_availability(week_avail: pd.DataFrame) -> pd.DataFrame:
    out = week_avail.copy()
    out["week_label"] = out["week"].map(lambda v: f"w{int(v)}")
    out["usable_fraction_of_total"] = out["usable_non_nan_cells"] / out["total_cells"]
    out["status_ok_fraction_of_total"] = out["status_ok_cells"] / out["total_cells"]
    out["feature_or_nan_unusable_cells"] = out["total_cells"] - out["usable_non_nan_cells"]
    return out[
        [
            "week",
            "week_label",
            "usable_non_nan_cells",
            "status_ok_cells",
            "missing_or_nan_cells",
            "feature_or_nan_unusable_cells",
            "total_cells",
            "usable_fraction_of_total",
            "status_ok_fraction_of_total",
        ]
    ]


def build_dataset_summary(
    valid: pd.DataFrame,
    group_cond: pd.DataFrame,
    eol_avail: pd.DataFrame,
    final: pd.DataFrame,
    cell_availability: pd.DataFrame,
) -> pd.DataFrame:
    eol70 = eol_avail.loc[eol_avail["EOL_percent"] == 70].iloc[0]
    rows = [
        {"item": f"Valid cells listed before {EOL70_SHORT} feature-table construction", "value": int(len(valid)), "source": "Valid_cells.csv"},
        {"item": "Operating-condition groups", "value": int(group_cond["Group"].nunique()), "source": "Groupcondi.csv"},
        {"item": f"Cells with available lifetime label under {EOL70_TERM}", "value": int(eol70["available_cells"]), "source": "ivas_lifetime_eol_availability.csv"},
        {"item": f"Cells missing lifetime label under {EOL70_TERM}", "value": int(eol70["missing_cells"]), "source": "ivas_lifetime_eol_availability.csv"},
        {"item": f"Cells retained in final multi-week feature table for {EOL70_SHORT}", "value": int(len(final)), "source": f"final feature table for {EOL70_SHORT}"},
        {"item": f"Groups retained in final multi-week feature table for {EOL70_SHORT}", "value": int(final["group_num"].nunique()), "source": f"final feature table for {EOL70_SHORT}"},
        {
            "item": f"Valid-list cells not retained in final feature table for {EOL70_SHORT}",
            "value": int((~cell_availability["in_final_70pct_soh_eol_feature_table"]).sum()),
            "source": "comparison between Valid_cells.csv and final feature table",
        },
    ]
    return pd.DataFrame(rows)


def save_tables() -> dict[str, pd.DataFrame]:
    valid, group_cond, eol_avail, master, final, week_avail = read_inputs()

    operating = build_operating_conditions(group_cond, final, valid)
    cell_availability = build_cell_availability(valid, master, final)
    cell_availability = cell_availability.rename(
        columns={"lifetime_weeks_EOL70": "lifetime_weeks_at_70pct_soh_eol_threshold"}
    )
    exclusion_summary = build_exclusion_summary(cell_availability)
    week_summary = build_week_availability(week_avail)
    dataset_summary = build_dataset_summary(valid, group_cond, eol_avail, final, cell_availability)
    eol_avail_out = eol_avail.copy()
    if "lifetime_column" in eol_avail_out.columns:
        eol_avail_out["lifetime_column"] = eol_avail_out["lifetime_column"].astype(str).str.replace(
            r"lifetime_weeks_EOL(\d+)",
            r"lifetime_weeks_at_\1pct_soh_eol_threshold",
            regex=True,
        )

    tables = {
        "appendix_A_dataset_summary": dataset_summary,
        "appendix_A_operating_conditions": operating,
        "appendix_A_valid_cells_70pct_soh_eol_threshold": cell_availability,
        "appendix_A_week_availability": week_summary,
        "appendix_A_excluded_cells_summary": exclusion_summary,
        "appendix_A_eol_label_availability": eol_avail_out,
    }
    for name, df in tables.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    return tables


def label_bars(ax, xs: np.ndarray, ys: np.ndarray, dy: float) -> None:
    for x, y in zip(xs, ys):
        ax.text(x, y + dy, f"{int(round(y))}", ha="center", va="bottom", fontsize=6)


def plot_week_availability(week_summary: pd.DataFrame) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(12.0 * CM, 5.5 * CM), dpi=DPI)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.205, top=0.840)

    labels = week_summary["week_label"].tolist()
    x = np.arange(len(labels), dtype=float)
    usable = week_summary["usable_non_nan_cells"].to_numpy(dtype=float)
    missing = week_summary["feature_or_nan_unusable_cells"].to_numpy(dtype=float)
    total = week_summary["total_cells"].to_numpy(dtype=float)
    highlight_idx = labels.index("w5") if "w5" in labels else None
    usable_colors = [COLORS["usable_highlight"] if i == highlight_idx else COLORS["usable"] for i in range(len(labels))]
    missing_colors = [COLORS["missing_highlight"] if i == highlight_idx else COLORS["missing"] for i in range(len(labels))]

    ax.bar(
        x,
        usable,
        width=0.58,
        color=usable_colors,
        edgecolor=COLORS["edge"],
        linewidth=0.45,
        label="Usable cells",
        zorder=3,
    )
    ax.bar(
        x,
        missing,
        bottom=usable,
        width=0.58,
        color=missing_colors,
        edgecolor=COLORS["edge"],
        linewidth=0.35,
        label="Unavailable cells",
        zorder=3,
    )
    ax.plot(
        x,
        usable,
        color=COLORS["trend"],
        lw=0.9,
        marker="o",
        ms=2.9,
        markerfacecolor="white",
        markeredgecolor=COLORS["trend"],
        label="Usable-cell trend",
        zorder=5,
    )
    label_bars(ax, x, usable, max(total) * 0.018)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Feature week", labelpad=2)
    ax.set_ylabel("Number of cells", labelpad=2)
    y_top = int(max(total))
    ax.set_ylim(0, y_top * 1.02)
    ax.set_yticks([0, 50, 100, 150, 200, y_top])
    ax.grid(True, axis="y", color=COLORS["grid"], lw=0.35, zorder=1)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        ncol=3,
        frameon=False,
        handlelength=1.25,
        columnspacing=1.2,
        borderpad=0.2,
        labelspacing=0.25,
    )
    ax.tick_params(length=2.0, width=0.6, pad=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)

    fig.savefig(OUT_DIR / "appendix_A_week_availability.png", dpi=DPI)
    fig.savefig(OUT_DIR / "appendix_A_week_availability.pdf")
    plt.close(fig)


def plot_eol_availability(eol_avail: pd.DataFrame) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(12.0 * CM, 5.5 * CM), dpi=DPI)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.205, top=0.840)

    x = np.arange(len(eol_avail), dtype=float)
    labels = [f"{int(v)}" for v in eol_avail["EOL_percent"]]
    available = eol_avail["available_cells"].to_numpy(dtype=float)
    missing = eol_avail["missing_cells"].to_numpy(dtype=float)
    highlight_idx = int(np.where(eol_avail["EOL_percent"].to_numpy() == 70)[0][0])
    available_colors = [
        COLORS["label_highlight"] if i == highlight_idx else COLORS["label_other"] for i in range(len(eol_avail))
    ]
    missing_colors = [
        COLORS["label_missing_highlight"] if i == highlight_idx else COLORS["label_missing"] for i in range(len(eol_avail))
    ]

    ax.bar(
        x,
        available,
        width=0.58,
        color=available_colors,
        edgecolor=COLORS["edge"],
        linewidth=0.45,
        label="Label available",
        zorder=3,
    )
    ax.bar(
        x,
        missing,
        bottom=available,
        width=0.58,
        color=missing_colors,
        edgecolor=COLORS["edge"],
        linewidth=0.35,
        label="Label missing",
        zorder=3,
    )
    ax.plot(
        x,
        available,
        color=COLORS["trend"],
        lw=0.9,
        marker="o",
        ms=2.9,
        markerfacecolor="white",
        markeredgecolor=COLORS["trend"],
        label="Available-label trend",
        zorder=5,
    )

    label_bars(ax, x, available, max(available + missing) * 0.018)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("End-of-life threshold (% state of health)", labelpad=2)
    ax.set_ylabel("Number of cells", labelpad=2)
    y_top = int(max(available + missing))
    ax.set_ylim(0, y_top * 1.02)
    ax.set_yticks([0, 50, 100, 150, 200, y_top])
    ax.grid(True, axis="y", color=COLORS["grid"], lw=0.35, zorder=1)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.985),
        ncol=3,
        frameon=False,
        handlelength=1.25,
        columnspacing=1.2,
        borderpad=0.2,
        labelspacing=0.25,
    )
    ax.tick_params(length=2.0, width=0.6, pad=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)

    fig.savefig(OUT_DIR / "appendix_A_eol_label_availability.png", dpi=DPI)
    fig.savefig(OUT_DIR / "appendix_A_eol_label_availability.pdf")
    plt.close(fig)


def write_markdown_fragment(tables: dict[str, pd.DataFrame]) -> None:
    summary = tables["appendix_A_dataset_summary"]
    week = tables["appendix_A_week_availability"]
    excluded = tables["appendix_A_excluded_cells_summary"]
    eol = tables["appendix_A_eol_label_availability"]
    eol70 = eol.loc[eol["EOL_percent"] == 70].iloc[0]

    lines: list[str] = []
    lines.append("# Appendix A Generated Tables")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append("")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    lines.append(f"## Label Availability at {EOL70_SHORT}")
    lines.append("")
    lines.append(
        f"For {EOL70_TERM}, {int(eol70['available_cells'])} of "
        f"{int(eol70['total_cells'])} cells have an available lifetime label; "
        f"{int(eol70['missing_cells'])} cells are missing this label."
    )
    lines.append("")
    lines.append("## Week-Based Feature Availability")
    lines.append("")
    lines.append(
        week[
            [
                "week_label",
                "usable_non_nan_cells",
                "status_ok_cells",
                "feature_or_nan_unusable_cells",
                "total_cells",
            ]
        ].to_markdown(index=False)
    )
    lines.append("")
    lines.append("## Cell-Retention Summary")
    lines.append("")
    lines.append(excluded.to_markdown(index=False))
    lines.append("")

    (OUT_DIR / "appendix_A_generated_tables.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = save_tables()
    plot_week_availability(tables["appendix_A_week_availability"])
    plot_eol_availability(tables["appendix_A_eol_label_availability"])
    write_markdown_fragment(tables)

    print("[OK] Appendix A tables and figures generated in:")
    print(f"     {OUT_DIR}")


if __name__ == "__main__":
    main()
