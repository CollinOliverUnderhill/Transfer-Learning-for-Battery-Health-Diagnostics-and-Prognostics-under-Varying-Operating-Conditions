#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 43  –  IVAS RUL prediction (EOL70, Week 5)
    Fig 43-1: Benchmark parity plot (source+target-finetune train, test)
    Fig 43-2: Benchmark error distribution (histogram + boxplot)
    Fig 43-4: Transfer-learning parity plot (train, finetune, test)
    Fig 43-5: Transfer-learning error distribution

Tables 43-3, 43-6, 43-7 are printed to stdout.
All data from: stage3_final_rerun_400/{benchmark, transfer_model}
Style: Figure_later/42 convention
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
IVAS    = Path(r"E:\Datasets\IVAS")
OUT     = IVAS / "Figure" / "Figure_later" / "43"
EXP     = IVAS / "week_based/Final/EOL70/3step/outputs_400/BasicModel/stage3_final_rerun_400"
BM      = EXP / "benchmark"
TM      = EXP / "transfer_model"
SEED    = IVAS / "week_based/Final/EOL70/3step/outputs_400/random_w5_EOL70_10seeds_legacy400"
WEEK    = IVAS / "week_based/Final/EOL70/3step/outputs_400/protocol_w6_10_from_stage3_final_rerun_400_legacy400"

# ── Style (Figure 42) ──────────────────────────────────────────────────────
CM  = 1 / 2.54
DPI = 600
FIG_PARITY = (12.74 * CM, 4.00 * CM)
FIG_ERR    = (12.74 * CM, 4.00 * CM)
FIG_PARITY_COMBO = (12.74 * CM, 4.00 * CM)
FIG_ERR_COMBO = (13.25 * CM, 7.85 * CM)

TRAIN_COLOR = "#6BB7B2"
FINETUNE_COLOR = "#73B66B"
TEST_COLOR = "#F28E2B"
BLACK = "#111111"
MID_GREY = "#B8B8B8"

POINT_SIZE = 13.0
POINT_EDGE_COLOR = "#FFFFFF"
POINT_EDGE_LW = 0.55
AXIS_LW = 0.75
TICK_LW = 0.60
TICK_LENGTH = 2.0
REFERENCE_LW = 1.15

mpl.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.linewidth": AXIS_LW,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

def style_axes(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(AXIS_LW)
    ax.tick_params(axis="both", which="major", width=TICK_LW, length=TICK_LENGTH, pad=1.2)
    ax.tick_params(axis="both", which="minor", width=0.5, length=1.4, pad=1.2)

# ── Helpers ─────────────────────────────────────────────────────────────────
def rcsv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def load_pred(p):
    rows = rcsv(p)
    return np.array([float(r["y_true"]) for r in rows]), \
           np.array([float(r["y_pred"]) for r in rows])

def load_met(p):
    return {k: float(v) for k, v in rcsv(p)[0].items() if k != "split"}

def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=DPI)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)
    print(f"  [saved] {stem}.png / .pdf")

# ── Fig 43-1 / 43-4: Parity plot ───────────────────────────────────────────
def plot_parity(model_dir, out_stem, has_finetune=False):
    """Single-panel parity plot (12.74cm x 4.0cm, left half used)."""
    y_tr, yp_tr = load_pred(model_dir / "predictions_source_all_train.csv")
    y_te, yp_te = load_pred(model_dir / "predictions_test.csv")
    met = load_met(model_dir / "test_overall_metrics.csv")

    all_v = np.concatenate([y_tr, yp_tr, y_te, yp_te])
    if has_finetune:
        y_ft, yp_ft = load_pred(model_dir / "predictions_target_finetune.csv")
        all_v = np.concatenate([all_v, y_ft, yp_ft])

    lo = 0
    hi = float(np.max(all_v)) * 1.08
    tick_max = int(np.ceil(hi / 10)) * 10
    ticks = np.arange(0, tick_max + 1, 10)

    fig, ax = plt.subplots(figsize=(6.37 * CM, 6.37 * CM))

    ax.plot([lo, hi], [lo, hi], color=BLACK, lw=REFERENCE_LW,
            ls=(0, (3.0, 2.0)), zorder=1)

    # Train (source + target finetune combined for benchmark, source-only for TL)
    ax.scatter(y_tr, yp_tr, s=POINT_SIZE, facecolor=TRAIN_COLOR,
               edgecolor=POINT_EDGE_COLOR, linewidth=POINT_EDGE_LW,
               alpha=0.58, zorder=2, rasterized=True, label="Train")

    if has_finetune:
        ax.scatter(y_ft, yp_ft, s=POINT_SIZE, facecolor=FINETUNE_COLOR,
                   edgecolor=POINT_EDGE_COLOR, linewidth=POINT_EDGE_LW,
                   alpha=0.82, zorder=3, marker="D", rasterized=True, label="Fine-tune")

    ax.scatter(y_te, yp_te, s=POINT_SIZE, facecolor=TEST_COLOR,
               edgecolor=POINT_EDGE_COLOR, linewidth=POINT_EDGE_LW,
               alpha=0.88, zorder=4, rasterized=True, label="Test")

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlabel("True RUL (weeks)")
    ax.set_ylabel("Predicted RUL (weeks)")

    text = (f"MAE={met['mae']:.2f}\n"
            f"RMSE={met['rmse']:.2f}\n"
            f"R$^2$={met['r2']:.3f}")
    ax.text(0.05, 0.95, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=6.5, linespacing=1.15)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.25,
              borderpad=0.1, markerscale=1.0)
    style_axes(ax)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.16, top=0.98)
    save(fig, out_stem)

# ── Fig 43-2 / 43-5: Error distribution ────────────────────────────────────
def plot_error(model_dir, out_stem):
    y_tr, yp_tr = load_pred(model_dir / "predictions_source_all_train.csv")
    y_te, yp_te = load_pred(model_dir / "predictions_test.csv")
    err_tr = yp_tr - y_tr
    err_te = yp_te - y_te
    datasets = [("Train", err_tr, TRAIN_COLOR), ("Test", err_te, TEST_COLOR)]

    fig, axes = plt.subplots(1, 2, figsize=FIG_ERR)

    # (a) histogram
    ax = axes[0]
    for label, err, clr in datasets:
        ax.hist(err, bins=25, alpha=0.66, color=clr, edgecolor=BLACK,
                linewidth=0.45, label=label, density=True, zorder=2)
    ax.axvline(0, ls=(0, (3.0, 2.0)), lw=0.80, color=MID_GREY, zorder=1)
    ax.set_xlabel("Prediction Error (weeks)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, handletextpad=0.25, borderpad=0.1)
    ax.text(0.04, 0.94, "a", transform=ax.transAxes, ha="left", va="top",
            fontsize=7, fontweight="bold")
    style_axes(ax)

    # (b) boxplot
    ax = axes[1]
    bps = ax.boxplot([np.abs(err_tr), np.abs(err_te)],
                     tick_labels=["Train", "Test"], patch_artist=True,
                     widths=0.45, showfliers=True,
                     flierprops=dict(marker="o", markersize=2.5, alpha=0.45,
                                     markerfacecolor="#D9D9D9",
                                     markeredgecolor="#777777",
                                     markeredgewidth=0.40),
                     medianprops=dict(color=BLACK, lw=0.85),
                     boxprops=dict(edgecolor=BLACK, linewidth=0.80),
                     whiskerprops=dict(color=BLACK, linewidth=0.70),
                     capprops=dict(color=BLACK, linewidth=0.70))
    for patch, clr in zip(bps["boxes"], [TRAIN_COLOR, TEST_COLOR]):
        patch.set_facecolor(clr)
        patch.set_alpha(0.66)
        patch.set_edgecolor(BLACK)
    ax.set_ylabel("|Error| (weeks)")
    ax.text(0.04, 0.94, "b", transform=ax.transAxes, ha="left", va="top",
            fontsize=7, fontweight="bold")
    style_axes(ax)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.25, top=0.95, wspace=0.35)
    save(fig, out_stem)

def _parity_data(model_dir, has_finetune=False):
    y_tr, yp_tr = load_pred(model_dir / "predictions_source_all_train.csv")
    y_te, yp_te = load_pred(model_dir / "predictions_test.csv")
    met = load_met(model_dir / "test_overall_metrics.csv")
    data = {"train": (y_tr, yp_tr), "test": (y_te, yp_te), "metrics": met}
    all_v = [y_tr, yp_tr, y_te, yp_te]
    if has_finetune:
        y_ft, yp_ft = load_pred(model_dir / "predictions_target_finetune.csv")
        data["finetune"] = (y_ft, yp_ft)
        all_v.extend([y_ft, yp_ft])
    data["all_values"] = np.concatenate(all_v)
    return data

def _draw_parity_panel(ax, data, panel_label, model_label, show_ylabel=True):
    lo = 0
    hi = float(np.max(data["all_values"])) * 1.08
    tick_max = int(np.ceil(hi / 15)) * 15
    ticks = np.arange(0, tick_max + 1, 15)

    ax.plot([lo, hi], [lo, hi], color="#404040", lw=0.75, zorder=1)
    y_tr, yp_tr = data["train"]
    ax.scatter(y_tr, yp_tr, s=5.8, facecolor=TRAIN_COLOR,
               edgecolor="white", linewidth=0.12,
               alpha=0.34, zorder=2, rasterized=True, label="Train")
    if "finetune" in data:
        y_ft, yp_ft = data["finetune"]
        ax.scatter(y_ft, yp_ft, s=7.5, facecolor=FINETUNE_COLOR,
                   edgecolor="white", linewidth=0.18,
                   alpha=0.78, zorder=3, marker="D", rasterized=True, label="Fine-tune")
    y_te, yp_te = data["test"]
    ax.scatter(y_te, yp_te, s=7.5, facecolor=TEST_COLOR,
               edgecolor="white", linewidth=0.18,
               alpha=0.88, zorder=4, rasterized=True, label="Test")

    ax.set_xlim(lo, tick_max)
    ax.set_ylim(lo, tick_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Predicted RUL (weeks)")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
        ax.set_xticklabels([""] + [f"{int(t)}" for t in ticks[1:]])

    met = data["metrics"]
    text = (f"MAE={met['mae']:.2f}\n"
            f"RMSE={met['rmse']:.2f}\n"
            f"MAPE={met['mape_percent']:.2f}%")
    ax.text(0.05, 0.95, panel_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=8, fontweight="bold")
    ax.text(0.14, 0.95, model_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=7)
    ax.text(0.05, 0.82, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=6, linespacing=1.15)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.00, 0.02),
        frameon=False,
        handletextpad=0.25,
        borderpad=0.1,
        markerscale=1.25,
    )
    style_axes(ax)

def plot_parity_comparison(out_stem="Fig43_1_4_parity_comparison"):
    bm_data = _parity_data(BM, has_finetune=False)
    tl_data = _parity_data(TM, has_finetune=True)
    global_hi = float(np.max(np.concatenate([bm_data["all_values"], tl_data["all_values"]]))) * 1.08
    for data in (bm_data, tl_data):
        data["all_values"] = np.append(data["all_values"], global_hi)

    fig, axes = plt.subplots(1, 2, figsize=FIG_PARITY_COMBO)
    _draw_parity_panel(axes[0], bm_data, "a", "Benchmark", show_ylabel=True)
    _draw_parity_panel(axes[1], tl_data, "b", "TL", show_ylabel=False)

    fig_w_cm, fig_h_cm = 12.74, 4.00
    ax_h = 0.76
    ax_w = ax_h * (fig_h_cm / fig_w_cm)
    ax_bottom = 0.20
    ax_left = 0.19
    ax_gap = 0.012
    axes[0].set_position([ax_left, ax_bottom, ax_w, ax_h])
    axes[1].set_position([ax_left + ax_w + ax_gap, ax_bottom, ax_w, ax_h])
    fig.text(
        ax_left + ax_w + ax_gap / 2,
        0.055,
        "True RUL (weeks)",
        ha="center",
        va="center",
        fontsize=8,
    )
    save(fig, out_stem)

def _error_data(model_dir):
    y_tr, yp_tr = load_pred(model_dir / "predictions_source_all_train.csv")
    y_te, yp_te = load_pred(model_dir / "predictions_test.csv")
    return yp_tr - y_tr, yp_te - y_te

def _draw_error_row(axes, err_tr, err_te, row_title, panel_labels, bins, hist_ymax, box_ymax, show_xlabels):
    datasets = [("Train", err_tr, TRAIN_COLOR), ("Test", err_te, TEST_COLOR)]

    ax = axes[0]
    for label, err, clr in datasets:
        ax.hist(err, bins=bins, alpha=0.66, color=clr, edgecolor=BLACK,
                linewidth=0.45, label=label, density=True, zorder=2)
    ax.axvline(0, ls=(0, (3.0, 2.0)), lw=0.80, color=MID_GREY, zorder=1)
    ax.set_xlim(float(bins[0]), float(bins[-1]))
    ax.set_ylim(0.0, hist_ymax)
    if show_xlabels:
        ax.set_xlabel("Prediction Error (weeks)")
    else:
        ax.tick_params(labelbottom=False)
    ax.set_ylabel("Density")
    ax.legend(frameon=False, handletextpad=0.25, borderpad=0.1)
    ax.text(0.04, 0.94, panel_labels[0], transform=ax.transAxes, ha="left", va="top",
            fontsize=7, fontweight="bold")
    style_axes(ax)

    ax = axes[1]
    bps = ax.boxplot([np.abs(err_tr), np.abs(err_te)],
                     tick_labels=["Train", "Test"], patch_artist=True,
                     widths=0.45, showfliers=True,
                     flierprops=dict(marker="o", markersize=2.3, alpha=0.45,
                                     markerfacecolor="#D9D9D9",
                                     markeredgecolor="#777777",
                                     markeredgewidth=0.40),
                     medianprops=dict(color=BLACK, lw=0.85),
                     boxprops=dict(edgecolor=BLACK, linewidth=0.80),
                     whiskerprops=dict(color=BLACK, linewidth=0.70),
                     capprops=dict(color=BLACK, linewidth=0.70))
    for patch, clr in zip(bps["boxes"], [TRAIN_COLOR, TEST_COLOR]):
        patch.set_facecolor(clr)
        patch.set_alpha(0.66)
        patch.set_edgecolor(BLACK)
    ax.set_ylim(0.0, box_ymax)
    if not show_xlabels:
        ax.tick_params(labelbottom=False)
    ax.set_ylabel("|Error| (weeks)")
    ax.text(0.04, 0.94, panel_labels[1], transform=ax.transAxes, ha="left", va="top",
            fontsize=7, fontweight="bold")
    style_axes(ax)

def plot_error_comparison(out_stem="Fig43_2_5_error_distribution_comparison"):
    bm_err_tr, bm_err_te = _error_data(BM)
    tl_err_tr, tl_err_te = _error_data(TM)
    all_err = np.concatenate([bm_err_tr, bm_err_te, tl_err_tr, tl_err_te])
    left = np.floor((float(np.min(all_err)) - 0.5) / 5.0) * 5.0
    right = np.ceil((float(np.max(all_err)) + 0.5) / 5.0) * 5.0
    bins = np.linspace(left, right, 31)

    hist_ymax = 0.0
    for err in (bm_err_tr, bm_err_te, tl_err_tr, tl_err_te):
        density, _ = np.histogram(err, bins=bins, density=True)
        hist_ymax = max(hist_ymax, float(np.nanmax(density)))
    hist_ymax = np.ceil(hist_ymax * 1.18 / 0.05) * 0.05
    box_ymax = np.ceil(float(np.max(np.abs(all_err))) * 1.10 / 5.0) * 5.0

    fig, axes = plt.subplots(2, 2, figsize=FIG_ERR_COMBO)
    _draw_error_row(axes[0], bm_err_tr, bm_err_te, "Benchmark", ("a", "b"),
                    bins, hist_ymax, box_ymax, show_xlabels=False)
    _draw_error_row(axes[1], tl_err_tr, tl_err_te, "Optuna+TPE transfer learning", ("c", "d"),
                    bins, hist_ymax, box_ymax, show_xlabels=True)
    fig.text(0.542, 0.995, "Benchmark", ha="center", va="top", fontsize=7)
    fig.text(0.542, 0.510, "Transfer Learning", ha="center", va="top", fontsize=7)
    fig.subplots_adjust(left=0.100, right=0.985, bottom=0.105, top=0.930,
                        wspace=0.310, hspace=0.420)
    save(fig, out_stem)

# ── Table helpers ───────────────────────────────────────────────────────────
def fmt_met(m, keys=None):
    if keys is None:
        keys = ["n_cells","mae","rmse","r2","mape_percent","smape_percent","wmape_percent","abs_err_median","abs_err_p90"]
    row = []
    for k in keys:
        v = m.get(k, 0)
        if k == "n_cells": row.append(f"{int(v)}")
        elif k == "r2":    row.append(f"{v:.3f}")
        else:              row.append(f"{v:.2f}")
    return row

def print_table(title, headers, rows):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    hdr = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(hdr)
    print("-+-".join("-"*w for w in widths))
    for r in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(r, widths)))
    print()

def improvement_rows():
    bm = load_met(BM / "test_overall_metrics.csv")
    tl = load_met(TM / "test_overall_metrics.csv")
    keys = [
        ("MAE", "mae"),
        ("RMSE", "rmse"),
        ("MAPE%", "mape_percent"),
        ("sMAPE%", "smape_percent"),
        ("wMAPE%", "wmape_percent"),
        ("MedAE", "abs_err_median"),
        ("P90", "abs_err_p90"),
    ]
    rows = []
    for label, key in keys:
        bm_v = bm[key]
        tl_v = tl[key]
        improve = (bm_v - tl_v) / bm_v * 100.0
        rows.append([label, f"{bm_v:.2f}", f"{tl_v:.2f}", f"{improve:+.1f}%"])
    return rows

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  Figure 43 - IVAS RUL (EOL70, W5)")
    print("  Data: stage3_final_rerun_400/{benchmark, transfer_model}")
    print("=" * 60)

    print("\n[1/3] Combined benchmark/transfer parity plot ...")
    plot_parity_comparison()

    print("[2/3] Combined benchmark/transfer error distribution ...")
    plot_error_comparison()

    print("[3/3] Test-set improvement table (lower is better; R^2 omitted) ...")
    print_table(
        "Figure 43 improvement: Benchmark vs Optuna+TPE based transfer learning",
        ["Metric", "Benchmark", "Optuna+TPE TL", "Improvement"],
        improvement_rows(),
    )

    print("=" * 60)
    print("  [DONE] Combined figures -> ", OUT)
    print("=" * 60)
    return

    # ── 1. Benchmark parity ──
    print("\n[1/7] Benchmark parity plot ...")
    plot_parity(BM, "Fig43_1_benchmark_parity", has_finetune=False)

    # ── 2. Benchmark error dist ──
    print("[2/7] Benchmark error distribution ...")
    plot_error(BM, "Fig43_2_benchmark_error_dist")

    # ── 3. Benchmark table (printed) ──
    print("[3/7] Benchmark performance table ...")
    hdrs = ["Split","n","MAE","RMSE","R^2","MAPE%","sMAPE%","wMAPE%","MedAE","P90"]
    rows3 = []
    for label, csv_name in [("Source Train", "source_all_train_overall_metrics.csv"),
                            ("Source Val",   "source_val_overall_metrics.csv"),
                            ("Test",         "test_overall_metrics.csv")]:
        m = load_met(BM / csv_name)
        rows3.append([label] + fmt_met(m))
    print_table("Table 43-3: Benchmark Performance (EOL70, W5)", hdrs, rows3)

    # ── 4. Transfer parity ──
    print("[4/7] Transfer-learning parity plot ...")
    plot_parity(TM, "Fig43_4_transfer_parity", has_finetune=True)

    # ── 5. Transfer error dist ──
    print("[5/7] Transfer-learning error distribution ...")
    plot_error(TM, "Fig43_5_transfer_error_dist")

    # ── 6. Transfer table (printed) ──
    print("[6/7] Transfer-learning performance table ...")
    rows6 = []
    for label, csv_name in [("Source Train",   "source_all_train_overall_metrics.csv"),
                            ("Source Val",     "source_val_overall_metrics.csv"),
                            ("Target FT",     "target_finetune_overall_metrics.csv"),
                            ("Target FT Val", "target_finetune_val_overall_metrics.csv"),
                            ("Test",          "test_overall_metrics.csv")]:
        m = load_met(TM / csv_name)
        rows6.append([label] + fmt_met(m))
    print_table("Table 43-6: Transfer Learning Performance (EOL70, W5)", hdrs, rows6)

    # ── 7a. Week sensitivity table ──
    print("[7/8] Week sensitivity table ...")
    report = load_json(EXP / "stage3_final_report.json")
    week_rows = rcsv(WEEK / "protocol_w6_10_from_stage3_final_rerun_400_legacy400_benchmark_transfer_summary_all_splits.csv")

    w_hdrs = ["Week","BM MAE","BM MAPE%","BM R^2","TL MAE","TL MAPE%","TL R^2","dMAE%","dMAPE%"]
    w_rows = []
    bm5 = report["benchmark_test_overall"]; tl5 = report["transfer_test_overall"]
    d1 = (1-tl5["mae"]/bm5["mae"])*100; d2 = (1-tl5["mape_percent"]/bm5["mape_percent"])*100
    w_rows.append(["5 (ref)",f"{bm5['mae']:.2f}",f"{bm5['mape_percent']:.2f}",f"{bm5['r2']:.3f}",
                   f"{tl5['mae']:.2f}",f"{tl5['mape_percent']:.2f}",f"{tl5['r2']:.3f}",
                   f"{d1:+.1f}",f"{d2:+.1f}"])
    for wr in week_rows:
        w = wr["stage3_dir"].split("/")[0].replace("week","")
        di1 = float(wr["test_transfer_vs_bench_mae_improve_percent"])
        di2 = float(wr["test_transfer_vs_bench_mape_improve_percent"])
        s = lambda k: f"{float(wr[k]):.2f}" if "r2" not in k else f"{float(wr[k]):.3f}"
        w_rows.append([w, s("bench_test_mae"), s("bench_test_mape"), s("bench_test_r2"),
                       s("transfer_test_mae"), s("transfer_test_mape"), s("transfer_test_r2"),
                       f"{di1:+.1f}", f"{di2:+.1f}"])
    print_table("Table 43-7a: Week Sensitivity (EOL70, Feature Extraction Week)", w_hdrs, w_rows)

    # ── 7b. Seed sensitivity table (20 seeds, W5) ──
    print("[8/8] Seed sensitivity table ...")
    seed_all = rcsv(SEED / "random_w5_EOL70_10seeds_legacy400_benchmark_transfer_summary_all_splits.csv")
    seed_agg = {}
    for r in rcsv(SEED / "random_w5_EOL70_10seeds_legacy400_benchmark_transfer_summary_numeric_aggregate.csv"):
        seed_agg[r["metric"]] = r

    sd_hdrs = ["Seed","BM MAE","BM MAPE%","BM R^2","TL MAE","TL MAPE%","TL R^2","dMAE%","dMAPE%"]
    sd_rows = []
    for sr in seed_all:
        seed_id = sr["stage3_dir"].split("/")[0].replace("seed","")
        bm_mae = float(sr["bench_test_mae"]); bm_mape = float(sr["bench_test_mape"]); bm_r2 = float(sr["bench_test_r2"])
        tl_mae = float(sr["transfer_test_mae"]); tl_mape = float(sr["transfer_test_mape"]); tl_r2 = float(sr["transfer_test_r2"])
        di_mae = float(sr["test_transfer_vs_bench_mae_improve_percent"])
        di_mape = float(sr["test_transfer_vs_bench_mape_improve_percent"])
        sd_rows.append([seed_id, f"{bm_mae:.2f}", f"{bm_mape:.2f}", f"{bm_r2:.3f}",
                        f"{tl_mae:.2f}", f"{tl_mape:.2f}", f"{tl_r2:.3f}",
                        f"{di_mae:+.1f}", f"{di_mape:+.1f}"])
    # Mean ± std summary row
    sa = seed_agg
    sd_rows.append(["Mean±Std",
        f"{float(sa['bench_test_mae']['mean']):.2f}±{float(sa['bench_test_mae']['std']):.2f}",
        f"{float(sa['bench_test_mape']['mean']):.1f}±{float(sa['bench_test_mape']['std']):.1f}",
        f"{float(sa['bench_test_r2']['mean']):.3f}±{float(sa['bench_test_r2']['std']):.3f}",
        f"{float(sa['transfer_test_mae']['mean']):.2f}±{float(sa['transfer_test_mae']['std']):.2f}",
        f"{float(sa['transfer_test_mape']['mean']):.1f}±{float(sa['transfer_test_mape']['std']):.1f}",
        f"{float(sa['transfer_test_r2']['mean']):.3f}±{float(sa['transfer_test_r2']['std']):.3f}",
        f"{float(sa['test_transfer_vs_bench_mae_improve_percent']['mean']):+.1f}±{float(sa['test_transfer_vs_bench_mae_improve_percent']['std']):.1f}",
        f"{float(sa['test_transfer_vs_bench_mape_improve_percent']['mean']):+.1f}±{float(sa['test_transfer_vs_bench_mape_improve_percent']['std']):.1f}"])
    print_table("Table 43-7b: Seed Sensitivity (W5, 20 Seeds, Fine-tune Battery Selection)", sd_hdrs, sd_rows)

    print("=" * 60)
    print("  [DONE] Figures -> ", OUT)
    print("=" * 60)

if __name__ == "__main__":
    main()
