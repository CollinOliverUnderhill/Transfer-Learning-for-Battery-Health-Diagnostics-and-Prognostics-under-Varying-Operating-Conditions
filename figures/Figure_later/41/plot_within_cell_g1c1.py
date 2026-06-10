#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


ROOT = Path(r"E:\Datasets\IVAS")
DATA_DIR = ROOT / "week_based" / "SOHest" / "results" / "ridge_single_cell" / "seed_sweep_by_cell" / "G1C1"
OUT_DIR = ROOT / "Figure" / "Figure_later" / "41"

RAW_CSV = ROOT / "Data" / "Processing_Data" / "SOHest" / "rpt_samples_feature_soh.csv"
PRED_CSV = DATA_DIR / "predictions.csv"
SEED_CSV = DATA_DIR / "seed_sweep_metrics.csv"

CM = 1 / 2.54
FIGSIZE = (6.37 * CM, 4.00 * CM)


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 0.75,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.labelsize": 8,
        "legend.fontsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def metric_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
    }


def random_split_indices(n: int, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_end = int(math.floor(n * float(train_frac)))
    train_end = max(5, min(train_end, n - 1))
    rng = np.random.default_rng(int(seed))
    idx_all = np.arange(n, dtype=int)
    rng.shuffle(idx_all)
    return np.sort(idx_all[:train_end]), np.sort(idx_all[train_end:])


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd <= 0, 1.0, sd)
    return mu, sd


def standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=float) - mu) / sd


def ridge_fit_closed_form(x_s: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    y_mean = float(np.mean(y))
    y_c = y - y_mean
    p = x_s.shape[1]
    beta_s = np.linalg.solve(x_s.T @ x_s + alpha * np.eye(p), x_s.T @ y_c)
    return beta_s, y_mean


def ridge_predict(x_s: np.ndarray, beta_s: np.ndarray, y_mean: float) -> np.ndarray:
    return y_mean + x_s @ beta_s


def rebuild_single_run_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    raw_rows = [
        r
        for r in read_csv(RAW_CSV)
        if r.get("cell") == "G1C1" and r.get("release") == "Release 1.0"
    ]
    raw_rows = [r for r in raw_rows if r.get("feature_mean_ic", "") != "" and r.get("soh", "") != ""]
    raw_rows.sort(key=lambda r: float(r["time_week"]))

    pred_rows = read_csv(PRED_CSV)
    target_test_rpt = [int(float(r["rpt_idx"])) for r in pred_rows]
    target_test_idx = np.asarray(
        [i for i, r in enumerate(raw_rows) if int(float(r["rpt_idx"])) in set(target_test_rpt)],
        dtype=int,
    )

    matched_seed = -1
    tr_idx = te_idx = np.asarray([], dtype=int)
    for seed in range(100000):
        tr_try, te_try = random_split_indices(len(raw_rows), train_frac=0.8, seed=seed)
        if np.array_equal(te_try, target_test_idx):
            matched_seed = seed
            tr_idx, te_idx = tr_try, te_try
            break
    if matched_seed < 0:
        raise RuntimeError("Could not recover the original train/test split from predictions.csv.")

    x = np.asarray([[float(r["feature_mean_ic"])] for r in raw_rows], dtype=float)
    y = np.asarray([float(r["soh"]) for r in raw_rows], dtype=float)
    mu, sd = standardize_fit(x[tr_idx])
    x_train_s = standardize_apply(x[tr_idx], mu, sd)
    x_test_s = standardize_apply(x[te_idx], mu, sd)
    beta_s, y_mean = ridge_fit_closed_form(x_train_s, y[tr_idx], alpha=0.001)
    y_train_pred = ridge_predict(x_train_s, beta_s, y_mean)
    y_test_pred = ridge_predict(x_test_s, beta_s, y_mean)
    return y[tr_idx], y_train_pred, y[te_idx], y_test_pred, matched_seed


def style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    ax.tick_params(axis="both", which="major", width=0.6, length=2.2, pad=1.5)
    ax.tick_params(axis="both", which="minor", width=0.5, length=1.4)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=600)
    plt.close(fig)


def plot_pred_vs_true() -> dict[str, float]:
    y_train, y_pred_train, y_true, y_pred, matched_seed = rebuild_single_run_predictions()
    abs_err = np.abs(y_pred - y_true)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mape = float(np.mean(abs_err / np.maximum(np.abs(y_true), 1e-12)) * 100.0)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    test_color = "#1f7f88"
    train_color = "#e6862e"
    all_true = np.concatenate([y_train, y_true])
    all_pred = np.concatenate([y_pred_train, y_pred])
    lim_min = min(float(np.min(all_true)), float(np.min(all_pred))) - 0.025
    lim_max = max(float(np.max(all_true)), float(np.max(all_pred))) + 0.025
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#404040", lw=0.75, zorder=1)
    ax.scatter(
        y_train,
        y_pred_train,
        s=14,
        facecolor=train_color,
        edgecolor="white",
        linewidth=0.3,
        zorder=2,
        label="Train",
    )
    ax.scatter(
        y_true,
        y_pred,
        s=18,
        facecolor=test_color,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
        label="Test",
    )

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True SOH")
    ax.set_ylabel("Predicted SOH")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    text = f"MAE={mae:.3f}\nRMSE={rmse:.3f}"
    ax.text(
        0.045,
        0.955,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6,
        linespacing=1.15,
    )
    ax.legend(loc="lower right", bbox_to_anchor=(0.72, 0.03), frameon=False, handletextpad=0.25, borderpad=0.15)
    ax.text(0.96, 0.05, "G1C1", transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
    style_axes(ax)
    fig.subplots_adjust(left=0.26, right=0.98, bottom=0.22, top=0.98)
    save_figure(fig, "G1C1_within_cell_pred_vs_true")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape_percent": mape,
        "n_test": float(len(y_true)),
        "n_train": float(len(y_train)),
        "matched_seed": float(matched_seed),
    }


def plot_seed_stability() -> dict[str, dict[str, float]]:
    rows = read_csv(SEED_CSV)
    seeds = np.asarray([int(float(r["seed"])) for r in rows], dtype=int)
    order = np.argsort(seeds)
    seeds = seeds[order]
    test_mae = np.asarray([as_float(r, "test_mae") for r in rows], dtype=float)[order]

    summary = {
        "test_mae": metric_summary([as_float(r, "test_mae") for r in rows]),
        "test_rmse": metric_summary([as_float(r, "test_rmse") for r in rows]),
        "test_mape_percent": metric_summary([as_float(r, "test_mape_percent") for r in rows]),
        "test_r2": metric_summary([as_float(r, "test_r2") for r in rows]),
        "test_p95_abs_error": metric_summary([as_float(r, "test_p95_abs_error") for r in rows]),
    }

    mean_mae = summary["test_mae"]["mean"]
    std_mae = summary["test_mae"]["std"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    color = "#1f7f88"
    ax.axhspan(mean_mae - std_mae, mean_mae + std_mae, color="#b9d9ee", alpha=0.85, lw=0, label="Mean ± Std")
    ax.axhline(mean_mae, color="#c23b22", lw=0.9, label="Mean")
    ax.plot(seeds, test_mae, color=color, lw=0.7, alpha=0.85)
    ax.scatter(seeds, test_mae, s=16, facecolor=color, edgecolor="white", linewidth=0.35, zorder=3)

    ax.set_xlabel("Random seed")
    ax.set_ylabel("Test MAE (SOH)")
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.set_xlim(float(np.min(seeds)) - 4, float(np.max(seeds)) + 4)
    pad = max(0.001, (float(np.max(test_mae)) - float(np.min(test_mae))) * 0.16)
    ax.set_ylim(float(np.min(test_mae)) - pad, float(np.max(test_mae)) + pad)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.48, 1.0),
        frameon=False,
        handlelength=1.6,
        borderpad=0.2,
        ncol=2,
        columnspacing=0.9,
    )
    ax.text(0.96, 0.05, "G1C1", transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
    style_axes(ax)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.23, top=0.96)
    save_figure(fig, "G1C1_seed_sweep_test_mae")

    with (OUT_DIR / "G1C1_seed_sweep_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "min", "max", "mean", "std"])
        for metric, stats in summary.items():
            writer.writerow([metric, stats["min"], stats["max"], stats["mean"], stats["std"]])
    return summary


def main() -> None:
    single = plot_pred_vs_true()
    seed_summary = plot_seed_stability()
    print("Single-run G1C1 test metrics")
    for key in ["n_train", "n_test", "matched_seed", "mae", "rmse", "r2", "mape_percent"]:
        print(f"{key}: {single[key]}")
    print("\nSeed-sweep summary")
    for metric, stats in seed_summary.items():
        print(
            f"{metric}: min={stats['min']:.10f}, max={stats['max']:.10f}, "
            f"mean={stats['mean']:.10f}, std={stats['std']:.10f}"
        )


if __name__ == "__main__":
    main()
