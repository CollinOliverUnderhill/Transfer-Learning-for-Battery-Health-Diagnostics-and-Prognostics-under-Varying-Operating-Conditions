#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


ROOT = Path(r"E:\Datasets\IVAS")
OUT_DIR = ROOT / "Figure" / "Figure_later" / "42"

RIDGE_DIR = ROOT / "week_based" / "SOHest" / "results" / "ridge_cross_cell" / "domain_shift" / "scheme2_37train_12test"
MLP_DIR = ROOT / "week_based" / "SOHest" / "results" / "mlp_cross_cell" / "scheme2_37train_12test"

CM = 1 / 2.54
FIG_PRED = (12.74 * CM, 4.00 * CM)
FIG_MAE = (6.37 * CM, 4.00 * CM)


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


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def sample_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def metric_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": sample_std(values),
    }


def load_model(model_name: str, base: Path) -> dict[str, object]:
    train_pred = read_csv(base / "predictions_train.csv")
    pred = read_csv(base / "predictions_test.csv")
    cell = read_csv(base / "test_cell_metrics.csv")
    overall = read_csv(base / "test_overall_metrics.csv")[0]

    y_train_true = np.asarray([f(r, "y_true") for r in train_pred], dtype=float)
    y_train_pred = np.asarray([f(r, "y_pred") for r in train_pred], dtype=float)
    y_true = np.asarray([f(r, "y_true") for r in pred], dtype=float)
    y_pred = np.asarray([f(r, "y_pred") for r in pred], dtype=float)
    cell_mae = np.asarray([f(r, "mae") for r in cell], dtype=float)
    cell_rmse = np.asarray([f(r, "rmse") for r in cell], dtype=float)
    cell_mape = np.asarray([f(r, "mape_percent") for r in cell], dtype=float)
    cell_r2 = np.asarray([f(r, "r2") for r in cell if r.get("r2", "") not in {"", "nan", "NaN"}], dtype=float)

    return {
        "name": model_name,
        "path": base,
        "train_pred": train_pred,
        "pred": pred,
        "cell": cell,
        "overall": overall,
        "y_train_true": y_train_true,
        "y_train_pred": y_train_pred,
        "y_true": y_true,
        "y_pred": y_pred,
        "cell_mae": cell_mae,
        "cell_rmse": cell_rmse,
        "cell_mape": cell_mape,
        "cell_r2": cell_r2,
        "mae_stats": metric_stats(cell_mae),
        "rmse_stats": metric_stats(cell_rmse),
        "mape_stats": metric_stats(cell_mape),
        "r2_stats": metric_stats(cell_r2) if cell_r2.size else {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan},
    }


def style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    ax.tick_params(axis="both", which="major", width=0.6, length=2.2, pad=1.5)
    ax.tick_params(axis="both", which="minor", width=0.5, length=1.4)


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=600)
    plt.close(fig)


def plot_prediction_panels(models: list[dict[str, object]]) -> None:
    all_true = np.concatenate(
        [np.concatenate([m["y_train_true"], m["y_true"]]) for m in models]  # type: ignore[index]
    )
    all_pred = np.concatenate(
        [np.concatenate([m["y_train_pred"], m["y_pred"]]) for m in models]  # type: ignore[index]
    )
    lo = min(float(np.min(all_true)), float(np.min(all_pred))) - 0.025
    hi = max(float(np.max(all_true)), float(np.max(all_pred))) + 0.025

    fig, axes = plt.subplots(1, 2, figsize=FIG_PRED)
    colors = {
        "Ridge": {"train": "#1f7f88", "test": "#e6862e"},
        "MLP": {"train": "#2f6fb0", "test": "#c83b2b"},
    }
    labels = ["a", "b"]

    for ax, model, panel in zip(axes, models, labels):
        y_true = model["y_true"]  # type: ignore[assignment]
        y_pred = model["y_pred"]  # type: ignore[assignment]
        y_train_true = model["y_train_true"]  # type: ignore[assignment]
        y_train_pred = model["y_train_pred"]  # type: ignore[assignment]
        overall = model["overall"]  # type: ignore[assignment]
        name = str(model["name"])

        ax.plot([lo, hi], [lo, hi], color="#404040", lw=0.75, zorder=1)
        ax.scatter(
            y_train_true,
            y_train_pred,
            s=5.8,
            facecolor=colors[name]["train"],
            edgecolor="white",
            linewidth=0.12,
            alpha=0.34,
            zorder=2,
            label="Train",
        )
        ax.scatter(
            y_true,
            y_pred,
            s=7.5,
            facecolor=colors[name]["test"],
            edgecolor="white",
            linewidth=0.18,
            alpha=0.88,
            zorder=3,
            label="Test",
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_xlabel("")
        if ax is axes[0]:
            ax.set_ylabel("Predicted SOH")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.text(0.05, 0.95, panel, transform=ax.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")
        ax.text(0.96, 0.07, name, transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(0.67, 0.05),
            frameon=False,
            handletextpad=0.25,
            borderpad=0.1,
            markerscale=1.25,
        )
        text = (
            f"MAE={float(overall['mae']):.3f}\n"
            f"RMSE={float(overall['rmse']):.3f}\n"
            f"R$^2$={float(overall['r2']):.3f}"
        )
        ax.text(0.05, 0.82, text, transform=ax.transAxes, ha="left", va="top", fontsize=6, linespacing=1.15)
        style_axes(ax)

    # Equal-aspect axes shrink inside their subplot slots; fixed positions keep
    # the two frames visually adjacent in the final 12.74 cm x 4.00 cm canvas.
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
        "True SOH",
        ha="center",
        va="center",
        fontsize=8,
    )
    save(fig, "cross_cell_soh_pred_vs_true_ridge_mlp")


def plot_mae_summary(models: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=FIG_MAE)
    colors = ["#1f4e79", "#d96c9f"]
    x = np.arange(len(models), dtype=float)
    width = 0.46

    rng = np.random.default_rng(2026)
    for i, (model, color) in enumerate(zip(models, colors)):
        cell_mae = model["cell_mae"]  # type: ignore[assignment]
        stats = model["mae_stats"]  # type: ignore[assignment]
        overall = model["overall"]  # type: ignore[assignment]
        mean = float(stats["mean"])
        std = float(stats["std"])

        ax.bar(i, mean, width=width, color=color, alpha=0.55, edgecolor="none", zorder=1)
        ax.errorbar(i, mean, yerr=std, color="#4d4d4d", lw=0.75, capsize=3.0, capthick=0.75, zorder=4)
        ax.boxplot(
            [cell_mae],
            positions=[i],
            widths=width * 0.52,
            patch_artist=True,
            showfliers=False,
            boxprops={
                "facecolor": "none",
                "edgecolor": "#333333",
                "linewidth": 0.75,
                "zorder": 5,
            },
            medianprops={"color": "#333333", "linewidth": 0.9, "zorder": 6},
            whiskerprops={"color": "#333333", "linewidth": 0.75, "zorder": 5},
            capprops={"color": "#333333", "linewidth": 0.75, "zorder": 5},
        )
        jitter = rng.normal(0.0, 0.035, size=len(cell_mae))
        ax.scatter(
            np.full(len(cell_mae), i) + jitter,
            cell_mae,
            s=18,
            facecolor="white",
            edgecolor=color,
            linewidth=0.55,
            alpha=0.52,
            zorder=7,
        )
        ax.plot([i - width * 0.40, i + width * 0.40], [float(overall["mae"]), float(overall["mae"])], color="#111111", lw=1.05, zorder=8)

    ax.set_xlim(-0.55, len(models) - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([str(m["name"]) for m in models])
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE (SOH)")
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(axis="y", color="#e5e5e5", lw=0.5, zorder=0)
    style_axes(ax)
    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.28, top=0.97)
    save(fig, "cross_cell_soh_mae_mean_std")


def write_summary(models: list[dict[str, object]]) -> None:
    out = OUT_DIR / "cross_cell_soh_summary.csv"
    with out.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "model",
                "setting",
                "n_rows",
                "n_cells",
                "n_groups",
                "overall_mae",
                "overall_rmse",
                "overall_r2",
                "overall_mape_percent",
                "cell_mae_min",
                "cell_mae_max",
                "cell_mae_mean",
                "cell_mae_std",
                "cell_rmse_mean",
                "cell_rmse_std",
                "cell_mape_mean",
                "cell_mape_std",
            ]
        )
        for model in models:
            overall = model["overall"]  # type: ignore[assignment]
            mae_stats = model["mae_stats"]  # type: ignore[assignment]
            rmse_stats = model["rmse_stats"]  # type: ignore[assignment]
            mape_stats = model["mape_stats"]  # type: ignore[assignment]
            writer.writerow(
                [
                    model["name"],
                    "source-cell training to held-out target-cell testing",
                    overall["n_rows"],
                    overall["n_cells"],
                    overall["n_groups"],
                    overall["mae"],
                    overall["rmse"],
                    overall["r2"],
                    overall["mape_percent"],
                    mae_stats["min"],
                    mae_stats["max"],
                    mae_stats["mean"],
                    mae_stats["std"],
                    rmse_stats["mean"],
                    rmse_stats["std"],
                    mape_stats["mean"],
                    mape_stats["std"],
                ]
            )


def main() -> None:
    ridge = load_model("Ridge", RIDGE_DIR)
    mlp = load_model("MLP", MLP_DIR)
    models = [ridge, mlp]
    plot_prediction_panels(models)
    plot_mae_summary(models)
    write_summary(models)
    for model in models:
        overall = model["overall"]  # type: ignore[assignment]
        mae_stats = model["mae_stats"]  # type: ignore[assignment]
        print(
            f"{model['name']}: n_rows={overall['n_rows']}, n_cells={overall['n_cells']}, "
            f"n_groups={overall['n_groups']}, overall_MAE={float(overall['mae']):.6f}, "
            f"overall_RMSE={float(overall['rmse']):.6f}, R2={float(overall['r2']):.6f}, "
            f"MAPE={float(overall['mape_percent']):.6f}, "
            f"cell_MAE_mean={float(mae_stats['mean']):.6f}, cell_MAE_std={float(mae_stats['std']):.6f}"
        )


if __name__ == "__main__":
    main()
