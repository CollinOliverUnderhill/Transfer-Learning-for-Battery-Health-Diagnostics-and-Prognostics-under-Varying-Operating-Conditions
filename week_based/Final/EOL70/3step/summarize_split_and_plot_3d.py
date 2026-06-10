#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def to_float(value: object) -> float:
    try:
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return float("nan")
        return float(text)
    except Exception:
        return float("nan")


def is_finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


def pick_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    fieldset = set(fieldnames)
    for col in candidates:
        if col in fieldset:
            return col
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Summarize split statistics and draw 3D operating-condition distribution by split."
    )
    ap.add_argument("--split_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--y_col", type=str, default="lifetime_weeks_EOL70")
    ap.add_argument("--title", type=str, default="Week5 High-Tail Split: 3D Operating Conditions")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    split_csv = Path(args.split_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    with split_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    for col in ("split", "cell", "group_num"):
        if col not in fieldnames:
            raise ValueError(f"Missing required column '{col}' in {split_csv}")

    x_col = pick_column(fieldnames, ["chg_c_rate", "charging_crate"])
    y_col_cond = pick_column(fieldnames, ["dchg_c_rate", "discharging_crate"])
    z_col = pick_column(fieldnames, ["dod_pct", "f3_w5", "f3", "step3_DoD"])
    if not x_col or not y_col_cond or not z_col:
        raise ValueError("Cannot find condition columns (charge/discharge/DoD).")

    # Summary by split
    by_split_rows: dict[str, int] = defaultdict(int)
    by_split_cells: dict[str, set[str]] = defaultdict(set)
    by_split_groups: dict[str, set[str]] = defaultdict(set)
    by_split_y: dict[str, list[float]] = defaultdict(list)

    # Summary by split x domain
    has_domain = "target_domain" in fieldnames
    by_sd_rows: dict[tuple[str, str], int] = defaultdict(int)
    by_sd_cells: dict[tuple[str, str], set[str]] = defaultdict(set)
    by_sd_groups: dict[tuple[str, str], set[str]] = defaultdict(set)

    # 3D data cache
    plot_cache: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    plot_cells: dict[str, set[str]] = defaultdict(set)

    for r in rows:
        split_name = str(r.get("split", "")).strip()
        cell = str(r.get("cell", "")).strip()
        group_num = str(r.get("group_num", "")).strip()
        by_split_rows[split_name] += 1
        if cell:
            by_split_cells[split_name].add(cell)
        if group_num:
            by_split_groups[split_name].add(group_num)

        y_val = to_float(r.get(args.y_col, ""))
        if is_finite(y_val):
            by_split_y[split_name].append(y_val)

        if has_domain:
            domain = str(r.get("target_domain", "")).strip()
            key = (split_name, domain)
            by_sd_rows[key] += 1
            if cell:
                by_sd_cells[key].add(cell)
            if group_num:
                by_sd_groups[key].add(group_num)

        xv = to_float(r.get(x_col, ""))
        yv = to_float(r.get(y_col_cond, ""))
        zv = to_float(r.get(z_col, ""))
        if is_finite(xv) and is_finite(yv) and is_finite(zv):
            plot_cache[split_name].append((xv, yv, zv))
            if cell:
                plot_cells[split_name].add(cell)

    summary_csv = out_dir / "split_summary_counts.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "split",
            "n_rows",
            "n_cells",
            "n_groups",
            f"{args.y_col}_mean",
            f"{args.y_col}_median",
            f"{args.y_col}_min",
            f"{args.y_col}_max",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for split_name in sorted(by_split_rows.keys()):
            ys = by_split_y.get(split_name, [])
            w.writerow(
                {
                    "split": split_name,
                    "n_rows": by_split_rows[split_name],
                    "n_cells": len(by_split_cells[split_name]),
                    "n_groups": len(by_split_groups[split_name]),
                    f"{args.y_col}_mean": mean(ys) if ys else "",
                    f"{args.y_col}_median": median(ys) if ys else "",
                    f"{args.y_col}_min": min(ys) if ys else "",
                    f"{args.y_col}_max": max(ys) if ys else "",
                }
            )

    split_domain_csv = None
    if has_domain:
        split_domain_csv = out_dir / "split_domain_summary_counts.csv"
        with split_domain_csv.open("w", encoding="utf-8-sig", newline="") as f:
            fields = ["split", "target_domain", "n_rows", "n_cells", "n_groups"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for key in sorted(by_sd_rows.keys()):
                split_name, domain = key
                w.writerow(
                    {
                        "split": split_name,
                        "target_domain": domain,
                        "n_rows": by_sd_rows[key],
                        "n_cells": len(by_sd_cells[key]),
                        "n_groups": len(by_sd_groups[key]),
                    }
                )

    # 3D scatter
    plot_png = out_dir / "split_3d_operating_conditions.png"
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        colors = {"train": "#1f77b4", "fine_tune": "#d62728", "test": "#2ca02c"}
        order = ["train", "fine_tune", "test"]

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for split_name in order:
            pts = plot_cache.get(split_name, [])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            ax.scatter(
                xs,
                ys,
                zs,
                s=38,
                alpha=0.85,
                c=colors.get(split_name, "#7f7f7f"),
                label=f"{split_name} (cells={len(plot_cells.get(split_name, []))})",
                edgecolors="k",
                linewidths=0.2,
            )

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col_cond)
        ax.set_zlabel(z_col)
        ax.set_title(args.title)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plot_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved 3D plot: {plot_png}")
    except ModuleNotFoundError:
        print("[WARN] matplotlib not installed; skip 3D plot.")

    print(f"Saved summary: {summary_csv}")
    if split_domain_csv is not None:
        print(f"Saved split-domain summary: {split_domain_csv}")


if __name__ == "__main__":
    main()

