#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Rebuild fine_tune cells from target domain with high-lifetime priority."
    )
    ap.add_argument("--in_split_csv", type=str, required=True, help="Original split CSV (train/fine_tune/test).")
    ap.add_argument("--out_split_csv", type=str, required=True, help="Output split CSV.")
    ap.add_argument("--y_col", type=str, default="lifetime_weeks_EOL70")
    ap.add_argument(
        "--target_pool_splits",
        type=str,
        default="fine_tune,test",
        help="Comma-separated split names used as target candidate pool.",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    in_csv = Path(args.in_split_csv)
    out_csv = Path(args.out_split_csv)

    if not in_csv.exists():
        raise FileNotFoundError(f"Input split CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)
    for col in ("split", "cell", args.y_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {in_csv}")

    df["split"] = df["split"].astype(str).str.strip()
    df["cell"] = df["cell"].astype(str).str.strip()
    df[args.y_col] = pd.to_numeric(df[args.y_col], errors="coerce")

    target_pool_splits = [x.strip() for x in str(args.target_pool_splits).split(",") if x.strip()]
    if not target_pool_splits:
        raise ValueError("--target_pool_splits is empty.")

    ft_cells_old = sorted(df.loc[df["split"] == "fine_tune", "cell"].dropna().unique().tolist())
    ft_quota = len(ft_cells_old)
    if ft_quota == 0:
        raise ValueError("No existing fine_tune cells found in input split.")

    target_pool = df[df["split"].isin(target_pool_splits)].copy()
    if len(target_pool) == 0:
        raise ValueError("Target pool is empty; check --target_pool_splits.")

    cell_lifetime = (
        target_pool[["cell", args.y_col]]
        .drop_duplicates(subset=["cell"])
        .dropna(subset=[args.y_col])
        .sort_values([args.y_col, "cell"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if len(cell_lifetime) < ft_quota:
        raise ValueError(
            f"Target pool valid cells ({len(cell_lifetime)}) < fine_tune quota ({ft_quota})."
        )

    new_ft_cells = set(cell_lifetime.head(ft_quota)["cell"].astype(str).tolist())
    pool_mask = df["split"].isin(target_pool_splits)
    df.loc[pool_mask, "split"] = "test"
    df.loc[df["cell"].isin(new_ft_cells) & pool_mask, "split"] = "fine_tune"

    if "split_label" in df.columns:
        df.loc[df["split"] == "train", "split_label"] = "Train"
        df.loc[df["split"] == "fine_tune", "split_label"] = "Fine-tune"
        df.loc[df["split"] == "test", "split_label"] = "Test"
    if "target_domain" in df.columns:
        df.loc[df["split"] == "train", "target_domain"] = "source"
        df.loc[df["split"].isin(["fine_tune", "test"]), "target_domain"] = "target"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Saved: {out_csv}")
    print(f"fine_tune quota kept: {ft_quota}")
    print(f"new fine_tune cells : {df.loc[df['split']=='fine_tune', 'cell'].nunique()}")


if __name__ == "__main__":
    main()

