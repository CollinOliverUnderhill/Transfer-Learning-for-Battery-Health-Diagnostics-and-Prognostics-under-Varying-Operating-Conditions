#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Clone feature candidate CSV from one week suffix to another.")
    ap.add_argument("--in_csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--from_week", type=int, required=True)
    ap.add_argument("--to_week", type=int, required=True)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)
    if "features" not in df.columns:
        raise ValueError(f"{in_csv} is missing required column 'features'.")

    pat = re.compile(rf"_w{int(args.from_week)}\b")
    df["features"] = df["features"].astype(str).apply(lambda s: pat.sub(f"_w{int(args.to_week)}", s))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()

