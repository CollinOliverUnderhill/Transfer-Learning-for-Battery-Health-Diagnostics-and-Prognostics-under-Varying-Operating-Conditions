#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Matplotlib helpers for IVAS figures using PulseBat-combo styling."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, List

import matplotlib as mpl
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "Figure"
WEEK = ROOT / "week_based"
DPI = 600
CM = 1 / 2.54

COLORS = {
    "source": "#6BB7B2",
    "benchmark": "#7F7F7F",
    "source_only": "#B54D4D",
    "transfer": "#2F9D6A",
    "fine_tune": "#F28E2B",
    "test": "#73B66B",
    "grid": "#E8E8E8",
    "black": "#111111",
    "mid_grey": "#B8B8B8",
    "missing": "#D2D6DA",
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


def cm_to_in(cm: float) -> float:
    return cm / 2.54


def read_csv(path: Path) -> List[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: object, default: float = float("nan")) -> float:
    if value is None:
        return default
    text = str(value).strip().replace("%", "")
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def finite(value: float) -> bool:
    return bool(np.isfinite(value))


def group_num_from_cell(cell: object) -> int | None:
    text = str(cell or "")
    match = re.search(r"(?:^|[_\-])G?(\d{1,2})(?:C|[_\-]|$)", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"group[_\-]?(\d{1,2})", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def style_axes(ax) -> None:
    ax.tick_params(length=2.0, width=0.6, pad=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)


def save_figure(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=DPI)
    fig.savefig(out_dir / f"{stem}.pdf")


def nice_limits(values: Iterable[float], pad_frac: float = 0.08, floor_zero: bool = False) -> tuple[float, float]:
    vals = [float(v) for v in values if finite(float(v))]
    if not vals:
        return (0.0, 1.0)
    lo, hi = min(vals), max(vals)
    if lo == hi:
        pad = abs(lo) * 0.1 if lo else 1.0
    else:
        pad = (hi - lo) * pad_frac
    lo -= pad
    hi += pad
    if floor_zero:
        lo = min(0.0, lo)
    return lo, hi

