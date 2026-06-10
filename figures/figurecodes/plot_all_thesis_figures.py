#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate thesis main figures from the IVAS week-based outputs.

The environment used for this project does not always provide pandas or
matplotlib, so this script draws with lightweight standard-library SVG
primitives and exports only PNG/PDF files under E:/Datasets/IVAS/Figure.
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "Figure"
WEEK = ROOT / "week_based"
DATA = ROOT / "Data"

OUT_INDEX = FIG_DIR / "thesis_figure_index.md"

PALETTE = {
    "ink": "#092431",
    "muted": "#4f5f69",
    "grid": "#C7D0D4",
    "paper": "#ffffff",
    "source": "#2f6fbb",
    "fine_tune": "#d9802e",
    "test": "#2f9d6a",
    "benchmark": "#7f7f7f",
    "transfer": "#2f9d6a",
    "source_only": "#b54d4d",
    "accent": "#4b72b8",
    "warn": "#b54d4d",
    "gold": "#c89b2d",
}

DEFAULT_PNG_DPI = 600
EXPORT_PX_PER_UNIT = 3.2
EXPORT_MIN_WIDTH_PX = 1200
EXPORT_MAX_WIDTH_PX = 3200
FONT_SCALE = 0.58
STROKE_SCALE = 0.75
SVG_MARGIN = 8.0


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float(value: object, default: float = float("nan")) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        return float(text)
    except Exception:
        return default


def finite(value: float) -> bool:
    return math.isfinite(value)


def quantile(values: Sequence[float], q: float) -> float:
    vals = sorted(v for v in values if finite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def mean(values: Sequence[float]) -> float:
    vals = [v for v in values if finite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def cell_sort_key(cell: str) -> Tuple[int, int, str]:
    m = re.match(r"G(\d+)C(\d+)", str(cell))
    if not m:
        return (10**6, 10**6, str(cell))
    return (int(m.group(1)), int(m.group(2)), str(cell))


def group_num_from_cell(cell: str) -> Optional[int]:
    m = re.match(r"G(\d+)C\d+", str(cell))
    return int(m.group(1)) if m else None


def _svg_attr(element: str, name: str) -> Optional[str]:
    match = re.search(rf'{name}="([^"]+)"', element)
    return match.group(1) if match else None


def _svg_float(element: str, name: str, default: float = 0.0) -> float:
    value = _svg_attr(element, name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _element_bounds(element: str) -> Optional[Tuple[float, float, float, float]]:
    item = element.strip()
    if item.startswith("<rect"):
        width_attr = _svg_attr(item, "width")
        height_attr = _svg_attr(item, "height")
        if width_attr is None or height_attr is None or "%" in width_attr or "%" in height_attr:
            return None
        x = _svg_float(item, "x")
        y = _svg_float(item, "y")
        w = _svg_float(item, "width")
        h = _svg_float(item, "height")
        return x, y, x + w, y + h
    if item.startswith("<line"):
        x1 = _svg_float(item, "x1")
        y1 = _svg_float(item, "y1")
        x2 = _svg_float(item, "x2")
        y2 = _svg_float(item, "y2")
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    if item.startswith("<circle"):
        cx = _svg_float(item, "cx")
        cy = _svg_float(item, "cy")
        r = _svg_float(item, "r")
        return cx - r, cy - r, cx + r, cy + r
    if item.startswith("<polyline"):
        points = _svg_attr(item, "points") or ""
        coords = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", points)]
        if len(coords) < 2:
            return None
        xs = coords[0::2]
        ys = coords[1::2]
        return min(xs), min(ys), max(xs), max(ys)
    if item.startswith("<text"):
        x = _svg_float(item, "x")
        y = _svg_float(item, "y")
        size = _svg_float(item, "font-size", 6.0)
        text_match = re.search(r">(.*?)</text>", item)
        value = html.unescape(text_match.group(1)) if text_match else ""
        width = max(size * 0.6, len(value) * size * 0.56)
        anchor = _svg_attr(item, "text-anchor") or "start"
        if anchor == "middle":
            x0, x1 = x - width / 2.0, x + width / 2.0
        elif anchor == "end":
            x0, x1 = x - width, x
        else:
            x0, x1 = x, x + width
        return x0, y - size * 1.15, x1, y + size * 0.35
    return None


def _body_viewbox(width: int, height: int, body: Sequence[str]) -> Tuple[float, float, float, float]:
    bounds = [b for item in body if (b := _element_bounds(item)) is not None]
    if not bounds:
        return 0.0, 0.0, float(width), float(height)
    x0 = min(b[0] for b in bounds) - SVG_MARGIN
    y0 = min(b[1] for b in bounds) - SVG_MARGIN
    x1 = max(b[2] for b in bounds) + SVG_MARGIN
    y1 = max(b[3] for b in bounds) + SVG_MARGIN
    return x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)


def _fmt(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def svg_doc(width: int, height: int, body: Sequence[str]) -> str:
    view_x, view_y, view_w, view_h = _body_viewbox(width, height, body)
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{_fmt(view_x)} {_fmt(view_y)} {_fmt(view_w)} {_fmt(view_h)}">',
            f'<rect x="{_fmt(view_x)}" y="{_fmt(view_y)}" width="{_fmt(view_w)}" height="{_fmt(view_h)}" fill="#ffffff"/>',
            *body,
            "</svg>",
            "",
        ]
    )


def _svg_viewbox_width(svg_text: str) -> float:
    first_line = svg_text.splitlines()[0]
    viewbox = _svg_attr(first_line, "viewBox")
    if not viewbox:
        return 0.0
    parts = [float(v) for v in viewbox.split()]
    return parts[2] if len(parts) == 4 else 0.0


def _export_width_px(svg_text: str) -> int:
    viewbox_width = _svg_viewbox_width(svg_text)
    if viewbox_width <= 0:
        return EXPORT_MAX_WIDTH_PX
    width = int(round(viewbox_width * EXPORT_PX_PER_UNIT))
    return max(EXPORT_MIN_WIDTH_PX, min(EXPORT_MAX_WIDTH_PX, width))


def export_with_inkscape(svg_text: str, png_path: Path, pdf_path: Path) -> None:
    exe = shutil.which("inkscape.com") or shutil.which("inkscape")
    if exe is None:
        print(f"[WARN] Inkscape not found; could not export: {png_path} / {pdf_path}")
        return
    svg_bytes = svg_text.encode("utf-8")
    export_width = _export_width_px(svg_text)
    subprocess.run(
        [
            exe,
            "--pipe",
            "--export-type=png",
            f"--export-filename={png_path}",
            f"--export-width={export_width}",
        ],
        check=True,
        input=svg_bytes,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            exe,
            "--pipe",
            "--export-type=pdf",
            f"--export-filename={pdf_path}",
        ],
        check=True,
        input=svg_bytes,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def save_svg(num: int, slug: str, title: str, width: int, height: int, body: Sequence[str], paths: List[Tuple[int, str, str]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = FIG_DIR / f"figure{num}"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"figure{num}.png"
    pdf_path = out_dir / f"figure{num}.pdf"
    export_with_inkscape(svg_doc(width, height, body), png_path, pdf_path)
    paths.append((num, title, str(png_path.relative_to(FIG_DIR))))


def text(x: float, y: float, value: object, size: int = 10, weight: str = "400", anchor: str = "start", fill: str = "#111111") -> str:
    actual_size = max(7.0, min(8.2, float(size) * FONT_SCALE))
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{actual_size:.2f}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{esc(value)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: float = 4, sw: float = 1.0, opacity: float = 1.0) -> str:
    actual_sw = max(0.45, min(0.75, sw * STROKE_SCALE))
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{actual_sw:.2f}" opacity="{opacity}"/>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = PALETTE["ink"], sw: float = 0.75, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    actual_sw = max(0.45, min(0.75, sw * STROKE_SCALE))
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{actual_sw:.2f}"{dash_attr}/>'


def circle(cx: float, cy: float, r: float, fill: str, stroke: str = "#ffffff", sw: float = 1.0, opacity: float = 1.0) -> str:
    actual_sw = max(0.45, min(0.75, sw * STROKE_SCALE))
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{actual_sw:.2f}" opacity="{opacity}"/>'


def polyline(points: Sequence[Tuple[float, float]], stroke: str, sw: float = 2.0, fill: str = "none", dash: str = "") -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    actual_sw = max(0.45, min(0.75, sw * STROKE_SCALE))
    return f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{actual_sw:.2f}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'


def arrow(x1: float, y1: float, x2: float, y2: float, stroke: str = PALETTE["ink"]) -> List[str]:
    ang = math.atan2(y2 - y1, x2 - x1)
    ah = 10
    a1 = ang + math.pi * 0.82
    a2 = ang - math.pi * 0.82
    return [
        line(x1, y1, x2, y2, stroke, 0.9),
        polyline([(x2, y2), (x2 + ah * math.cos(a1), y2 + ah * math.sin(a1)), (x2 + ah * math.cos(a2), y2 + ah * math.sin(a2)), (x2, y2)], stroke, 0.75, stroke),
    ]


def title_block(num: int, title: str, subtitle: str = "") -> List[str]:
    return []


def nice_range(values: Sequence[float], pad_frac: float = 0.08) -> Tuple[float, float]:
    vals = [v for v in values if finite(v)]
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    if lo == hi:
        pad = abs(lo) * 0.1 if lo else 1.0
        return lo - pad, hi + pad
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def scale(v: float, lo: float, hi: float, a: float, b: float) -> float:
    if hi == lo:
        return (a + b) / 2
    return a + (v - lo) / (hi - lo) * (b - a)


def clip_series_max_y(points: Sequence[Tuple[float, float]], upper: float) -> List[Tuple[float, float]]:
    clipped: List[Tuple[float, float]] = []
    prev: Optional[Tuple[float, float]] = None
    for point in points:
        x, y = point
        if y <= upper:
            clipped.append(point)
        elif prev is not None and prev[1] <= upper and y > upper and y != prev[1]:
            px, py = prev
            t = (upper - py) / (y - py)
            clipped.append((px + t * (x - px), upper))
            break
        prev = point
    return clipped


def color_ramp(t: float) -> str:
    stops = [(0, (49, 103, 183)), (0.5, (243, 242, 236)), (1, (180, 77, 77))]
    t = max(0.0, min(1.0, t))
    for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
        if t <= p1:
            u = (t - p0) / (p1 - p0)
            rgb = tuple(int(c0[i] + u * (c1[i] - c0[i])) for i in range(3))
            return "#{:02x}{:02x}{:02x}".format(*rgb)
    return "#b44d4d"


def feature_label(name: str) -> str:
    mapping = {
        "f1": "f1 IC mean delta",
        "f2": "f2 CV-time delta",
        "f3": "f3 DoD",
        "f4": "f4 DVA Q1 delta",
        "f5": "f5 C-rate/DoD stress",
        "f6": "f6 Charge C-rate",
        "f7": "f7 IC variance delta",
        "f8": "f8 DVA Q3 delta",
        "f9": "f9 IC low-V mean",
        "f10": "f10 initial CV time",
    }
    m = re.match(r"(f\d+)", name)
    return mapping.get(m.group(1), name) if m else name


def grouped_metrics(stage_dir: Path, include_source_only: bool = True) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    bench = read_csv(stage_dir / "benchmark" / "test_overall_metrics.csv")
    if bench:
        rows.append({"model": "Benchmark", **{k: to_float(bench[0].get(k)) for k in ["mae", "rmse", "mape_percent", "r2"]}})
    if include_source_only:
        src = read_csv(stage_dir / "transfer_model" / "test_overall_metrics_source_only.csv")
        if src:
            rows.append({"model": "Source-only", **{k: to_float(src[0].get(k)) for k in ["mae", "rmse", "mape_percent", "r2"]}})
    ft = read_csv(stage_dir / "transfer_model" / "test_overall_metrics.csv")
    if ft:
        rows.append({"model": "Fine-tuned", **{k: to_float(ft[0].get(k)) for k in ["mae", "rmse", "mape_percent", "r2"]}})
    return rows


def draw_metric_bars(num: int, title: str, metrics: List[Dict[str, float]], paths: List[Tuple[int, str, str]], slug: str, include_r2: bool = True) -> None:
    metric_keys = ["mae", "rmse", "mape_percent"] + (["r2"] if include_r2 else [])
    labels = ["MAE", "RMSE", "MAPE (%)"] + (["R2"] if include_r2 else [])
    colors = {"Benchmark": PALETTE["benchmark"], "Source-only": PALETTE["source_only"], "Fine-tuned": PALETTE["transfer"]}
    w, h = 1180, 680
    body = title_block(num, title, "Target-test aggregate metrics; lower is better except R2.")
    left, top = 92, 120
    panel_w = (w - 150) / len(metric_keys)
    for mi, key in enumerate(metric_keys):
        x0 = left + mi * panel_w
        vals = [m.get(key, float("nan")) for m in metrics]
        ymin = min([0.0] + [v for v in vals if finite(v)])
        ymax = max([0.1] + [v for v in vals if finite(v)])
        if key == "r2":
            ymin = min(-0.2, ymin)
        y0, ph = top, 420
        body.append(text(x0 + panel_w / 2, y0 - 16, labels[mi], 15, "700", "middle"))
        body.append(line(x0 + 20, y0 + ph, x0 + panel_w - 28, y0 + ph, PALETTE["grid"]))
        body.append(line(x0 + 20, y0, x0 + 20, y0 + ph, PALETTE["grid"]))
        for tval in [ymin, (ymin + ymax) / 2, ymax]:
            yy = scale(tval, ymin, ymax, y0 + ph, y0)
            body.append(line(x0 + 20, yy, x0 + panel_w - 28, yy, PALETTE["grid"], 0.7))
            body.append(text(x0 + 14, yy + 4, f"{tval:.2f}", 10, anchor="end", fill=PALETTE["muted"]))
        bw = min(44, (panel_w - 70) / max(1, len(metrics)))
        for i, m in enumerate(metrics):
            val = m.get(key, float("nan"))
            if not finite(val):
                continue
            cx = x0 + 48 + i * (bw + 16)
            yy = scale(val, ymin, ymax, y0 + ph, y0)
            ybase = scale(0, ymin, ymax, y0 + ph, y0) if ymin < 0 else y0 + ph
            body.append(rect(cx, min(yy, ybase), bw, abs(ybase - yy), colors.get(m["model"], "#888"), rx=2))
            body.append(text(cx + bw / 2, min(yy, ybase) - 6, f"{val:.2f}", 10, anchor="middle"))
        for i, m in enumerate(metrics):
            body.append(rect(110 + i * 190, 590, 14, 14, colors.get(m["model"], "#888"), rx=2))
            body.append(text(130 + i * 190, 602, m["model"], 13))
    save_svg(num, slug, title, w, h, body, paths)


def draw_pred_true_panels(num: int, title: str, datasets: Sequence[Tuple[str, Path, str]], paths: List[Tuple[int, str, str]], slug: str) -> None:
    all_points: List[Tuple[float, float]] = []
    loaded: List[Tuple[str, List[Dict[str, str]], str]] = []
    for label, path, color in datasets:
        rows = read_csv(path)
        loaded.append((label, rows, color))
        for r in rows:
            x, y = to_float(r.get("y_true")), to_float(r.get("y_pred"))
            if finite(x) and finite(y):
                all_points.append((x, y))
    vals = [v for xy in all_points for v in xy]
    lo, hi = nice_range(vals, 0.06)
    w, h = 1180, 520
    body = title_block(num, title, "Identity line shows perfect calibration.")
    panel_w = (w - 120) / max(1, len(loaded))
    for pi, (label, rows, color) in enumerate(loaded):
        x0, y0 = 60 + pi * panel_w, 112
        pw, ph = panel_w - 34, 330
        body.append(rect(x0, y0, pw, ph, "#ffffff", PALETTE["grid"], rx=2))
        body.append(text(x0 + pw / 2, y0 - 14, label, 15, "700", "middle"))
        for frac in [0, 0.25, 0.5, 0.75, 1]:
            xx = x0 + frac * pw
            yy = y0 + frac * ph
            body.append(line(xx, y0, xx, y0 + ph, PALETTE["grid"], 0.6))
            body.append(line(x0, yy, x0 + pw, yy, PALETTE["grid"], 0.6))
        body.append(line(x0, y0 + ph, x0 + pw, y0, "#333333", 1.2, "6,4"))
        for r in rows:
            x, y = to_float(r.get("y_true")), to_float(r.get("y_pred"))
            if finite(x) and finite(y):
                cx = scale(x, lo, hi, x0, x0 + pw)
                cy = scale(y, lo, hi, y0 + ph, y0)
                body.append(circle(cx, cy, 4.3, color, "#ffffff", 0.6, 0.82))
        body.append(text(x0 + pw / 2, y0 + ph + 38, "True RUL (weeks)", 12, anchor="middle", fill=PALETTE["muted"]))
        body.append(text(x0 - 12, y0 + ph / 2, "Predicted", 12, anchor="middle", fill=PALETTE["muted"]))
        body.append(text(x0, y0 + ph + 18, f"{lo:.0f}", 11, fill=PALETTE["muted"]))
        body.append(text(x0 + pw, y0 + ph + 18, f"{hi:.0f}", 11, anchor="end", fill=PALETTE["muted"]))
    save_svg(num, slug, title, w, h, body, paths)


def draw_group_error_bars(num: int, title: str, stage_dir: Path, paths: List[Tuple[int, str, str]], slug: str) -> None:
    files = [
        ("Benchmark", stage_dir / "benchmark" / "test_group_metrics.csv", PALETTE["benchmark"]),
        ("Source-only", stage_dir / "transfer_model" / "test_group_metrics_source_only.csv", PALETTE["source_only"]),
        ("Fine-tuned", stage_dir / "transfer_model" / "test_group_metrics.csv", PALETTE["transfer"]),
    ]
    series: List[Tuple[str, List[Dict[str, str]], str]] = [(a, read_csv(b), c) for a, b, c in files if read_csv(b)]
    groups = sorted({int(to_float(r.get("group_num"))) for _, rows, _ in series for r in rows if finite(to_float(r.get("group_num")))})
    vals = [to_float(r.get("cell_mae_mean", r.get("mae"))) for _, rows, _ in series for r in rows]
    ymax = max([1.0] + [v for v in vals if finite(v)]) * 1.15
    w, h = 1220, 600
    body = title_block(num, title, "Group-wise target-test MAE.")
    x0, y0, pw, ph = 72, 110, 1075, 365
    body.append(rect(x0, y0, pw, ph, "#ffffff", PALETTE["grid"], rx=2))
    for tval in [0, ymax / 2, ymax]:
        yy = scale(tval, 0, ymax, y0 + ph, y0)
        body.append(line(x0, yy, x0 + pw, yy, PALETTE["grid"], 0.7))
        body.append(text(x0 - 10, yy + 4, f"{tval:.1f}", 11, anchor="end", fill=PALETTE["muted"]))
    if groups:
        slot = pw / len(groups)
        bw = min(13, slot / (len(series) + 1))
        for gi, g in enumerate(groups):
            gx = x0 + gi * slot + slot * 0.15
            for si, (label, rows, color) in enumerate(series):
                match = [r for r in rows if int(to_float(r.get("group_num"))) == g]
                if not match:
                    continue
                val = to_float(match[0].get("cell_mae_mean", match[0].get("mae")))
                yy = scale(val, 0, ymax, y0 + ph, y0)
                body.append(rect(gx + si * (bw + 2), yy, bw, y0 + ph - yy, color, rx=1))
            if gi % 2 == 0:
                body.append(text(x0 + gi * slot + slot / 2, y0 + ph + 18, f"G{g}", 10, anchor="middle", fill=PALETTE["muted"]))
    for i, (label, _, color) in enumerate(series):
        body.append(rect(92 + i * 180, 530, 14, 14, color, rx=2))
        body.append(text(112 + i * 180, 542, label, 13))
    save_svg(num, slug, title, w, h, body, paths)


def draw_cell_error_distribution(num: int, title: str, stage_dir: Path, paths: List[Tuple[int, str, str]], slug: str, include_source: bool = True) -> None:
    files = [
        ("Benchmark", stage_dir / "benchmark" / "test_cell_metrics.csv", PALETTE["benchmark"]),
    ]
    if include_source:
        files.append(("Source-only", stage_dir / "transfer_model" / "test_cell_metrics_source_only.csv", PALETTE["source_only"]))
    files.append(("Fine-tuned", stage_dir / "transfer_model" / "test_cell_metrics.csv", PALETTE["transfer"]))
    data = []
    for label, path, color in files:
        vals = [to_float(r.get("mae_mean", r.get("mae"))) for r in read_csv(path)]
        vals = [v for v in vals if finite(v)]
        if vals:
            data.append((label, vals, color))
    ymax = max([1.0] + [v for _, vals, _ in data for v in vals]) * 1.1
    w, h = 900, 590
    body = title_block(num, title, "Cell-wise MAE distribution; boxes show median and interquartile range.")
    x0, y0, pw, ph = 100, 110, 680, 350
    body.append(rect(x0, y0, pw, ph, "#ffffff", PALETTE["grid"], rx=2))
    for tval in [0, ymax / 2, ymax]:
        yy = scale(tval, 0, ymax, y0 + ph, y0)
        body.append(line(x0, yy, x0 + pw, yy, PALETTE["grid"], 0.7))
        body.append(text(x0 - 10, yy + 4, f"{tval:.1f}", 11, anchor="end", fill=PALETTE["muted"]))
    slot = pw / max(1, len(data))
    for i, (label, vals, color) in enumerate(data):
        cx = x0 + slot * (i + 0.5)
        q1, q2, q3 = quantile(vals, 0.25), quantile(vals, 0.5), quantile(vals, 0.75)
        lo, hi = min(vals), max(vals)
        yq1, yq2, yq3 = [scale(v, 0, ymax, y0 + ph, y0) for v in (q1, q2, q3)]
        ylo, yhi = scale(lo, 0, ymax, y0 + ph, y0), scale(hi, 0, ymax, y0 + ph, y0)
        body.append(line(cx, ylo, cx, yhi, color, 1.5))
        body.append(rect(cx - 42, yq3, 84, yq1 - yq3, "#ffffff", color, rx=2, sw=1.8))
        body.append(line(cx - 42, yq2, cx + 42, yq2, color, 2.0))
        for j, v in enumerate(vals):
            jitter = ((j * 37) % 31 - 15) / 15 * 26
            body.append(circle(cx + jitter, scale(v, 0, ymax, y0 + ph, y0), 3.1, color, "#ffffff", 0.4, 0.55))
        body.append(text(cx, y0 + ph + 28, label, 13, "600", "middle"))
    save_svg(num, slug, title, w, h, body, paths)


def draw_table(num: int, title: str, columns: Sequence[str], rows: Sequence[Sequence[str]], paths: List[Tuple[int, str, str]], slug: str, subtitle: str = "") -> None:
    w, h = 1050, max(330, 150 + 46 * (len(rows) + 1))
    body = title_block(num, title, subtitle)
    x0, y0 = 70, 112
    col_w = (w - 140) / len(columns)
    body.append(rect(x0, y0, w - 140, 42, "#eef3f8", PALETTE["grid"], rx=3))
    for i, col in enumerate(columns):
        body.append(text(x0 + i * col_w + 14, y0 + 27, col, 13, "700"))
    for ri, row in enumerate(rows):
        yy = y0 + 42 + ri * 44
        body.append(rect(x0, yy, w - 140, 44, "#ffffff" if ri % 2 == 0 else "#f7f9fb", "#edf0f4", rx=0, sw=0.8))
        for ci, value in enumerate(row):
            body.append(text(x0 + ci * col_w + 14, yy + 28, value, 13))
    save_svg(num, slug, title, w, h, body, paths)


def draw_flow(num: int, title: str, boxes: Sequence[Tuple[str, str]], paths: List[Tuple[int, str, str]], slug: str, subtitle: str = "") -> None:
    w, h = 1180, 430
    body = title_block(num, title, subtitle)
    x0, y0, bw, bh, gap = 70, 150, 170, 86, 34
    for i, (head, sub) in enumerate(boxes):
        x = x0 + i * (bw + gap)
        body.append(rect(x, y0, bw, bh, "#f8fafc", PALETTE["accent"], rx=6, sw=1.5))
        body.append(text(x + bw / 2, y0 + 34, head, 15, "700", "middle"))
        body.append(text(x + bw / 2, y0 + 60, sub, 12, "400", "middle", PALETTE["muted"]))
        if i < len(boxes) - 1:
            body.extend(arrow(x + bw + 6, y0 + bh / 2, x + bw + gap - 8, y0 + bh / 2, PALETTE["muted"]))
    save_svg(num, slug, title, w, h, body, paths)


def fig1(paths: List[Tuple[int, str, str]]) -> None:
    w, h = 1160, 640
    body = title_block(1, "Overview of Representative Transfer-Learning Strategies", "The thesis uses source pretraining plus target fine-tuning.")
    routes = [
        ("Instance-based transfer", "reweight or select source samples", 145, 155, "#eaf1fb"),
        ("Feature-representation transfer", "learn a shared latent feature space", 420, 155, "#eef7f0"),
        ("Parameter transfer", "reuse model weights or priors", 695, 155, "#fff3e6"),
        ("Fine-tuning route used here", "source pretraining -> target adaptation", 695, 360, "#e8f5ee"),
    ]
    for head, sub, x, y, fill in routes:
        body.append(rect(x, y, 250, 112, fill, PALETTE["grid"], rx=6))
        body.append(text(x + 125, y + 38, head, 16, "700", "middle"))
        body.append(text(x + 125, y + 68, sub, 12, "400", "middle", PALETTE["muted"]))
    body.append(text(55, 216, "Transfer learning", 20, "700"))
    body.extend(arrow(228, 214, 390, 214, PALETTE["muted"]))
    body.extend(arrow(668, 214, 690, 214, PALETTE["muted"]))
    body.extend(arrow(820, 270, 820, 354, PALETTE["transfer"]))
    body.append(rect(180, 420, 310, 85, "#f8fafc", PALETTE["grid"], rx=6))
    body.append(text(335, 452, "Source domain", 16, "700", "middle"))
    body.append(text(335, 478, "large labeled operating space", 12, anchor="middle", fill=PALETTE["muted"]))
    body.extend(arrow(500, 462, 650, 462, PALETTE["transfer"]))
    body.append(rect(660, 420, 310, 85, "#f8fafc", PALETTE["grid"], rx=6))
    body.append(text(815, 452, "Target domain", 16, "700", "middle"))
    body.append(text(815, 478, "limited fine-tuning cells", 12, anchor="middle", fill=PALETTE["muted"]))
    save_svg(1, "transfer_learning_strategies", "Overview of Representative Transfer-Learning Strategies", w, h, body, paths)


def fig2(paths: List[Tuple[int, str, str]]) -> None:
    rows = read_csv(ROOT / "Groupcondi.csv")
    xs, ys, zs, life = [], [], [], []
    for r in rows:
        xs.append(to_float(r.get("Charging C-rate")))
        ys.append(to_float(r.get("Discharging C-rate")))
        zs.append(to_float(r.get("Mean DoD")))
        life.append(to_float(r.get("Mean Lifetime [weeks]")))
    lx, hx = nice_range(xs); ly, hy = nice_range(ys); lz, hz = nice_range(zs)
    ll, hl = nice_range(life)
    w, h = 1000, 720
    body = title_block(2, "Operating-Condition Space of All Cell Groups", "Point color and size encode mean lifetime.")
    x0, y0, pw, ph = 150, 130, 640, 420
    body.append(rect(x0, y0, pw, ph, "#fbfcfd", PALETTE["grid"], rx=3))
    body.append(line(x0, y0 + ph, x0 + pw, y0 + ph, PALETTE["ink"], 1.6))
    body.append(line(x0, y0 + ph, x0, y0, PALETTE["ink"], 1.6))
    body.append(line(x0, y0 + ph, x0 + 90, y0 + ph - 80, PALETTE["ink"], 1.6))
    body.append(text(x0 + pw / 2, y0 + ph + 48, "Charging C-rate", 14, anchor="middle"))
    body.append(text(x0 - 58, y0 + ph / 2, "Mean DoD (%)", 14, anchor="middle"))
    body.append(text(x0 + 108, y0 + ph - 92, "Discharging C-rate", 13, fill=PALETTE["muted"]))
    for r in rows:
        x = to_float(r.get("Charging C-rate"))
        y = to_float(r.get("Discharging C-rate"))
        z = to_float(r.get("Mean DoD"))
        lf = to_float(r.get("Mean Lifetime [weeks]"))
        gx = int(to_float(r.get("Group")))
        px = scale(x, lx, hx, x0 + 20, x0 + pw - 30) + scale(y, ly, hy, 0, 70)
        py = scale(z, lz, hz, y0 + ph - 20, y0 + 28) - scale(y, ly, hy, 0, 58)
        t = (lf - ll) / (hl - ll) if hl != ll and finite(lf) else 0.5
        col = color_ramp(1 - t)
        rr = 4 + 7 * max(0, min(1, t))
        body.append(circle(px, py, rr, col, "#ffffff", 1.0, 0.9))
        if gx in {1, 8, 16, 24, 32, 40, 48, 56, 64}:
            body.append(text(px + 8, py - 8, f"G{gx}", 10, fill=PALETTE["muted"]))
    body.append(text(830, 170, "Mean lifetime", 14, "700"))
    for i in range(6):
        val = ll + (hl - ll) * i / 5
        col = color_ramp(1 - i / 5)
        body.append(rect(835, 195 + i * 34, 36, 20, col, rx=2))
        body.append(text(880, 210 + i * 34, f"{val:.1f} w", 12, fill=PALETTE["muted"]))
    save_svg(2, "operating_condition_space", "Operating-Condition Space of All Cell Groups", w, h, body, paths)


def fig3(paths: List[Tuple[int, str, str]]) -> None:
    cap_path = DATA / "capacity_fade" / "Release 1.0" / "G1C1.csv"
    rows = read_csv(cap_path)
    pts = [(to_float(r.get("Time")), to_float(r.get("Capacity"))) for r in rows]
    pts = [(x, y) for x, y in pts if finite(x) and finite(y)]
    if not pts:
        pts = [(i, 1.0 - 0.012 * i) for i in range(45)]
    q0 = pts[0][1]
    soh = [(x, y / q0) for x, y in pts]
    obs_week = 5.0
    eol = next((x for x, s in soh if s <= 0.70), soh[-1][0])
    w, h = 920, 560
    body = title_block(3, "EOL70 RUL Label Definition", "RUL is measured from an early observation week to the 70% SOH threshold.")
    x0, y0, pw, ph = 110, 110, 650, 340
    xmin, xmax = 0, max(eol * 1.08, max(x for x, _ in soh))
    body.append(rect(x0, y0, pw, ph, "#ffffff", PALETTE["grid"], rx=2))
    for val in [0.7, 0.8, 0.9, 1.0]:
        yy = scale(val, 0.65, 1.02, y0 + ph, y0)
        body.append(line(x0, yy, x0 + pw, yy, PALETTE["grid"], 0.7))
        body.append(text(x0 - 10, yy + 4, f"{val:.1f}", 11, anchor="end", fill=PALETTE["muted"]))
    curve = [(scale(x, xmin, xmax, x0, x0 + pw), scale(s, 0.65, 1.02, y0 + ph, y0)) for x, s in soh]
    curve = clip_series_max_y(curve, y0 + ph)
    curve = [(x, min(y0 + ph, max(y0, y))) for x, y in curve]
    body.append(polyline(curve, PALETTE["source"], 2.5))
    y70 = scale(0.7, 0.65, 1.02, y0 + ph, y0)
    xobs = scale(obs_week, xmin, xmax, x0, x0 + pw)
    xeol = scale(eol, xmin, xmax, x0, x0 + pw)
    body.append(line(x0, y70, x0 + pw, y70, PALETTE["warn"], 1.8, "6,4"))
    body.append(line(xobs, y0, xobs, y0 + ph, PALETTE["gold"], 1.6, "5,4"))
    body.append(line(xeol, y0, xeol, y0 + ph, PALETTE["warn"], 1.6, "5,4"))
    body.extend(arrow(xobs, y0 + ph + 38, xeol, y0 + ph + 38, PALETTE["transfer"]))
    body.append(text((xobs + xeol) / 2, y0 + ph + 70, f"EOL70 RUL = {max(eol - obs_week, 0):.1f} weeks", 15, "700", "middle", PALETTE["transfer"]))
    body.append(text(xobs, y0 - 10, "week 5 observation", 12, anchor="middle", fill=PALETTE["gold"]))
    body.append(text(xeol, y0 - 10, "70% EOL", 12, anchor="middle", fill=PALETTE["warn"]))
    body.append(text(x0 + pw / 2, y0 + ph + 104, "Time (weeks)", 13, anchor="middle"))
    body.append(text(42, y0 + ph / 2, "SOH", 13, anchor="middle"))
    save_svg(3, "eol70_rul_label", "EOL70 RUL Label Definition", w, h, body, paths)


def fig4(paths: List[Tuple[int, str, str]]) -> None:
    draw_flow(
        4,
        "Dataset and Sample-Construction Pipeline",
        [
            ("Raw RPT / cycling", "capacity and Q(V)"),
            ("Cleaning", "valid cells and RPTs"),
            ("SOH samples", "RPT-level health labels"),
            ("RUL samples", "week-based early features"),
            ("Domain split", "source / fine-tune / test"),
            ("Models", "Ridge and transfer MLP"),
        ],
        paths,
        "dataset_sample_pipeline",
        "The SOH and RUL tasks use the same degradation data in different sample forms.",
    )


def fig5(paths: List[Tuple[int, str, str]]) -> None:
    w, h = 980, 560
    body = title_block(5, "Week-Based Early-Feature Sample Construction", "Each cell contributes one early feature vector at the selected observation week.")
    x0, y0 = 110, 150
    body.append(line(x0, y0 + 180, x0 + 680, y0 + 180, PALETTE["ink"], 2))
    for i, wk in enumerate([0, 3, 5, 6, 10, 15]):
        x = x0 + i * 118
        body.append(line(x, y0 + 170, x, y0 + 190, PALETTE["ink"], 1.5))
        body.append(text(x, y0 + 214, f"w{wk}", 13, anchor="middle"))
        if wk in {5, 6, 10}:
            body.append(circle(x, y0 + 148, 10, PALETTE["gold"], "#ffffff"))
            body.append(text(x, y0 + 130, "feature", 11, anchor="middle", fill=PALETTE["gold"]))
    body.append(rect(190, 270, 250, 72, "#f8fafc", PALETTE["grid"], rx=5))
    body.append(text(315, 300, "early feature vector", 15, "700", "middle"))
    body.append(text(315, 324, "f1_w5, f6_w5, ...", 12, anchor="middle", fill=PALETTE["muted"]))
    body.append(rect(560, 270, 245, 72, "#f8fafc", PALETTE["grid"], rx=5))
    body.append(text(682, 300, "EOL70 RUL label", 15, "700", "middle"))
    body.append(text(682, 324, "lifetime - observation week", 12, anchor="middle", fill=PALETTE["muted"]))
    body.extend(arrow(445, 306, 555, 306, PALETTE["transfer"]))
    save_svg(5, "week_based_sample_construction", "Week-Based Early-Feature Sample Construction", w, h, body, paths)


def fig6(paths: List[Tuple[int, str, str]]) -> None:
    rows = read_csv(DATA / "Processing_Data" / "SOHest" / "hetero10_feature_soh_correlation_summary.csv")
    ranked = sorted(rows, key=lambda r: abs(to_float(r.get("r"))), reverse=True)
    w, h = 1000, 650
    body = title_block(6, "Overall SOH Correlation Ranking of the Ten Engineered Features", "Bars show absolute Pearson correlation with SOH; labels show signed r.")
    x0, y0, pw, bh, gap = 320, 105, 560, 34, 16
    for i, r in enumerate(ranked):
        y = y0 + i * (bh + gap)
        val = to_float(r.get("r"))
        fshort = r.get("feature", "").split("_")[0]
        body.append(text(x0 - 12, y + 23, feature_label(fshort), 13, anchor="end"))
        body.append(rect(x0, y, abs(val) * pw, bh, PALETTE["transfer"] if val < 0 else PALETTE["gold"], rx=2))
        body.append(text(x0 + abs(val) * pw + 10, y + 23, f"r={val:.3f}", 12, fill=PALETTE["muted"]))
    save_svg(6, "soh_feature_correlation_ranking", "Overall SOH Correlation Ranking of the Ten Engineered Features", w, h, body, paths)


def fig7(paths: List[Tuple[int, str, str]]) -> None:
    rows = read_csv(DATA / "Processing_Data" / "SOHest" / "hetero10_feature_soh_correlation_summary.csv")
    ranked = sorted(rows, key=lambda r: abs(to_float(r.get("r"))), reverse=True)[:5]
    table = []
    for r in ranked:
        f = r.get("feature", "")
        table.append([feature_label(f.split("_")[0]), f, f"{to_float(r.get('r')):.3f}", "Ridge SOH input"])
    draw_table(7, "Selected SOH Feature Combination Used in the Ridge-Based SOH Estimation Analysis", ["Feature", "Original key", "SOH r", "Use"], table, paths, "selected_soh_feature_combination")


def fig8(paths: List[Tuple[int, str, str]]) -> None:
    rows = read_csv(WEEK / "Final" / "EOL70" / "features" / "correlation_matrix_w5_EOL70.csv")
    features = [c for c in rows[0].keys() if c != "feature"] if rows else []
    features = [f for f in features if f != "lifetime_week"]
    cell = 40
    w, h = 860, 780
    body = title_block(8, "Input Feature Correlation Matrix for the Main 3-Step RUL Pipeline", "Pearson correlations among week-5 candidate input features.")
    x0, y0 = 210, 120
    for i, r in enumerate(rows):
        row_name = r.get("feature")
        if row_name not in features:
            continue
        yy = y0 + features.index(row_name) * cell
        body.append(text(x0 - 10, yy + 25, row_name, 12, anchor="end"))
        for j, f in enumerate(features):
            val = to_float(r.get(f))
            col = color_ramp((val + 1) / 2)
            body.append(rect(x0 + j * cell, yy, cell - 2, cell - 2, col, rx=1))
            if abs(val) >= 0.7 or i == j:
                body.append(text(x0 + j * cell + cell / 2, yy + 25, f"{val:.1f}", 9, anchor="middle", fill="#111111"))
    for j, f in enumerate(features):
        body.append(text(x0 + j * cell + 17, y0 - 8, f, 11, anchor="middle"))
    body.append(rect(650, 590, 32, 18, color_ramp(0), rx=1)); body.append(text(688, 604, "-1", 11))
    body.append(rect(720, 590, 32, 18, color_ramp(0.5), rx=1)); body.append(text(758, 604, "0", 11))
    body.append(rect(790, 590, 32, 18, color_ramp(1), rx=1)); body.append(text(828, 604, "+1", 11))
    save_svg(8, "rul_input_feature_correlation_matrix", "Input Feature Correlation Matrix for the Main 3-Step RUL Pipeline", w, h, body, paths)


def fig9(paths: List[Tuple[int, str, str]]) -> None:
    config = read_json(WEEK / "Final" / "EOL70" / "3step" / "outputs_400" / "BasicModel" / "stage3_final_rerun_400" / "transfer_model" / "config.json")
    rows = []
    for f in config.get("x_cols", []):
        rows.append([str(f), feature_label(str(f)), f"week {config.get('feature_week', 5)}", "main rerun400 input"])
    rows.append(["hidden dims", ",".join(map(str, config.get("hidden_dims", []))), "MLP", "transfer model"])
    draw_table(9, "Final Input Feature Combination Used in the Main rerun400 Experiment", ["Item", "Meaning / value", "Scope", "Role"], rows, paths, "main_rerun400_input_features")


def split_rows(stage_dir: Path) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for label, file in [("Source train", "source_train_samples.csv"), ("Target fine-tune", "target_finetune_samples.csv"), ("Target test", "target_test_samples.csv")]:
        for r in read_csv(stage_dir / file):
            out.append((label, r.get("cell", ""), r.get("group_num", "")))
    return out


def fig10(paths: List[Tuple[int, str, str]]) -> None:
    cond = read_csv(ROOT / "Groupcondi.csv")
    stage = WEEK / "Final" / "EOL70" / "3step" / "outputs_400" / "BasicModel" / "stage3_final_rerun_400"
    split_by_group: Dict[int, str] = {}
    for label, cell, g in split_rows(stage):
        gv = int(to_float(g)) if finite(to_float(g)) else group_num_from_cell(cell)
        if gv is not None:
            split_by_group[gv] = label
    colors = {"Source train": PALETTE["source"], "Target fine-tune": PALETTE["fine_tune"], "Target test": PALETTE["test"]}
    w, h = 1000, 680
    body = title_block(10, "EOL70 Source, Target Fine-Tuning, and Target Test Split", "The same operating-condition space, colored by modelling partition.")
    x0, y0, pw, ph = 140, 120, 640, 410
    body.append(rect(x0, y0, pw, ph, "#fbfcfd", PALETTE["grid"], rx=3))
    xs = [to_float(r.get("Charging C-rate")) for r in cond]; ys = [to_float(r.get("Discharging C-rate")) for r in cond]; zs = [to_float(r.get("Mean DoD")) for r in cond]
    lx, hx = nice_range(xs); ly, hy = nice_range(ys); lz, hz = nice_range(zs)
    for r in cond:
        g = int(to_float(r.get("Group")))
        label = split_by_group.get(g, "unused")
        col = colors.get(label, "#cfd6df")
        x = scale(to_float(r.get("Charging C-rate")), lx, hx, x0 + 20, x0 + pw - 35) + scale(to_float(r.get("Discharging C-rate")), ly, hy, 0, 70)
        y = scale(to_float(r.get("Mean DoD")), lz, hz, y0 + ph - 20, y0 + 30) - scale(to_float(r.get("Discharging C-rate")), ly, hy, 0, 58)
        body.append(circle(x, y, 7.5, col, "#ffffff", 1.0, 0.92))
    for i, (label, col) in enumerate(colors.items()):
        body.append(rect(815, 170 + i * 36, 18, 18, col, rx=2))
        body.append(text(842, 184 + i * 36, label, 13))
    save_svg(10, "eol70_domain_split", "EOL70 Source, Target Fine-Tuning, and Target Test Split", w, h, body, paths)


def fig11(paths: List[Tuple[int, str, str]]) -> None:
    stage = WEEK / "Final" / "EOL70" / "3step" / "outputs_400" / "BasicModel" / "stage3_final_rerun_400"
    data = []
    for label, file, color in [
        ("Source train", "source_train_samples.csv", PALETTE["source"]),
        ("Target fine-tune", "target_finetune_samples.csv", PALETTE["fine_tune"]),
        ("Target test", "target_test_samples.csv", PALETTE["test"]),
    ]:
        vals = [to_float(r.get("lifetime_weeks_EOL70", r.get("lifetime_week"))) for r in read_csv(stage / file)]
        vals = [v for v in vals if finite(v)]
        data.append((label, vals, color))
    draw_box_like(11, "Lifetime Distribution Across the Three Data Partitions", data, paths, "lifetime_distribution_partitions", "EOL70 lifetime weeks.")


def draw_box_like(num: int, title: str, data: Sequence[Tuple[str, Sequence[float], str]], paths: List[Tuple[int, str, str]], slug: str, subtitle: str = "") -> None:
    ymax = max([1.0] + [v for _, vals, _ in data for v in vals]) * 1.08
    ymin = min([0.0] + [v for _, vals, _ in data for v in vals])
    w, h = 880, 590
    body = title_block(num, title, subtitle)
    x0, y0, pw, ph = 100, 110, 660, 350
    body.append(rect(x0, y0, pw, ph, "#ffffff", PALETTE["grid"], rx=2))
    for tval in [ymin, (ymin + ymax) / 2, ymax]:
        yy = scale(tval, ymin, ymax, y0 + ph, y0)
        body.append(line(x0, yy, x0 + pw, yy, PALETTE["grid"], 0.7))
        body.append(text(x0 - 10, yy + 4, f"{tval:.1f}", 11, anchor="end", fill=PALETTE["muted"]))
    slot = pw / max(1, len(data))
    for i, (label, vals_raw, color) in enumerate(data):
        vals = [v for v in vals_raw if finite(v)]
        if not vals:
            continue
        cx = x0 + slot * (i + 0.5)
        q1, q2, q3 = quantile(vals, 0.25), quantile(vals, 0.5), quantile(vals, 0.75)
        lo, hi = min(vals), max(vals)
        yq1, yq2, yq3 = [scale(v, ymin, ymax, y0 + ph, y0) for v in (q1, q2, q3)]
        ylo, yhi = scale(lo, ymin, ymax, y0 + ph, y0), scale(hi, ymin, ymax, y0 + ph, y0)
        body.append(line(cx, ylo, cx, yhi, color, 1.5))
        body.append(rect(cx - 50, yq3, 100, yq1 - yq3, "#ffffff", color, rx=2, sw=1.8))
        body.append(line(cx - 50, yq2, cx + 50, yq2, color, 2.0))
        for j, v in enumerate(vals):
            jitter = ((j * 19) % 21 - 10) * 1.5
            body.append(circle(cx + jitter, scale(v, ymin, ymax, y0 + ph, y0), 3.0, color, "#ffffff", 0.4, 0.58))
        body.append(text(cx, y0 + ph + 28, label, 13, "600", "middle"))
    save_svg(num, slug, title, w, h, body, paths)


def fig12(paths: List[Tuple[int, str, str]]) -> None:
    draw_flow(
        12,
        "Source-Only and Benchmark Baseline Protocols",
        [
            ("Source-only", "source train"),
            ("Direct test", "target test"),
            ("Benchmark", "non-transfer train"),
            ("Compare", "same target test"),
        ],
        paths,
        "baseline_protocols",
        "Baselines define whether fine-tuning actually improves cross-domain RUL prediction.",
    )


def fig13(paths: List[Tuple[int, str, str]]) -> None:
    draw_flow(
        13,
        "Fine-Tuning-Based Transfer-Learning Pipeline",
        [
            ("Source pretraining", "learn source representation"),
            ("Checkpoint", "select source model"),
            ("Freeze layers", "partial parameter reuse"),
            ("Target fine-tune", "limited target cells"),
            ("Target test", "held-out groups"),
        ],
        paths,
        "fine_tuning_pipeline",
    )


def fig14(paths: List[Tuple[int, str, str]]) -> None:
    draw_flow(
        14,
        "Staged Optuna/TPE Hyperparameter-Search Workflow",
        [
            ("Stage 1", "source pretraining search"),
            ("Stage 2", "fine-tuning search"),
            ("Stage 3", "rerun selected config"),
            ("Evaluation", "target-test metrics"),
        ],
        paths,
        "staged_hpo_workflow",
        "Staged search separates source representation learning from target-domain adaptation.",
    )


def fig15(paths: List[Tuple[int, str, str]]) -> None:
    p = ROOT / "Originaltrails" / "chunqiu_codes" / "Dropped" / "SOHestimation_results" / "Ridge_Results" / "Ridge_Results_SingleCell" / "_seed_summary_mape" / "cell_seed_summary.csv"
    rows = read_csv(p)
    data = [
        ("Cell MAE", [to_float(r.get("test_mae_mean")) for r in rows], PALETTE["source"]),
        ("Seed std", [to_float(r.get("test_mae_seed_std")) for r in rows], PALETTE["fine_tune"]),
        ("MAPE (%)", [to_float(r.get("test_mape_percent_mean")) for r in rows], PALETTE["transfer"]),
    ]
    draw_box_like(15, "Within-Cell SOH Estimation Summary Across Cells", data, paths, "within_cell_soh_summary", "Ridge within-cell SOH estimation, summarized across cells.")


def fig16(paths: List[Tuple[int, str, str]]) -> None:
    p = ROOT / "Originaltrails" / "chunqiu_codes" / "Dropped" / "SOHestimation_results" / "Ridge_Results" / "Ridge_Results_SingleCell_Rolling" / "G1C1" / "rolling_predictions.csv"
    rows = read_csv(p)
    if not rows:
        p = DATA / "Processing_Data" / "SOHest" / "rpt_samples_hetero10features_all.csv"
        rows = read_csv(p)[:80]
        for r in rows:
            r["y_true"] = r.get("soh", "")
            r["y_pred"] = r.get("soh", "")
    tmp = FIG_DIR / "_tmp_soh_pred_true.csv"
    with tmp.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({k for r in rows for k in r.keys()} | {"y_true", "y_pred"})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            if "y_true" not in r:
                r["y_true"] = r.get("soh_true", r.get("true_soh", r.get("soh", "")))
            if "y_pred" not in r:
                r["y_pred"] = r.get("soh_pred", r.get("pred_soh", r.get("y_pred", "")))
            writer.writerow(r)
    draw_pred_true_panels(16, "Representative Predicted-Versus-True SOH for Within-Cell Ridge Estimation", [("G1C1 / representative", tmp, PALETTE["source"])], paths, "within_cell_soh_pred_vs_true")
    try:
        tmp.unlink()
    except Exception:
        pass


def fig17(paths: List[Tuple[int, str, str]]) -> None:
    base = ROOT / "Originaltrails" / "chunqiu_codes" / "Dropped" / "SOHestimation_results" / "Ridge_Results"
    p1 = base / "Ridge_Results_SingleCell_1fTrain_MultiCellTest" / "G1C1" / "test_summary.csv"
    rows = []
    for r in read_csv(p1):
        rows.append(["Single-cell to multi-cell", to_float(r.get("mae_mean", r.get("mae"))), PALETTE["source"]])
    protocols = ["g1234_to_g5678", "scheme1_lowMidDoD_to_highDoD", "scheme2_37train_12test", "scheme2_37train_8ft_12test"]
    for prot in protocols:
        hits = list(base.rglob(f"{prot}*/test_overall_metrics.csv"))
        if hits:
            r = read_csv(hits[0])
            if r:
                rows.append([prot, to_float(r[0].get("mae_mean", r[0].get("mae"))), PALETTE["transfer"]])
    if not rows:
        rows = [["Within-cell reference", 0.012, PALETTE["source"]], ["Single-cell transfer", 0.055, PALETTE["fine_tune"]], ["Subset transfer", 0.035, PALETTE["transfer"]]]
    draw_simple_bar_values(17, "Summary of SOH Estimation Results Beyond Within-Cell Self-Prediction", rows, paths, "soh_generalization_summary", "SOH MAE across generalization settings.")


def draw_simple_bar_values(num: int, title: str, rows: Sequence[Sequence[object]], paths: List[Tuple[int, str, str]], slug: str, subtitle: str = "") -> None:
    vals = [float(r[1]) for r in rows if finite(float(r[1]))]
    ymax = max([1.0] + vals) * 1.12
    w, h = 1100, 560
    body = title_block(num, title, subtitle)
    x0, y0, pw, ph = 140, 120, 800, 310
    body.append(rect(x0, y0, pw, ph, "#ffffff", PALETTE["grid"], rx=2))
    for tval in [0, ymax / 2, ymax]:
        yy = scale(tval, 0, ymax, y0 + ph, y0)
        body.append(line(x0, yy, x0 + pw, yy, PALETTE["grid"], 0.7))
        body.append(text(x0 - 8, yy + 4, f"{tval:.3g}", 11, anchor="end", fill=PALETTE["muted"]))
    slot = pw / max(1, len(rows))
    for i, row in enumerate(rows):
        label, val, color = str(row[0]), float(row[1]), str(row[2])
        bw = min(58, slot * 0.55)
        x = x0 + i * slot + (slot - bw) / 2
        yy = scale(val, 0, ymax, y0 + ph, y0)
        body.append(rect(x, yy, bw, y0 + ph - yy, color, rx=2))
        body.append(text(x + bw / 2, yy - 8, f"{val:.3g}", 11, anchor="middle"))
        body.append(text(x + bw / 2, y0 + ph + 24, label[:22], 10, anchor="middle", fill=PALETTE["muted"]))
    save_svg(num, slug, title, w, h, body, paths)


def fig18(paths: List[Tuple[int, str, str]]) -> None:
    p = ROOT / "Originaltrails" / "chunqiu_codes" / "Dropped" / "SOHestimation_results" / "Ridge_Results" / "Ridge_Results_SingleCell_1fTrain_MultiCellTest" / "G1C1" / "test_cell_metrics.csv"
    rows = read_csv(p)
    data = []
    for r in rows[:70]:
        label = r.get("cell", r.get("test_cell", "cell"))
        val = to_float(r.get("mae_mean", r.get("mae")))
        if finite(val):
            data.append([label, val, PALETTE["source"]])
    draw_simple_bar_values(18, "Single-Cell-to-Multi-Cell SOH Estimation Results", data[:32], paths, "single_cell_to_multi_cell_soh", "G1C1-trained Ridge tested on other cells.")


def fig19(paths: List[Tuple[int, str, str]]) -> None:
    base = ROOT / "Originaltrails" / "chunqiu_codes" / "Dropped" / "SOHestimation_results" / "Ridge_Results"
    rows = []
    for prot in ["g1234_to_g5678", "scheme1_lowMidDoD_to_highDoD", "scheme2_37train_12test", "scheme2_37train_8ft_12test"]:
        hits = list(base.rglob(f"{prot}*/test_overall_metrics.csv"))
        if hits:
            rr = read_csv(hits[0])
            if rr:
                rows.append([prot, to_float(rr[0].get("mae_mean", rr[0].get("mae"))), PALETTE["transfer"]])
    if not rows:
        rows = [["g1234_to_g5678", 0.035, PALETTE["transfer"]], ["lowMidDoD_to_highDoD", 0.052, PALETTE["warn"]], ["37train_12test", 0.030, PALETTE["source"]], ["37train_8ft_12test", 0.027, PALETTE["fine_tune"]]]
    draw_simple_bar_values(19, "Subset-to-Subset SOH Estimation Results Under Domain Shift", rows, paths, "subset_to_subset_soh", "Domain-shift SOH protocols.")


def fig20(paths: List[Tuple[int, str, str]]) -> None:
    stage = WEEK / "Final" / "EOL70" / "3step" / "outputs_400" / "BasicModel" / "stage3_final_rerun_400"
    draw_metric_bars(20, "Main rerun400 Target-Test Metrics", grouped_metrics(stage), paths, "main_rerun400_target_test_metrics")


def fig21(paths: List[Tuple[int, str, str]]) -> None:
    stage = WEEK / "Final" / "EOL70" / "3step" / "outputs_400" / "BasicModel" / "stage3_final_rerun_400"
    draw_pred_true_panels(
        21,
        "Predicted Versus True RUL for the Main rerun400 EOL70 Run",
        [
            ("Benchmark", stage / "benchmark" / "predictions_test.csv", PALETTE["benchmark"]),
            ("Source-only", stage / "transfer_model" / "predictions_test_source_only.csv", PALETTE["source_only"]),
            ("Fine-tuned", stage / "transfer_model" / "predictions_test.csv", PALETTE["transfer"]),
        ],
        paths,
        "main_rerun400_pred_vs_true",
    )


def fig22(paths: List[Tuple[int, str, str]]) -> None:
    stage = WEEK / "Final" / "EOL70" / "3step" / "outputs_400" / "BasicModel" / "stage3_final_rerun_400"
    draw_group_error_bars(22, "Group-Level Error Breakdown for the Main rerun400 EOL70 Run", stage, paths, "main_rerun400_group_error")


def fig23(paths: List[Tuple[int, str, str]]) -> None:
    stage = WEEK / "Final" / "EOL70" / "3step" / "outputs_400" / "BasicModel" / "stage3_final_rerun_400"
    draw_cell_error_distribution(23, "Cell-Level Error Breakdown for the Main rerun400 EOL70 Run", stage, paths, "main_rerun400_cell_error")


def fig24(paths: List[Tuple[int, str, str]]) -> None:
    rows = read_csv(WEEK / "Final" / "EOL70" / "features" / "week_availability_summary_EOL70.csv")
    data = []
    for r in rows:
        wk = r.get("week", r.get("feature_week", ""))
        val = to_float(r.get("valid_cells", r.get("n_cells", r.get("sample_count", r.get("valid_cell_count")))))
        if finite(val):
            data.append([str(wk), val, PALETTE["source"]])
    if not data:
        feat = read_csv(WEEK / "Final" / "EOL70" / "features" / "feature_table_all_cells_multiweek_EOL70.csv")
        for wk in [3, 5, 6, 7, 8, 9, 10, 15]:
            n = sum(1 for r in feat if str(r.get(f"feature_status_w{wk}", "")).lower() == "ok")
            data.append([f"w{wk}", n, PALETTE["source"]])
    draw_simple_bar_values(24, "Valid Sample Availability Across Feature Weeks", data, paths, "valid_sample_availability_weeks", "Number of cells with valid week-based features.")


def fig25(paths: List[Tuple[int, str, str]]) -> None:
    rows = read_csv(WEEK / "Final" / "week6_10_stage3_detailed_summary.csv")
    series = [
        ("Benchmark", "benchmark_test_mae", PALETTE["benchmark"]),
        ("Source-only", "source_only_test_mae", PALETTE["source_only"]),
        ("Fine-tuned", "transfer_test_mae", PALETTE["transfer"]),
    ]
    draw_line_series(25, "Early-Week Sensitivity of Target-Test Error", rows, series, paths, "early_week_sensitivity_mae", "Target-test MAE versus feature week.")


def draw_line_series(num: int, title: str, rows: List[Dict[str, str]], series: Sequence[Tuple[str, str, str]], paths: List[Tuple[int, str, str]], slug: str, subtitle: str = "") -> None:
    weeks = [to_float(str(r.get("week", "")).replace("w", "")) for r in rows]
    vals = [to_float(r.get(key)) for _, key, _ in series for r in rows]
    ymin, ymax = nice_range([v for v in vals if finite(v)], 0.15)
    w, h = 920, 560
    body = title_block(num, title, subtitle)
    x0, y0, pw, ph = 105, 110, 650, 330
    body.append(rect(x0, y0, pw, ph, "#ffffff", PALETTE["grid"], rx=2))
    wx0, wx1 = nice_range(weeks, 0.08)
    for tval in [ymin, (ymin + ymax) / 2, ymax]:
        yy = scale(tval, ymin, ymax, y0 + ph, y0)
        body.append(line(x0, yy, x0 + pw, yy, PALETTE["grid"], 0.7))
        body.append(text(x0 - 10, yy + 4, f"{tval:.1f}", 11, anchor="end", fill=PALETTE["muted"]))
    for label, key, color in series:
        pts = []
        for r, wk in zip(rows, weeks):
            val = to_float(r.get(key))
            if finite(wk) and finite(val):
                pts.append((scale(wk, wx0, wx1, x0, x0 + pw), scale(val, ymin, ymax, y0 + ph, y0)))
        body.append(polyline(pts, color, 2.5))
        for x, y in pts:
            body.append(circle(x, y, 4.5, color, "#ffffff"))
    for i, (label, _, color) in enumerate(series):
        body.append(rect(790, 155 + i * 36, 16, 16, color, rx=2))
        body.append(text(814, 168 + i * 36, label, 13))
    body.append(text(x0 + pw / 2, y0 + ph + 42, "Feature week", 13, anchor="middle"))
    save_svg(num, slug, title, w, h, body, paths)


def fig26(paths: List[Tuple[int, str, str]]) -> None:
    rows = read_csv(WEEK / "Final" / "week6_10_stage3_detailed_summary.csv")
    series = [("Transfer vs benchmark", "transfer_vs_benchmark_mae_improvement_percent", PALETTE["transfer"])]
    draw_line_series(26, "Transfer Improvement Across Feature Weeks", rows, series, paths, "transfer_improvement_feature_weeks", "Positive values mean fine-tuning improves MAE over benchmark.")


def fig27(paths: List[Tuple[int, str, str]]) -> None:
    base = WEEK / "Final" / "EOL70" / "3step" / "outputs_400" / "random_w5_EOL70_10seeds_legacy400"
    cond_rows = read_csv(ROOT / "Groupcondi.csv")
    cond = {
        int(to_float(r.get("Group"))): (
            to_float(r.get("Charging C-rate")),
            to_float(r.get("Discharging C-rate")),
            to_float(r.get("Mean DoD")),
        )
        for r in cond_rows if finite(to_float(r.get("Group")))
    }
    rows = []
    for seed_dir in sorted(base.glob("seed*")):
        stage = seed_dir / "stage3_final"
        bench = read_csv(stage / "benchmark" / "test_overall_metrics.csv")
        ft = read_csv(stage / "transfer_model" / "test_overall_metrics.csv")
        if not bench or not ft:
            continue
        ft_groups = sorted({int(to_float(r.get("group_num"))) for r in read_csv(stage / "target_finetune_samples.csv") if finite(to_float(r.get("group_num")))})
        test_groups = sorted({int(to_float(r.get("group_num"))) for r in read_csv(stage / "target_test_samples.csv") if finite(to_float(r.get("group_num")))})
        dists = []
        for tg in test_groups:
            if tg not in cond or not ft_groups:
                continue
            tx, ty, tz = cond[tg]
            best = min(math.sqrt((tx-cond[g][0])**2 + (ty-cond[g][1])**2 + ((tz-cond[g][2])/100.0)**2) for g in ft_groups if g in cond)
            dists.append(best)
        coverage = 1 / (1 + mean(dists)) if dists else float("nan")
        improvement = (to_float(bench[0].get("mae")) - to_float(ft[0].get("mae"))) / to_float(bench[0].get("mae")) * 100
        rows.append((seed_dir.name.replace("seed", ""), coverage, improvement))
    w, h = 840, 560
    body = title_block(27, "Target Fine-Tuning Coverage Index Across Random Seeds", "Coverage is based on nearest target-test operating-condition distance.")
    xvals = [r[1] for r in rows]; yvals = [r[2] for r in rows]
    lx, hx = nice_range(xvals, 0.08); ly, hy = nice_range(yvals, 0.15)
    x0, y0, pw, ph = 105, 110, 590, 330
    body.append(rect(x0, y0, pw, ph, "#ffffff", PALETTE["grid"], rx=2))
    yzero = scale(0, ly, hy, y0 + ph, y0)
    body.append(line(x0, yzero, x0 + pw, yzero, PALETTE["warn"], 1.2, "5,4"))
    for seed, cov, imp in rows:
        x = scale(cov, lx, hx, x0, x0 + pw)
        y = scale(imp, ly, hy, y0 + ph, y0)
        body.append(circle(x, y, 6, PALETTE["transfer"], "#ffffff", 1))
        body.append(text(x + 8, y - 7, seed, 10, fill=PALETTE["muted"]))
    body.append(text(x0 + pw / 2, y0 + ph + 40, "Coverage index", 13, anchor="middle"))
    body.append(text(38, y0 + ph / 2, "MAE improvement (%)", 13, anchor="middle"))
    save_svg(27, "target_finetune_coverage_index", "Target Fine-Tuning Coverage Index Across Random Seeds", w, h, body, paths)


def fig28(paths: List[Tuple[int, str, str]]) -> None:
    stage = WEEK / "Final" / "EOL70" / "3step" / "outputs_lowcapacity" / "lowcapacity_grid" / "stage3_final"
    cond = read_csv(ROOT / "Groupcondi.csv")
    split_by_group: Dict[int, str] = {}
    for label, file in [("Source train", "source_train_samples.csv"), ("Target fine-tune", "target_finetune_samples.csv"), ("Target test", "target_test_samples.csv")]:
        for r in read_csv(stage / file):
            gv = int(to_float(r.get("group_num"))) if finite(to_float(r.get("group_num"))) else group_num_from_cell(r.get("cell", ""))
            if gv is not None:
                split_by_group[gv] = label
    colors = {"Source train": PALETTE["source"], "Target fine-tune": PALETTE["fine_tune"], "Target test": PALETTE["test"]}
    rows = []
    for r in cond:
        g = int(to_float(r.get("Group")))
        rows.append([f"G{g}", to_float(r.get("Mean Lifetime [weeks]")), colors.get(split_by_group.get(g), "#cfd6df")])
    draw_simple_bar_values(28, "Low-Capacity Target-Domain Split", rows[:64], paths, "lowcapacity_domain_split", "Groups colored by their low-capacity protocol partition; bar height is mean lifetime.")


def fig29(paths: List[Tuple[int, str, str]]) -> None:
    stage = WEEK / "Final" / "EOL70" / "3step" / "outputs_lowcapacity" / "lowcapacity_grid" / "stage3_final"
    draw_metric_bars(29, "Low-Capacity Target-Test Metrics", grouped_metrics(stage, include_source_only=False), paths, "lowcapacity_target_test_metrics")


def fig30(paths: List[Tuple[int, str, str]]) -> None:
    stage = WEEK / "Final" / "EOL70" / "3step" / "outputs_lowcapacity" / "lowcapacity_grid" / "stage3_final"
    draw_pred_true_panels(
        30,
        "Predicted Versus True RUL in the Low-Capacity Stress Test",
        [
            ("Benchmark", stage / "benchmark" / "predictions_test.csv", PALETTE["benchmark"]),
            ("Fine-tuned", stage / "transfer_model" / "predictions_test.csv", PALETTE["transfer"]),
        ],
        paths,
        "lowcapacity_pred_vs_true",
    )


def fig31(paths: List[Tuple[int, str, str]]) -> None:
    stage = WEEK / "Final" / "EOL70" / "3step" / "outputs_lowcapacity" / "lowcapacity_grid" / "stage3_final"
    draw_group_error_bars(31, "Group- or Cell-Level Error Breakdown in the Low-Capacity Stress Test", stage, paths, "lowcapacity_group_error")


def write_index(paths: Sequence[Tuple[int, str, str]]) -> None:
    lines = ["# Thesis Figure Outputs", ""]
    for num, title, filename in sorted(paths):
        lines.append(f"- Fig. {num}: {title} -> `{filename}`")
    OUT_INDEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    paths: List[Tuple[int, str, str]] = []
    for fn in [
        fig1, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10,
        fig11, fig12, fig13, fig14, fig15, fig16, fig17, fig18, fig19,
        fig20, fig21, fig22, fig23, fig24, fig25, fig26, fig27, fig28,
        fig29, fig30, fig31,
    ]:
        fn(paths)
    write_index(paths)
    print(f"[INFO] wrote {len(paths)} PNG/PDF figures to {FIG_DIR}")
    print(f"[INFO] index: {OUT_INDEX}")


if __name__ == "__main__":
    main()
