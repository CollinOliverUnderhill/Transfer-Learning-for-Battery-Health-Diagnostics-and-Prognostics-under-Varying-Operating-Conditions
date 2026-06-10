#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 1 only.

This file keeps the figure-specific drawing logic local so the figure can be
modified without touching the batch script. Shared SVG/data helpers are imported
from plot_all_thesis_figures.py.
"""

from plot_all_thesis_figures import *  # noqa: F401,F403


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


def main() -> None:
    paths = []
    fig1(paths)
    for _, title, filename in paths:
        print(f"[INFO] {title} -> {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
