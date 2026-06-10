from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "electric-vehicle-battery-demand-by-mode-2018-2024.csv"
PNG_PATH = BASE_DIR / "electric_vehicle_battery_demand_by_mode_2018_2024.png"
SVG_PATH = BASE_DIR / "electric_vehicle_battery_demand_by_mode_2018_2024.svg"


def main() -> None:
    df = pd.read_csv(CSV_PATH, sep=";", skiprows=3, index_col=0)
    df.index = df.index.astype(int)

    categories = ["LDV", "Two/three-wheeler", "Bus", "Truck"]
    colors = {
        "LDV": "#4CC9ED",
        "Two/three-wheeler": "#437DD8",
        "Bus": "#55EA85",
        "Truck": "#18B9AE",
    }

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(8.6, 6.2), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = df.index.to_numpy()
    bottom = None
    for category in categories:
        ax.bar(
            x,
            df[category].to_numpy(),
            bottom=bottom,
            width=0.62,
            color=colors[category],
            edgecolor="#333333",
            linewidth=0.75,
            label=category,
            zorder=3,
        )
        bottom = df[category].to_numpy() if bottom is None else bottom + df[category].to_numpy()

    ax.set_title(
        "Electric vehicle battery demand by mode",
        fontsize=22,
        fontweight="normal",
        color="#333333",
        pad=26,
    )
    ax.set_ylabel("GWh/year", fontsize=17, fontweight="normal", color="#333333", labelpad=14)
    ax.set_xlabel("Year", fontsize=17, fontweight="normal", color="#333333", labelpad=10)

    ax.set_xlim(2017.35, 2024.65)
    ax.set_ylim(0, 1100)
    ax.set_xticks(x)
    ax.set_yticks(range(0, 1101, 200))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}" if value else "0"))

    ax.grid(axis="y", color="#E7E7E7", linewidth=1.1, zorder=0)
    ax.grid(axis="x", visible=False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(2.8)
    ax.spines["bottom"].set_linewidth(2.8)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    ax.tick_params(axis="x", labelsize=16, width=2.2, length=6, colors="#333333", rotation=0)
    ax.tick_params(axis="y", labelsize=16, width=2.2, length=6, colors="#333333")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("normal")

    totals = df[categories].sum(axis=1)
    peak_year = int(totals.idxmax())
    peak_value = int(totals.max())
    ax.annotate(
        f"{peak_value:,}",
        xy=(peak_year, peak_value),
        xytext=(peak_year - 0.75, peak_value + 70),
        ha="right",
        va="center",
        fontsize=21,
        fontweight="bold",
        color="#7DE0F7",
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 2.8,
            "color": "#7DE0F7",
            "shrinkA": 5,
            "shrinkB": 8,
            "mutation_scale": 16,
        },
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[category],
            markeredgecolor="#333333",
            markeredgewidth=0.6,
            markersize=8,
            label=category,
        )
        for category in categories
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.13, 0.03),
        ncol=4,
        frameon=False,
        fontsize=12,
        handlelength=0.8,
        handletextpad=0.5,
        columnspacing=1.6,
    )

    fig.text(
        0.97,
        0.045,
        "Source: IEA. Licence: CC BY 4.0",
        ha="right",
        va="center",
        fontsize=9.5,
        color="#666666",
    )

    fig.subplots_adjust(left=0.13, right=0.97, top=0.83, bottom=0.23)
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
