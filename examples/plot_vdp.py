#!/usr/bin/env python3
"""Publication-ready plot for VDP clustering outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def configure_publication_style() -> None:
    # Compact publication-friendly defaults.
    plt.rcParams.update(
        {
            "figure.figsize": (3.5, 2.6),
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_points(points_csv: Path):
    xs = []
    ys = []
    clusters = []
    with points_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            clusters.append(int(row["cluster"]))
    return xs, ys, clusters


def load_centers(centers_csv: Path):
    cx = []
    cy = []
    with centers_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cx.append(float(row["mean_x"]))
            cy.append(float(row["mean_y"]))
    return cx, cy


def plot(points_csv: Path, centers_csv: Path, out_path: Path) -> None:
    configure_publication_style()

    xs, ys, clusters = load_points(points_csv)
    cx, cy = load_centers(centers_csv)

    fig, ax = plt.subplots()

    cmap = plt.get_cmap("tab10")
    unique_clusters = sorted(set(clusters))

    for i, c in enumerate(unique_clusters):
        px = [x for x, cc in zip(xs, clusters) if cc == c]
        py = [y for y, cc in zip(ys, clusters) if cc == c]
        ax.scatter(
            px,
            py,
            s=10,
            alpha=0.78,
            color=cmap(i % 10),
            edgecolors="none",
            label=f"Cluster {c}",
            rasterized=True,
        )

    ax.scatter(
        cx,
        cy,
        s=42,
        marker="X",
        color="black",
        linewidths=0.6,
        label="Posterior mean",
        zorder=5,
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("VDP Clustering (VCTK)")
    ax.grid(True)

    # Compact legend for publication layout.
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    fig.tight_layout(pad=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--points",
        type=Path,
        default=Path("build/examples/vdp_points.csv"),
        help="CSV with columns: x,y,cluster,max_prob",
    )
    parser.add_argument(
        "--centers",
        type=Path,
        default=Path("build/examples/vdp_centers.csv"),
        help="CSV with columns: cluster,mean_x,mean_y,weight",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("build/examples/vdp_plot.pdf"),
        help="Output figure path (.pdf/.png)",
    )
    args = parser.parse_args()

    plot(args.points, args.centers, args.out)
    print(f"Wrote figure: {args.out}")


if __name__ == "__main__":
    main()
