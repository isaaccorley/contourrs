"""Generate reproducible synthetic documentation figures at 5.5 inches wide."""

from itertools import pairwise
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
from contourrs import contours_arrow, shapes_arrow
from figstyle import COLORS, WARM, apply_style, export
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRS = (ROOT / "assets", ROOT / "docs/assets")


def save_figure(fig, name: str) -> None:
    for directory in OUTPUT_DIRS:
        export(fig, directory / name)
        fig.savefig(directory / f"{name}.png", dpi=300)
    plt.close(fig)


def spatial_axes(axes, width: int, height: int) -> None:
    for ax in axes:
        ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
        ax.set_axis_off()


def polygonize_figure() -> None:
    rng = np.random.default_rng(42)
    raster = rng.integers(1, 5, size=(128, 128), dtype=np.uint8)
    colors = [COLORS[key] for key in ("coral", "blue", "green", "ochre")]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(0.5, 5.5), cmap.N)
    frame = gpd.GeoDataFrame.from_arrow(shapes_arrow(raster, connectivity=4))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.05))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.15, top=0.9, wspace=0.06)
    axes[0].imshow(
        raster, cmap=cmap, norm=norm, interpolation="nearest", extent=(0, 128, 128, 0)
    )
    frame.plot(
        ax=axes[1],
        rasterized=True,
        column="value",
        cmap=cmap,
        norm=norm,
        edgecolor=COLORS["ink"],
        linewidth=0.08,
    )
    axes[0].set_title("(a) Categorical raster")
    axes[1].set_title(f"(b) Polygons ({len(frame):,} regions)")
    spatial_axes(axes, 128, 128)
    fig.legend(
        handles=[
            Patch(facecolor=color, label=f"Class {i}")
            for i, color in enumerate(colors, 1)
        ],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.02),
    )
    save_figure(fig, "polygonize")


def contour_figure() -> None:
    y, x = np.mgrid[-3:3:128j, -3:3:128j]
    dem = (
        np.exp(-(x**2 + y**2))
        + 0.7 * np.exp(-((x - 1.5) ** 2 + (y - 1) ** 2) / 0.5)
        + 0.5 * np.exp(-((x + 1.5) ** 2 + (y + 1.5) ** 2) / 0.8)
    ).astype(np.float32)
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    frame = gpd.GeoDataFrame.from_arrow(contours_arrow(dem, thresholds=thresholds))
    norm = Normalize(0, 1.1)
    # Band midpoints use the same scale as the continuous input, not a rescaled palette.
    midpoint = {lo: (lo + hi) / 2 for lo, hi in pairwise(thresholds)}
    colors = [WARM(norm(midpoint[float(value)])) for value in frame["value"]]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.25))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.23, top=0.9, wspace=0.06)
    # Marching squares samples sit at integer coordinates.
    axes[0].imshow(
        dem,
        cmap=WARM,
        norm=norm,
        interpolation="nearest",
        extent=(-0.5, 127.5, 127.5, -0.5),
    )
    frame.plot(
        ax=axes[1],
        rasterized=True,
        color=colors,
        edgecolor=COLORS["ink"],
        linewidth=0.3,
    )
    axes[0].set_title("(a) Synthetic elevation field")
    axes[1].set_title("(b) Interpolated contour bands")
    spatial_axes(axes, 127, 127)
    cax = fig.add_axes((0.16, 0.115, 0.68, 0.032))
    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=WARM),
        cax=cax,
        orientation="horizontal",
        ticks=[0, 0.3, 0.6, 0.9, 1.1],
    )
    bar.set_label("Synthetic elevation (arbitrary units)")
    save_figure(fig, "contours")


def main() -> None:
    print(f"Figure font: {apply_style()}")
    polygonize_figure()
    contour_figure()


if __name__ == "__main__":
    main()
