"""Generate plots for README.md."""

from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
from contourrs import contours_arrow, shapes_arrow
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIRS = (Path("assets"), Path("docs/assets"))


def _save_figure(fig, name: str) -> None:
    for output_dir in OUTPUT_DIRS:
        output_path = output_dir / name
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved {output_path}")


def _align_to_raster_grid(ax, width: int, height: int) -> None:
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")


# ── Polygonize: raster vs vector ────────────────────────────────────────

rng = np.random.default_rng(42)
raster = rng.integers(1, 5, size=(128, 128), dtype=np.uint8)
cmap = ListedColormap(["#2d6a4f", "#52b788", "#d4a373", "#e9c46a"])

gdf = gpd.GeoDataFrame.from_arrow(shapes_arrow(raster, connectivity=4))
color_map = {1.0: "#2d6a4f", 2.0: "#52b788", 3.0: "#d4a373", 4.0: "#e9c46a"}
gdf["color"] = [color_map[float(value)] for value in gdf["value"]]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
height, width = raster.shape
axes[0].imshow(
    raster,
    cmap=cmap,
    interpolation="nearest",
    extent=(0, width, height, 0),
)
axes[0].set_title("Input raster (4 classes)")
axes[0].set_axis_off()
_align_to_raster_grid(axes[0], width, height)
gdf.plot(ax=axes[1], color=gdf["color"], edgecolor="black", linewidth=0.2)
axes[1].set_title(f"Vector polygons ({len(gdf)} features)")
axes[1].set_axis_off()
_align_to_raster_grid(axes[1], width, height)
plt.tight_layout()
_save_figure(fig, "polygonize.png")
plt.close(fig)

# ── Contours: DEM vs isobands ───────────────────────────────────────────

y, x = np.mgrid[-3:3:128j, -3:3:128j]
dem = (
    np.exp(-(x**2 + y**2))
    + 0.7 * np.exp(-((x - 1.5) ** 2 + (y - 1) ** 2) / 0.5)
    + 0.5 * np.exp(-((x + 1.5) ** 2 + (y + 1.5) ** 2) / 0.8)
).astype(np.float32)

thresholds = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
gdf_contour = gpd.GeoDataFrame.from_arrow(contours_arrow(dem, thresholds=thresholds))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
height, width = dem.shape
axes[0].imshow(
    dem,
    cmap="terrain",
    interpolation="bilinear",
    extent=(0, width, height, 0),
)
axes[0].set_title("Continuous raster (synthetic DEM)")
axes[0].set_axis_off()
_align_to_raster_grid(axes[0], width, height)
gdf_contour.plot(
    ax=axes[1],
    column="value",
    cmap="terrain",
    edgecolor="black",
    linewidth=0.3,
    legend=True,
    legend_kwds={"label": "Band threshold", "shrink": 0.8},
)
axes[1].set_title(f"Isoband contours ({len(gdf_contour)} polygons)")
axes[1].set_axis_off()
_align_to_raster_grid(axes[1], width, height)
plt.tight_layout()
_save_figure(fig, "contours.png")
plt.close(fig)
