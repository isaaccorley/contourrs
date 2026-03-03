"""Generate plots for README.md."""

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from contourrs import contours_arrow, shapes_arrow
from matplotlib.colors import ListedColormap

# ── Polygonize: raster vs vector ────────────────────────────────────────

rng = np.random.default_rng(42)
raster = rng.integers(1, 5, size=(128, 128), dtype=np.uint8)
cmap = ListedColormap(["#2d6a4f", "#52b788", "#d4a373", "#e9c46a"])

gdf = gpd.GeoDataFrame.from_arrow(shapes_arrow(raster, connectivity=4))
color_map = {1.0: "#2d6a4f", 2.0: "#52b788", 3.0: "#d4a373", 4.0: "#e9c46a"}
gdf["color"] = gdf["value"].map(color_map)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(raster, cmap=cmap, interpolation="nearest")
axes[0].set_title("Input raster (4 classes)")
axes[0].set_axis_off()
gdf.plot(ax=axes[1], color=gdf["color"], edgecolor="black", linewidth=0.2)
axes[1].set_title(f"Vector polygons ({len(gdf)} features)")
axes[1].set_axis_off()
plt.tight_layout()
fig.savefig("assets/polygonize.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved assets/polygonize.png")

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
axes[0].imshow(dem, cmap="terrain", interpolation="bilinear")
axes[0].set_title("Continuous raster (synthetic DEM)")
axes[0].set_axis_off()
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
plt.tight_layout()
fig.savefig("assets/contours.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved assets/contours.png")
