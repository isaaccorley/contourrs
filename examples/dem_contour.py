"""Generate synthetic and real DEM example plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.figure
import numpy as np
import rasterio
from contourrs import contours_arrow, shapes_arrow
from matplotlib.colors import BoundaryNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIRS = (Path("assets"), Path("docs/assets"))
DEFAULT_DEM = Path("examples/data/mt_rainier_dem_2048.tif")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--real-thresholds", type=float, nargs="+", default=None)
    return parser.parse_args()


def save_figure(fig: matplotlib.figure.Figure, name: str, dpi: int = 150) -> None:
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / name
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {out_path}")


def transform_tuple(transform) -> tuple[float, float, float, float, float, float]:
    return (
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
    )


def quantile_thresholds(data: np.ndarray, bands: int) -> list[float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        message = "DEM has no finite values"
        raise ValueError(message)
    breaks = np.quantile(finite, np.linspace(0.0, 1.0, max(2, bands) + 1))
    return sorted({float(value) for value in breaks})


def style_geo_axes(ax, bounds) -> None:
    ax.set_axis_off()
    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    ax.set_aspect("equal")


def synthetic_dem(size: int = 256) -> np.ndarray:
    y, x = np.mgrid[-3 : 3 : complex(size), -3 : 3 : complex(size)]
    dem = (
        np.exp(-(x**2 + y**2))
        + 0.7 * np.exp(-((x - 1.5) ** 2 + (y - 1) ** 2) / 0.5)
        + 0.5 * np.exp(-((x + 1.5) ** 2 + (y + 1.5) ** 2) / 0.8)
    )
    return dem.astype(np.float32)


def plot_synthetic() -> None:
    dem = synthetic_dem(256)
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    gdf = gpd.GeoDataFrame.from_arrow(contours_arrow(dem, thresholds=thresholds))

    h, w = dem.shape
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(dem, cmap="terrain", interpolation="bilinear", extent=(0, w, h, 0))
    axes[0].set_title("Synthetic DEM")
    axes[0].set_axis_off()
    axes[0].set_xlim(0, w)
    axes[0].set_ylim(h, 0)
    axes[0].set_aspect("equal")

    gdf.plot(
        ax=axes[1],
        column="value",
        cmap="terrain",
        edgecolor="black",
        linewidth=0.3,
        legend=True,
        legend_kwds={"label": "Threshold", "shrink": 0.8},
    )
    axes[1].set_title(f"Isobands ({len(gdf)} polygons)")
    axes[1].set_axis_off()
    axes[1].set_xlim(0, w)
    axes[1].set_ylim(h, 0)
    axes[1].set_aspect("equal")

    plt.tight_layout()
    save_figure(fig, "contours_synthetic.png")
    plt.close(fig)


def plot_real(dem_path: Path, bands: int, thresholds: list[float] | None) -> None:
    with rasterio.open(dem_path) as src:
        data = src.read(1)
        bounds = src.bounds
        transform = transform_tuple(src.transform)

    threshold_values = thresholds or quantile_thresholds(data, bands)
    band_count = len(threshold_values) - 1
    bins = np.digitize(data, threshold_values[1:-1], right=False).astype(np.int32)
    mask = np.isfinite(data)

    gdf = gpd.GeoDataFrame.from_arrow(
        shapes_arrow(bins, mask=mask, connectivity=4, transform=transform)
    )
    gdf["band"] = gdf["value"].astype(np.int32)
    gdf["band"] = np.clip(gdf["band"], 0, band_count - 1)

    print("Real DEM thresholds (m): " + ", ".join(f"{v:.0f}" for v in threshold_values))
    print(
        f"Real DEM: {data.shape[1]}x{data.shape[0]}, "
        f"range {np.nanmin(data):.0f}-{np.nanmax(data):.0f} m, "
        f"{len(gdf)} polygons"
    )

    cmap = plt.get_cmap("terrain", band_count)
    norm = BoundaryNorm(threshold_values, cmap.N, clip=True)
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(data, cmap="terrain", interpolation="bilinear", extent=extent)
    axes[0].set_title("Mount Rainier DEM (USGS 3DEP)")
    style_geo_axes(axes[0], bounds)

    gdf.plot(ax=axes[1], column="band", cmap=cmap, edgecolor="black", linewidth=0.05)
    axes[1].set_title(f"Elevation bins ({len(gdf)} polygons)")
    style_geo_axes(axes[1], bounds)

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=axes[1], shrink=0.8)
    colorbar.set_label("Elevation (m)")

    plt.tight_layout()
    save_figure(fig, "contours_mt_rainier.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_synthetic()
    print()

    if args.synthetic_only:
        print("Skipping real DEM (--synthetic-only)")
        return

    if not args.dem.exists():
        print(f"DEM not found: {args.dem}")
        return

    plot_real(args.dem, args.bands, args.real_thresholds)


if __name__ == "__main__":
    main()
