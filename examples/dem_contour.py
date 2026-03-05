"""Contour a real DEM (Mount Rainier) and a synthetic DEM, saving plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.figure
import numpy as np
import rasterio
from contourrs import contours_arrow

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIRS = (Path("assets"), Path("docs/assets"))
DEFAULT_DEM = Path("examples/data/mt_rainier_dem_2048.tif")


def _save_figure(fig: matplotlib.figure.Figure, name: str, dpi: int = 150) -> None:
    for d in OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {path}")


def _align_to_raster_grid(ax, width: int, height: int) -> None:
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")


# ---------------------------------------------------------------------------
# Synthetic DEM
# ---------------------------------------------------------------------------


def make_synthetic_dem(size: int = 256) -> np.ndarray:
    """Multi-peak Gaussian surface — deterministic, no download."""
    y, x = np.mgrid[-3 : 3 : complex(size), -3 : 3 : complex(size)]
    z = (
        np.exp(-(x**2 + y**2))
        + 0.7 * np.exp(-((x - 1.5) ** 2 + (y - 1) ** 2) / 0.5)
        + 0.5 * np.exp(-((x + 1.5) ** 2 + (y + 1.5) ** 2) / 0.8)
    )
    return z.astype(np.float32)


def choose_histogram_thresholds(
    data: np.ndarray,
    *,
    band_count: int = 8,
    sample_size: int = 200_000,
    seed: int = 42,
) -> list[float]:
    """Choose thresholds from histogram peaks via 1D k-means clustering."""
    finite = data[np.isfinite(data)].astype(np.float64)
    if finite.size == 0:
        msg = "DEM has no finite values"
        raise ValueError(msg)

    if finite.size > sample_size:
        rng = np.random.default_rng(seed)
        sample = rng.choice(finite, size=sample_size, replace=False)
    else:
        sample = finite

    k = min(max(2, band_count), sample.size)
    centroids = np.quantile(sample, np.linspace(0.0, 1.0, k))

    for _ in range(30):
        distances = np.abs(sample[:, None] - centroids[None, :])
        labels = np.argmin(distances, axis=1)
        updated = np.array(
            [
                sample[labels == i].mean() if np.any(labels == i) else centroids[i]
                for i in range(k)
            ]
        )
        if np.allclose(updated, centroids, atol=1e-3):
            break
        centroids = updated

    centroids = np.sort(centroids)
    thresholds = [float(finite.min())]
    thresholds.extend(
        float((centroids[i] + centroids[i + 1]) / 2.0)
        for i in range(len(centroids) - 1)
    )
    thresholds.append(float(finite.max()))

    thresholds = sorted(set(thresholds))
    if len(thresholds) < 2:
        thresholds = [float(finite.min()), float(finite.max())]
    return thresholds


def plot_synthetic_dem(
    dem: np.ndarray,
    thresholds: list[float],
    *,
    save_name: str = "contours_synthetic.png",
) -> None:
    threshold_values = [float(value) for value in thresholds]
    gdf = gpd.GeoDataFrame.from_arrow(contours_arrow(dem, thresholds=threshold_values))
    h, w = dem.shape

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(dem, cmap="terrain", interpolation="bilinear", extent=(0, w, h, 0))
    axes[0].set_title("Synthetic DEM")
    axes[0].set_axis_off()
    _align_to_raster_grid(axes[0], w, h)

    gdf.plot(
        ax=axes[1],
        column="value",
        cmap="terrain",
        edgecolor="black",
        linewidth=0.3,
        legend=True,
        legend_kwds={"label": "Threshold (m)", "shrink": 0.8},
    )
    axes[1].set_title(f"Isoband contours ({len(gdf)} polygons)")
    axes[1].set_axis_off()
    _align_to_raster_grid(axes[1], w, h)

    plt.tight_layout()
    _save_figure(fig, save_name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Real DEM (Mount Rainier)
# ---------------------------------------------------------------------------


def plot_real_dem(
    dem_path: Path,
    thresholds: list[float] | None = None,
    *,
    band_count: int = 8,
    save_name: str = "contours_mt_rainier.png",
) -> None:
    with rasterio.open(dem_path) as src:
        data = src.read(1)
        transform_obj = src.transform
        bounds = src.bounds

    transform = (
        transform_obj.a,
        transform_obj.b,
        transform_obj.c,
        transform_obj.d,
        transform_obj.e,
        transform_obj.f,
    )

    mask = np.isfinite(data)
    if thresholds is None:
        threshold_values = choose_histogram_thresholds(data, band_count=band_count)
    else:
        threshold_values = [float(value) for value in thresholds]

    print(
        "Real DEM thresholds (m): "
        + ", ".join(f"{value:.0f}" for value in threshold_values)
    )

    gdf = gpd.GeoDataFrame.from_arrow(
        contours_arrow(
            data,
            thresholds=threshold_values,
            mask=mask,
            transform=transform,
        )
    )

    h, w = data.shape
    print(
        f"Real DEM: {w}x{h}, range {np.nanmin(data):.0f}-{np.nanmax(data):.0f} m, "
        f"{len(gdf)} contour polygons"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    axes[0].imshow(
        data,
        cmap="terrain",
        interpolation="bilinear",
        extent=extent,
    )
    axes[0].set_title("Mount Rainier DEM (USGS 3DEP 1/3″)")
    axes[0].set_axis_off()
    axes[0].set_xlim(bounds.left, bounds.right)
    axes[0].set_ylim(bounds.bottom, bounds.top)
    axes[0].set_aspect("equal")

    gdf.plot(
        ax=axes[1],
        column="value",
        cmap="terrain",
        edgecolor="black",
        linewidth=0.15,
        legend=True,
        legend_kwds={"label": "Elevation (m)", "shrink": 0.8},
    )
    axes[1].set_title(f"Isoband contours ({len(gdf)} polygons)")
    axes[1].set_axis_off()
    axes[1].set_xlim(bounds.left, bounds.right)
    axes[1].set_ylim(bounds.bottom, bounds.top)
    axes[1].set_aspect("equal")

    plt.tight_layout()
    _save_figure(fig, save_name, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Contour synthetic + real DEMs, save plots")
    p.add_argument(
        "--dem",
        type=Path,
        default=DEFAULT_DEM,
        help="Path to real DEM GeoTIFF",
    )
    p.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Skip real DEM, only generate synthetic plot",
    )
    p.add_argument(
        "--bands",
        type=int,
        default=8,
        help="Number of real-DEM contour bands for auto threshold selection",
    )
    p.add_argument(
        "--real-thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Optional explicit thresholds for real DEM (overrides --bands)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Synthetic
    dem = make_synthetic_dem(256)
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    plot_synthetic_dem(dem, thresholds)
    print()

    # Real
    if args.synthetic_only:
        print("Skipping real DEM (--synthetic-only)")
        return

    if not args.dem.exists():
        print(f"DEM not found: {args.dem}")
        print("Run the download first or pass --synthetic-only")
        return

    plot_real_dem(
        args.dem,
        thresholds=args.real_thresholds,
        band_count=args.bands,
    )


if __name__ == "__main__":
    main()
