"""Generate synthetic and real DEM example plots."""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.figure
import numpy as np
import rasterio
from contourrs import contours_arrow, shapes_arrow
from matplotlib.colors import Normalize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.figstyle import apply_style, export

apply_style()

OUTPUT_DIRS = (Path("assets"), Path("docs/assets"))
DEFAULT_DEM = Path("examples/data/mt_rainier_dem_2048.tif")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--real-thresholds", type=float, nargs="+", default=None)
    return parser.parse_args()


def save_figure(fig: matplotlib.figure.Figure, name: str, dpi: int = 300) -> None:
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / name
        fig.savefig(out_path, dpi=dpi)
        export(fig, out_path.with_suffix(""))
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
    norm = Normalize(0, 1.1)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.2))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.23, top=0.9, wspace=0.06)
    axes[0].imshow(
        dem,
        cmap="terrain",
        norm=norm,
        interpolation="nearest",
        extent=(-0.5, w - 0.5, h - 0.5, -0.5),
    )
    axes[0].set_title("(a) Synthetic elevation field")
    colors = plt.get_cmap("terrain")(norm(gdf["value"].to_numpy() + 0.1))
    gdf.plot(
        ax=axes[1], rasterized=True, color=colors, edgecolor="black", linewidth=0.3
    )
    axes[1].set_title("(b) Interpolated contour bands")
    for ax in axes:
        ax.set_axis_off()
        ax.set(xlim=(0, w - 1), ylim=(h - 1, 0), aspect="equal")
    cax = fig.add_axes((0.16, 0.115, 0.68, 0.032))
    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="terrain"),
        cax=cax,
        orientation="horizontal",
    )
    bar.set_label("Synthetic elevation (arbitrary units)")
    save_figure(fig, "contours_synthetic.png")
    plt.close(fig)


def plot_real(dem_path: Path, bands: int, thresholds: list[float] | None) -> None:
    with rasterio.open(dem_path) as src:
        data = src.read(1, masked=True).filled(np.nan)
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

    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.2))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.23, top=0.9, wspace=0.06)
    continuous_norm = Normalize(threshold_values[0], threshold_values[-1])
    axes[0].imshow(
        data,
        cmap="terrain",
        norm=continuous_norm,
        interpolation="nearest",
        extent=extent,
    )
    axes[0].set_title("(a) Mount Rainier elevation")
    style_geo_axes(axes[0], bounds)

    midpoints = (
        np.asarray(threshold_values[:-1]) + np.asarray(threshold_values[1:])
    ) / 2
    colors = plt.get_cmap("terrain")(continuous_norm(midpoints[gdf["band"].to_numpy()]))
    gdf.plot(
        ax=axes[1], rasterized=True, color=colors, edgecolor="black", linewidth=0.025
    )
    axes[1].set_title(f"(b) Elevation bins ({len(gdf):,} regions)")
    style_geo_axes(axes[1], bounds)

    scalar = plt.cm.ScalarMappable(norm=continuous_norm, cmap="terrain")
    scalar.set_array([])
    cax = fig.add_axes((0.16, 0.115, 0.68, 0.032))
    colorbar = fig.colorbar(scalar, cax=cax, orientation="horizontal")
    colorbar.set_label("Elevation (m)")

    save_figure(fig, "contours_mt_rainier.png")
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
