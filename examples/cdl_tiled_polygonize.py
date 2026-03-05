"""Download a USDA CDL tile and polygonize it in blocks."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, urlretrieve

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
import shapely
from contourrs import shapes_arrow
from rasterio.windows import Window

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CDL_GET_FILE_URL = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a sample USDA CDL county tile, polygonize in tiles, "
            "then merge touching polygons by class."
        )
    )
    parser.add_argument("--year", type=int, default=2023, help="CDL year")
    parser.add_argument(
        "--fips",
        default="19153",
        help="County FIPS code (default: 19153 = Polk County, IA)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size in pixels for block polygonization",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=4,
        help="Pixel connectivity passed to contourrs.shapes_arrow",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("examples/data"),
        help="Directory for downloaded sample raster",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output GeoParquet path "
            "(default: examples/output/cdl_<year>_<fips>_merged.parquet)"
        ),
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate a raster-vs-polygons comparison plot (default: on)",
    )
    return parser.parse_args()


def fetch_cdl_url(*, year: int, fips: str) -> str:
    request_url = f"{CDL_GET_FILE_URL}?{urlencode({'year': year, 'fips': fips})}"
    with urlopen(request_url, timeout=60) as response:  # noqa: S310
        payload = response.read().decode("utf-8", errors="replace")

    fault_match = re.search(r"<faultstring>([^<]+)</faultstring>", payload)
    if fault_match:
        message = fault_match.group(1).strip()
        raise RuntimeError(message)

    return_url_match = re.search(r"<returnURL>([^<]+)</returnURL>", payload)
    if return_url_match is None:
        message = "Could not parse return URL from CDL service response"
        raise RuntimeError(message)
    return return_url_match.group(1).strip()


def download_if_missing(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"Using cached raster: {path}")
        return

    print(f"Downloading sample CDL tile: {url}")
    urlretrieve(url, path)  # noqa: S310
    print(f"Saved raster: {path}")


def iter_windows(height: int, width: int, tile_size: int):
    for row_off in range(0, height, tile_size):
        for col_off in range(0, width, tile_size):
            yield Window.from_slices(
                (row_off, min(row_off + tile_size, height)),
                (col_off, min(col_off + tile_size, width)),
            )


def polygonize_tiled(
    raster_path: Path,
    *,
    tile_size: int,
    connectivity: int,
) -> gpd.GeoDataFrame:
    value_chunks: list[np.ndarray] = []
    geometry_parts: list[object] = []

    with rasterio.open(raster_path) as src:
        total_tiles = ((src.height + tile_size - 1) // tile_size) * (
            (src.width + tile_size - 1) // tile_size
        )
        print(
            f"Raster {src.width}x{src.height}, tile_size={tile_size}, "
            f"tiles={total_tiles}"
        )

        for tile_index, window in enumerate(
            iter_windows(src.height, src.width, tile_size),
            start=1,
        ):
            block = src.read(1, window=window)
            mask = np.ones(block.shape, dtype=bool)
            if src.nodata is not None:
                mask &= block != src.nodata
            mask &= block != 0

            if not mask.any():
                continue

            transform = src.window_transform(window)
            table = shapes_arrow(
                block,
                mask=mask,
                connectivity=connectivity,
                transform=(
                    transform.a,
                    transform.b,
                    transform.c,
                    transform.d,
                    transform.e,
                    transform.f,
                ),
            )
            if table.num_rows == 0:
                continue

            tile_gdf = gpd.GeoDataFrame.from_arrow(table)
            value_chunks.append(tile_gdf["value"].to_numpy(dtype=np.int32, copy=False))
            geometry_parts.extend(tile_gdf.geometry.array)

            print(f"  tile {tile_index}/{total_tiles}: +{len(tile_gdf)} polygons")

        if not geometry_parts:
            return gpd.GeoDataFrame(columns=["value", "geometry"], geometry="geometry")

        values = np.concatenate(value_chunks)
        return gpd.GeoDataFrame(
            {"value": values, "geometry": geometry_parts},
            geometry="geometry",
            crs=src.crs,
        )


def merge_touching_same_class(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf

    values = gdf["value"].to_numpy(dtype=np.int32, copy=False)
    geometries = np.asarray(gdf.geometry.array, dtype=object)

    merged_values: list[int] = []
    merged_geometries: list[object] = []
    unique_values = np.unique(values)

    for value in unique_values:
        class_geometries = geometries[values == value]
        dissolved = shapely.union_all(class_geometries)
        for part in shapely.get_parts(dissolved):
            if part.is_empty or part.geom_type != "Polygon":
                continue
            merged_values.append(int(value))
            merged_geometries.append(part)

    if not merged_geometries:
        return gpd.GeoDataFrame(columns=["value", "geometry"], geometry="geometry")

    return gpd.GeoDataFrame(
        {
            "value": np.asarray(merged_values, dtype=np.int32),
            "geometry": merged_geometries,
        },
        geometry="geometry",
        crs=gdf.crs,
    )


OUTPUT_DIRS = (Path("assets"), Path("docs/assets"))


def summarize_area_by_class(gdf: gpd.GeoDataFrame) -> dict[int, float]:
    if gdf.empty:
        return {}

    values = gdf["value"].to_numpy(dtype=np.int32, copy=False)
    areas = gdf.geometry.area.to_numpy(dtype=np.float64, copy=False)
    unique_values, inverse = np.unique(values, return_inverse=True)
    area_sums = np.bincount(inverse, weights=areas)
    return {
        int(value): float(area)
        for value, area in zip(unique_values, area_sums, strict=False)
    }


def _save_figure(fig, name: str, dpi: int = 150) -> None:
    for d in OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {path}")


def plot_cdl_result(
    raster_path: Path,
    merged: gpd.GeoDataFrame,
    *,
    save_name: str = "cdl_polygonize.png",
) -> None:
    """Side-by-side: CDL raster vs merged polygons."""
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        bounds = src.bounds

    h, w = data.shape

    # Build a random but deterministic colormap keyed on class value
    classes = sorted(merged["value"].unique())
    rng = np.random.default_rng(42)
    palette = {int(c): rng.random(3) for c in classes}
    colors = [palette[int(v)] for v in merged["value"]]

    # Raster colormap: map pixel values to same palette
    raster_rgb = np.zeros((*data.shape, 3), dtype=np.float32)
    for cls, rgb in palette.items():
        raster_rgb[data == cls] = rgb

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

    axes[0].imshow(raster_rgb, interpolation="nearest", extent=extent)
    axes[0].set_title(f"CDL raster ({w}x{h})")
    axes[0].set_axis_off()
    axes[0].set_xlim(bounds.left, bounds.right)
    axes[0].set_ylim(bounds.bottom, bounds.top)
    axes[0].set_aspect("equal")

    merged.plot(ax=axes[1], color=colors, edgecolor="black", linewidth=0.05)
    axes[1].set_title(
        f"Merged polygons ({len(merged):,} features, {len(classes)} classes)"
    )
    axes[1].set_axis_off()
    axes[1].set_xlim(bounds.left, bounds.right)
    axes[1].set_ylim(bounds.bottom, bounds.top)
    axes[1].set_aspect("equal")

    plt.tight_layout()
    _save_figure(fig, save_name)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_path = args.output or Path(
        f"examples/output/cdl_{args.year}_{args.fips}_merged.parquet"
    )
    raster_path = args.download_dir / f"cdl_{args.year}_{args.fips}.tif"

    cdl_url = fetch_cdl_url(year=args.year, fips=args.fips)
    download_if_missing(cdl_url, raster_path)

    tiled = polygonize_tiled(
        raster_path,
        tile_size=args.tile_size,
        connectivity=args.connectivity,
    )
    merged = merge_touching_same_class(tiled)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)

    print(f"Raw tiled polygons: {len(tiled):,}")
    print(f"Merged polygons: {len(merged):,}")
    print(f"Classes in output: {merged['value'].nunique():,}")

    tiled_area = summarize_area_by_class(tiled)
    merged_area = summarize_area_by_class(merged)
    all_classes = set(tiled_area) | set(merged_area)
    if all_classes:
        max_class_area_delta = max(
            abs(tiled_area.get(value, 0.0) - merged_area.get(value, 0.0))
            for value in all_classes
        )
    else:
        max_class_area_delta = 0.0
    print(f"Max per-class area delta after merge: {max_class_area_delta:.6f}")
    print(f"Saved merged polygons: {output_path}")

    if args.plot:
        print()
        plot_cdl_result(raster_path, merged)


if __name__ == "__main__":
    main()
