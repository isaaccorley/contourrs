"""Download USDA CDL, polygonize in tiles, merge classes, and plot."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, urlretrieve

import geopandas as gpd
import matplotlib
import pandas as pd
import rasterio
from contourrs import shapes_arrow
from rasterio.windows import Window

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CDL_GET_FILE_URL = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"
OUTPUT_DIRS = (Path("assets"), Path("docs/assets"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--fips", default="19153", help="Polk County, IA")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=4)
    parser.add_argument("--download-dir", type=Path, default=Path("examples/data"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def resolve_cdl_url(year: int, fips: str) -> str:
    request_url = f"{CDL_GET_FILE_URL}?{urlencode({'year': year, 'fips': fips})}"
    with urlopen(request_url, timeout=60) as response:  # noqa: S310
        payload = response.read().decode("utf-8", errors="replace")

    fault_match = re.search(r"<faultstring>([^<]+)</faultstring>", payload)
    if fault_match:
        raise RuntimeError(fault_match.group(1).strip())

    return_url_match = re.search(r"<returnURL>([^<]+)</returnURL>", payload)
    if return_url_match is None:
        message = "Could not parse return URL from CDL response"
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


def transform_tuple(transform) -> tuple[float, float, float, float, float, float]:
    return (
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
    )


def polygonize_tiled(raster_path: Path, tile_size: int, connectivity: int):
    frames = []

    with rasterio.open(raster_path) as src:
        rows = range(0, src.height, tile_size)
        cols = range(0, src.width, tile_size)
        total_tiles = len(rows) * len(cols)
        print(
            f"Raster {src.width}x{src.height}, "
            f"tile_size={tile_size}, tiles={total_tiles}"
        )

        tile_index = 0
        for row_off in rows:
            for col_off in cols:
                tile_index += 1
                window = Window.from_slices(
                    (row_off, min(row_off + tile_size, src.height)),
                    (col_off, min(col_off + tile_size, src.width)),
                )
                block = src.read(1, window=window)
                mask = block != 0
                if src.nodata is not None:
                    mask &= block != src.nodata
                if not mask.any():
                    continue

                table = shapes_arrow(
                    block,
                    mask=mask,
                    connectivity=connectivity,
                    transform=transform_tuple(src.window_transform(window)),
                )
                if table.num_rows == 0:
                    continue

                tile_gdf = gpd.GeoDataFrame.from_arrow(table)
                tile_gdf = gpd.GeoDataFrame(
                    {"value": tile_gdf["value"], "geometry": tile_gdf.geometry},
                    geometry="geometry",
                    crs=tile_gdf.crs,
                )
                frames.append(tile_gdf)
                print(f"  tile {tile_index}/{total_tiles}: +{len(tile_gdf)} polygons")

        if not frames:
            return gpd.GeoDataFrame(columns=["value", "geometry"], geometry="geometry")

        gdf = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True),
            geometry="geometry",
        )
        gdf["value"] = gdf["value"].astype("int32")
        gdf.set_crs(src.crs, inplace=True)
        return gdf


def merge_touching_same_class(gdf: gpd.GeoDataFrame):
    if gdf.empty:
        return gdf

    merged = gdf.dissolve(by="value", as_index=False).explode(
        index_parts=False,
        ignore_index=True,
    )
    merged = merged[~merged.geometry.is_empty]
    merged["value"] = merged["value"].astype("int32")
    return merged[["value", "geometry"]]


def class_area(gdf: gpd.GeoDataFrame):
    return gdf.assign(area=gdf.geometry.area).groupby("value")["area"].sum()


def save_figure(fig, name: str, dpi: int = 140) -> None:
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / name
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {out_path}")


def plot_result(raster_path: Path, merged: gpd.GeoDataFrame) -> None:
    with rasterio.open(raster_path) as src:
        raster = src.read(1)
        bounds = src.bounds

    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(raster, cmap="tab20", interpolation="nearest", extent=extent)
    axes[0].set_title("CDL raster")

    merged.plot(
        ax=axes[1],
        column="value",
        cmap="tab20",
        edgecolor="black",
        linewidth=0.03,
        legend=False,
    )
    axes[1].set_title(f"Merged polygons ({len(merged):,})")

    for ax in axes:
        ax.set_axis_off()
        ax.set_xlim(bounds.left, bounds.right)
        ax.set_ylim(bounds.bottom, bounds.top)
        ax.set_aspect("equal")

    plt.tight_layout()
    save_figure(fig, "cdl_polygonize.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_path = args.output or Path(
        f"examples/output/cdl_{args.year}_{args.fips}_merged.parquet"
    )
    raster_path = args.download_dir / f"cdl_{args.year}_{args.fips}.tif"

    cdl_url = resolve_cdl_url(args.year, args.fips)
    download_if_missing(cdl_url, raster_path)

    tiled = polygonize_tiled(raster_path, args.tile_size, args.connectivity)
    merged = merge_touching_same_class(tiled)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)

    tiled_area = class_area(tiled)
    merged_area = class_area(merged)
    tiled_area, merged_area = tiled_area.align(merged_area, fill_value=0.0)
    max_delta = (tiled_area - merged_area).abs().max()

    print(f"Raw tiled polygons: {len(tiled):,}")
    print(f"Merged polygons: {len(merged):,}")
    print(f"Classes in output: {merged['value'].nunique():,}")
    print(f"Max per-class area delta after merge: {max_delta:.6f}")
    print(f"Saved merged polygons: {output_path}")

    if args.plot:
        print()
        plot_result(raster_path, merged)


if __name__ == "__main__":
    main()
