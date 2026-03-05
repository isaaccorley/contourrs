"""Download a USDA CDL tile and polygonize it in blocks."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, urlretrieve

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from contourrs import shapes_arrow
from rasterio.windows import Window

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
    frames: list[gpd.GeoDataFrame] = []

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
            tile_gdf["value"] = tile_gdf["value"].astype("int32")
            tile_subset = gpd.GeoDataFrame(
                {
                    "value": tile_gdf["value"],
                    "geometry": tile_gdf.geometry,
                },
                geometry="geometry",
                crs=tile_gdf.crs,
            )
            frames.append(tile_subset)

            print(f"  tile {tile_index}/{total_tiles}: +{len(tile_gdf)} polygons")

        if not frames:
            return gpd.GeoDataFrame(columns=["value", "geometry"], geometry="geometry")

        return gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True),
            geometry="geometry",
            crs=src.crs,
        )


def merge_touching_same_class(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf

    merged = gdf.dissolve(by="value", as_index=False)
    merged = merged.explode(index_parts=False, ignore_index=True)
    merged = merged.loc[merged.geometry.notna() & ~merged.geometry.is_empty]
    merged["value"] = merged["value"].astype("int32")
    return merged[["value", "geometry"]].reset_index(drop=True)


def summarize_area_by_class(gdf: gpd.GeoDataFrame) -> pd.Series:
    if gdf.empty:
        return pd.Series(dtype="float64")
    area_series = pd.Series(gdf.geometry.area.to_numpy())
    value_series = pd.Series(gdf["value"].to_numpy())
    return area_series.groupby(value_series).sum()


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
    tiled_area, merged_area = tiled_area.align(merged_area, fill_value=0.0)
    max_class_area_delta = (tiled_area - merged_area).abs().max()
    print(f"Max per-class area delta after merge: {max_class_area_delta:.6f}")
    print(f"Saved merged polygons: {output_path}")


if __name__ == "__main__":
    main()
