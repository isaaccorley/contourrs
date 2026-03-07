# Limitations vs GDAL

contourrs reimplements GDAL's polygonize algorithm in pure Rust but is not a full replacement for the GDAL ecosystem.

## Comparison

| Feature | GDAL | contourrs |
|---|---|---|
| Large rasters | Scanline-based, disk-backed | Full array must fit in memory |
| File I/O | Reads any GDAL/OGR raster/vector format | Array-in, GeoJSON/Arrow out |
| CRS propagation | Automatic from source dataset | Manual — caller attaches CRS |
| Nodata handling | Auto from band metadata / mask band | Explicit bool mask required |
| Simplification | Ecosystem tools (`ogr2ogr`, PostGIS) | None — bring your own (e.g. `shapely.simplify`) |
| Dtypes | int8–64, uint8–64, float, complex | u8/16/32, i16/32, f32/64 |
| Progress reporting | Callback API | None |

## When to use GDAL instead

- **Streaming very large rasters** that exceed available RAM — GDAL processes scanline-by-scanline with a configurable block cache
- **Direct format conversion** (e.g. GeoTIFF → GeoPackage) without intermediate array representation
- **CRS-aware output layers** where the spatial reference needs to transfer automatically from source to destination

## When contourrs wins

- **In-memory ML/CV pipelines** where speed matters and the raster is already a numpy array
- **Arrow/GeoParquet-first workflows** — zero-copy handoff to PyArrow with low Python-heap allocation; total native/process memory is still meaningful
- **Environments where installing GDAL is painful** — contourrs is a single `pip install` with no system dependencies
- **Isoband contouring from arrays** — `contourrs` exposes filled isobands directly from in-memory arrays; GDAL's `gdal_contour` CLI is centered on isolines rather than this API shape

## What's the same

Both GDAL and contourrs:

- Use connected-component labeling + boundary tracing (same general algorithmic approach)
- Support 4- and 8-connectivity
- Use exact equality for pixel value grouping (no tolerance-based merging)
- Match rasterio/GDAL on the discrete polygonize cases covered by our [integration tests](https://github.com/isaaccorley/contourrs/blob/main/tests/test_compare_rasterio.py)
