# Architecture

## Workspace layout

```
contourrs           Pure Rust library, returns geo_types::Polygon<f64>
contourrs-python    PyO3/maturin Python bindings
```

## Polygonize pipeline

Polygonization uses two passes, as in GDAL's `GDALPolygonize`.

### Region labeling

The first pass labels connected components using union-find with path compression.
It accepts 4- or 8-connectivity and an optional mask.

### Boundary tracing

The second pass traces each labeled region, normalizes its rings, and applies the affine transform to the output coordinates.

## Contours pipeline

Marching squares extracts the region above each threshold.
The algorithm assembles the resulting rings into polygons, including interior holes.
For each band `[lo, hi)`, a polygon difference subtracts the upper-threshold region from the lower-threshold region.
This preserves basins and handles boundaries that meet the raster edge.
Interpolation along cell edges places boundaries between raster samples, and the affine transform is applied after the polygon difference.

## Memory and computation

- Marching squares processes rows with Rayon when the grid contains at least 128x128 cells.
- Adjacent bands reuse the previous upper-threshold polygons as their lower-threshold polygons, so each threshold is traced once.
- Connected-component labeling allocates union-find entries for provisional regions rather than reserving an entry for every pixel.
- Hole assignment uses an R-tree to find candidate exterior rings when a threshold produces more than 16 exterior rings.
- Contouring borrows contiguous float64 input; other input dtypes require a float64 conversion buffer.
- Arrow output avoids Python geometry dictionaries but still allocates WKB and native geometry buffers.
- Polygon complexity affects runtime and memory. Noisy rasters can produce many small rings, so a smooth DEM and a random raster of the same size can have very different costs.

The [performance measurements](performance.md) describe an earlier implementation and dependency environment.
Rerun the benchmark before using those values to estimate the current implementation's cost.

## Feature flags (Rust crate)

| Flag | Default | Description |
|---|---|---|
| `arrow` | off (on in Python bindings) | Arrow RecordBatch export with WKB geometry + GeoParquet metadata |
