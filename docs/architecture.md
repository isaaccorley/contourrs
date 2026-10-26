# Architecture

## Workspace layout

```
contourrs           Pure Rust library, returns geo_types::Polygon<f64>
contourrs-python    PyO3/maturin Python bindings
```

The Python crate separates NumPy validation and dtype dispatch (`input`) from
GeoJSON and Arrow conversion (`output`). Both algorithms and output formats use
the same typed input path. NumPy borrow guards stay alive while the Rust core runs
without the GIL.

## Rust input validation

`RasterGrid::try_new(data, width, height)` checks the buffer length and dimension
multiplication, returning `RasterShapeError` on invalid input. `RasterGrid::new`
retains the panicking convenience API. `get_checked(col, row)` returns `None` for
out-of-range coordinates; `get(col, row)` panics in both debug and release builds.
Existing public fields remain available, so callers constructing grids directly
must preserve the documented dimension and buffer invariants.

`RasterValue::to_f64_slice` converts samples for interpolation. Its default
implementation allocates a float64 buffer; the float64 implementation borrows the
original slice. Custom raster value implementations need only provide
`to_f64_value` as before.

## Polygonize pipeline

Polygonization uses two passes, as in GDAL's `GDALPolygonize`.

### Region labeling

The first pass labels connected components using union-find with path compression.
It accepts 4- or 8-connectivity and an optional mask.

### Boundary tracing

The second pass traces each labeled region and splits boundaries at repeated
integer pixel vertices. It transforms these simple rings, assigns holes to
shells, and normalizes their winding. Splitting before coordinate conversion
preserves corner contacts under rotated or reflected affine transforms.
The tracer owns the visited-edge state and uses a direction enum to follow each
boundary in the original scan order.
Eight-connected regions can have several polygon parts when their pixels touch
only at corners; each part is exported separately.

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

## Arrow ownership and limits

WKB encoding validates ring and point counts before writing. Rust callers can use
`try_polygon_to_wkb_into` to handle encoding errors; the existing
`polygon_to_wkb_into` convenience function panics on unrepresentable counts.
Record-batch export checks the Binary array's signed 32-bit offset limit before
appending each polygon.

During Python export, Rust owns the Arrow C interface structs until the synchronous
PyArrow import finishes. PyArrow moves the buffers and clears the release callbacks;
Rust then drops the structs. If import fails, any unconsumed buffers are released
automatically. Returned tables own their buffers independently of the input array.

## Feature flags (Rust crate)

| Flag | Default | Description |
|---|---|---|
| `arrow` | off (on in Python bindings) | Arrow RecordBatch export with WKB geometry + GeoParquet metadata |
