# API Reference

All functions are importable from the top-level `contourrs` package.

## shapes

```python
def shapes(
    source: NDArray,
    mask: NDArray[np.bool_] | None = None,
    connectivity: int = 4,
    transform: Affine | tuple[float, ...] | None = None,
    nodata: int | float | None = None,
) -> list[tuple[dict, float]]
```

Extract polygon shapes from a raster array.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | `NDArray` | *required* | 2D numpy array (uint8/16/32, int16/32, float32/64) |
| `mask` | `NDArray[np.bool_]` | `None` | 2D boolean array. `True` = include pixel |
| `connectivity` | `int` | `4` | Pixel neighborhood: `4` or `8` |
| `transform` | `Affine \| tuple[float, ...]` | `None` | `affine.Affine` or `(a, b, c, d, e, f)` |
| `nodata` | `int \| float` | `None` | Exclude pixels equal to this value; `np.nan` excludes NaNs |

**Returns:** `list[tuple[dict, float]]` — GeoJSON geometry dicts paired with pixel values. Returns an eager list; `rasterio.features.shapes` returns an iterator.

**Example:**

```python
import numpy as np
from contourrs import shapes

raster = np.array([[1, 1, 2], [1, 2, 2], [3, 3, 3]], dtype=np.uint8)
results = shapes(raster, connectivity=4)

for geojson, value in results:
    print(f"value={value}, type={geojson['type']}")
```

---

## shapes_arrow

```python
def shapes_arrow(
    source: NDArray,
    mask: NDArray[np.bool_] | None = None,
    connectivity: int = 4,
    transform: Affine | tuple[float, ...] | None = None,
    nodata: int | float | None = None,
) -> pyarrow.Table
```

Extract polygon shapes as a PyArrow Table with WKB geometry.

Zero-copy from Rust via Arrow C Data Interface. Schema includes GeoParquet-compatible metadata for direct parquet export. **5-6x faster** than `shapes()` at scale by eliminating Python dict overhead.

**Parameters:** Same as [`shapes()`](#shapes).

**Returns:** `pyarrow.Table` with columns:

| Column | Type | Description |
|---|---|---|
| `geometry` | Binary | WKB-encoded polygons |
| `value` | Float64 | Pixel value |

**Example:**

```python
from contourrs import shapes_arrow
import pyarrow.parquet as pq

table = shapes_arrow(raster, connectivity=4)
pq.write_table(table, "polygons.parquet")
```

---

## contours

```python
def contours(
    source: NDArray,
    thresholds: list[float],
    mask: NDArray[np.bool_] | None = None,
    transform: Affine | tuple[float, ...] | None = None,
    nodata: int | float | None = None,
) -> list[tuple[dict, float]]
```

Generate filled contour (isoband) polygons from a continuous raster.

Uses marching squares to produce polygons between consecutive threshold pairs. Returns the same format as `shapes()` for compatibility.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | `NDArray` | *required* | 2D numpy array (uint8/16/32, int16/32, float32/64) |
| `thresholds` | `list[float]` | *required* | Break values; bands formed from consecutive pairs (min 2) |
| `mask` | `NDArray[np.bool_]` | `None` | 2D boolean array. `True` = include pixel |
| `transform` | `Affine \| tuple[float, ...]` | `None` | `affine.Affine` or `(a, b, c, d, e, f)` |
| `nodata` | `int \| float` | `None` | Exclude pixels equal to this value; `np.nan` excludes NaNs |

**Returns:** `list[tuple[dict, float]]` — GeoJSON geometry dicts paired with the lower threshold of each band.

**Example:**

```python
import numpy as np
from contourrs import contours

dem = np.random.default_rng(42).random((256, 256)).astype(np.float32)

for geojson, value in contours(dem, thresholds=[0.25, 0.5, 0.75]):
    print(f"band={value}, rings={len(geojson['coordinates'])}")
```

---

## contours_arrow

```python
def contours_arrow(
    source: NDArray,
    thresholds: list[float],
    mask: NDArray[np.bool_] | None = None,
    transform: Affine | tuple[float, ...] | None = None,
    nodata: int | float | None = None,
) -> pyarrow.Table
```

Generate filled contour polygons as a PyArrow Table with WKB geometry.

Same as `contours()` but returns an Arrow Table via zero-copy FFI. Schema includes GeoParquet-compatible metadata.

**Parameters:** Same as [`contours()`](#contours).

**Returns:** `pyarrow.Table` with columns:

| Column | Type | Description |
|---|---|---|
| `geometry` | Binary | WKB-encoded polygons |
| `value` | Float64 | Band lower threshold |

**Example:**

```python
from contourrs import contours_arrow
import pyarrow.parquet as pq

dem = np.random.default_rng(42).random((256, 256)).astype(np.float32)
table = contours_arrow(dem, thresholds=[0.25, 0.5, 0.75])
pq.write_table(table, "contours.parquet")
```

---

## Supported dtypes

All functions accept the following input types. All output coordinates are `f64`.

| Input dtype | Python type | Precision |
|---|---|---|
| `uint8` | `np.uint8` | Lossless -> f64 |
| `uint16` | `np.uint16` | Lossless -> f64 |
| `uint32` | `np.uint32` | Lossless -> f64 |
| `int16` | `np.int16` | Lossless -> f64 |
| `int32` | `np.int32` | Lossless -> f64 |
| `float32` | `np.float32` | Promoted to f64 |
| `float64` | `np.float64` | Native (zero-copy in contours) |

## Mask and nodata semantics

- `mask` uses rasterio-style semantics: `True` includes a pixel, `False` excludes it
- `nodata=` is optional sugar for building that exclusion mask from the source array
- If both `mask` and `nodata` are passed, the effective mask is `mask & (source != nodata)`
- `nodata=np.nan` excludes NaN pixels

## Affine transform

The `transform` parameter accepts either an `affine.Affine` instance or a 6-element tuple `(a, b, c, d, e, f)` representing the affine transformation:

```
x' = a * col + b * row + c
y' = d * col + e * row + f
```

This matches the convention used by `rasterio` and GDAL. When `None`, pixel coordinates are used directly (identity transform).
