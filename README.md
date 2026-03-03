# contourrs

Fast raster polygonization in pure Rust with Python bindings. Drop-in replacement for `rasterio.features.shapes` — no GDAL dependency.

Converts discrete/categorical rasters (segmentation masks, land cover, classified imagery) into vector polygons with their pixel values. Built for the ML-to-GIS pipeline: model inference output goes in, GeoJSON or GeoParquet comes out.

## Install

```bash
pip install contourrs
```

For Arrow/GeoParquet support:

```bash
pip install contourrs[arrow]
```

## Usage

### GeoJSON output (rasterio compatible)

```python
import numpy as np
from contourrs import shapes

raster = np.array([[1, 1, 2], [1, 2, 2], [3, 3, 3]], dtype=np.uint8)

for geojson, value in shapes(raster, connectivity=4):
    print(f"value={value}, type={geojson['type']}")
```

### Arrow output (zero-copy, GeoParquet-ready)

```python
from contourrs import shapes_arrow

table = shapes_arrow(raster, connectivity=4)
# table.schema: geometry (binary/WKB), value (float64)
# GeoParquet metadata included — write directly:
import pyarrow.parquet as pq
pq.write_table(table, "output.parquet")
```

### With affine transform and mask

```python
transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 4500000.0)  # (a, b, c, d, e, f)
mask = raster != 0  # exclude nodata

shapes(raster, mask=mask, connectivity=8, transform=transform)
```

## API

### `shapes(source, mask=None, connectivity=4, transform=None)`

Returns `list[tuple[dict, float]]` — GeoJSON geometry dicts paired with pixel values. Signature matches `rasterio.features.shapes`.

**Parameters:**
- `source` — 2D numpy array (uint8/16/32, int16/32, float32/64)
- `mask` — optional 2D bool array (True = include)
- `connectivity` — `4` or `8` pixel neighborhood
- `transform` — 6-element affine tuple `(a, b, c, d, e, f)`

### `shapes_arrow(source, mask=None, connectivity=4, transform=None)`

Returns `pyarrow.Table` with columns:
- `geometry` — Binary (WKB-encoded polygons)
- `value` — Float64

Schema includes GeoParquet metadata. 5-6x faster than `shapes()` at scale by eliminating Python dict overhead.

## Performance

Benchmarked against `rasterio.features.shapes` on random categorical rasters:

| Grid size | Values | `shapes()` | `rasterio` | Speedup |
|-----------|--------|------------|------------|---------|
| 64x64 | 5 | 0.3ms | 0.7ms | 2.4x |
| 256x256 | 5 | 6ms | 14ms | 2.3x |
| 1024x1024 | 5 | 105ms | 190ms | 1.8x |
| 2048x2048 | 5 | 680ms* | 5,380ms | 7.9x |

*\*`shapes_arrow()` — eliminates Python GeoJSON dict construction.*

## Architecture

Two-pass algorithm mirroring GDAL's `GDALPolygonize`:

1. **Region labeling** — connected-component labeling via union-find with path compression. Supports 4- and 8-connectivity with optional mask.
2. **Boundary tracing** — direct contour tracing with turn-priority logic. Classifies exterior rings (CCW) and holes (CW). Applies affine transform to output coordinates.

**Workspace layout:**
- `contourrs-core` — pure Rust library, returns `geo_types::Polygon<f64>`
- `contourrs-python` — PyO3/maturin bindings

**Feature flags** (Rust crate):
- `arrow` — Arrow RecordBatch export with WKB geometry + GeoParquet metadata
- `cuda` — GPU-accelerated connected-component labeling (scaffolded, requires CUDA toolkit)

## License

Apache-2.0
