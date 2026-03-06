# Examples

## Quick snippets

### Basic polygonization

```python
import numpy as np
from contourrs import shapes

raster = np.array([
    [1, 1, 2],
    [1, 2, 2],
    [3, 3, 3],
], dtype=np.uint8)

for geojson, value in shapes(raster, connectivity=4):
    print(f"value={value}, type={geojson['type']}")
```

### Basic contouring

```python
import numpy as np
from contourrs import contours

dem = np.random.default_rng(42).random((256, 256)).astype(np.float32)

for geojson, value in contours(dem, thresholds=[0.25, 0.5, 0.75]):
    print(f"band={value}, rings={len(geojson['coordinates'])}")
```

### Arrow output and GeoParquet

Arrow variants return a `pyarrow.Table` with WKB geometry and GeoParquet metadata — write directly to parquet:

```python
from contourrs import shapes_arrow, contours_arrow
import pyarrow.parquet as pq

# Discrete raster
raster = np.random.randint(0, 5, (512, 512), dtype=np.uint8)
table = shapes_arrow(raster, connectivity=4)
pq.write_table(table, "polygons.parquet")

# Continuous raster
dem = np.random.default_rng(42).random((512, 512)).astype(np.float32)
table = contours_arrow(dem, thresholds=[0.2, 0.4, 0.6, 0.8])
pq.write_table(table, "contours.parquet")
```

### Convert to GeoPandas

Both Arrow functions return tables with GeoParquet metadata, so GeoPandas reads them directly:

```python
import geopandas as gpd
import numpy as np
from contourrs import shapes_arrow, contours_arrow

raster = np.random.randint(0, 5, (256, 256), dtype=np.uint8)
gdf = gpd.GeoDataFrame.from_arrow(shapes_arrow(raster))

dem = np.random.default_rng(42).random((256, 256)).astype(np.float32)
gdf = gpd.GeoDataFrame.from_arrow(contours_arrow(dem, thresholds=[0.25, 0.5, 0.75]))
```

### Using a mask

Exclude pixels from processing (e.g. nodata regions):

```python
import numpy as np
from contourrs import shapes

raster = np.array([
    [0, 1, 1],
    [0, 2, 2],
    [3, 3, 3],
], dtype=np.uint8)

mask = raster != 0  # exclude nodata
results = shapes(raster, mask=mask, connectivity=4)
```

### With affine transform

Apply a georeferencing transform to output coordinates:

```python
import numpy as np
from contourrs import shapes, contours

raster = np.random.randint(0, 5, (256, 256), dtype=np.uint8)
dem = np.random.default_rng(42).random((256, 256)).astype(np.float32)

# (a, b, c, d, e, f) — 10m pixel, UTM origin
transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 4500000.0)

results = shapes(raster, connectivity=8, transform=transform)
results = contours(dem, thresholds=[0.25, 0.5, 0.75], transform=transform)
```

### 8-connectivity

Use 8-connectivity to merge diagonally-adjacent pixels:

```python
import numpy as np
from contourrs import shapes

raster = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
], dtype=np.uint8)

# 4-connectivity: each "1" pixel is a separate region
results_4 = shapes(raster, connectivity=4)

# 8-connectivity: diagonal "1" pixels merge into one region
results_8 = shapes(raster, connectivity=8)
```

### Mask + transform + Arrow (full pipeline)

Combine all features for a complete ML-to-GIS pipeline:

```python
import numpy as np
import pyarrow.parquet as pq
from contourrs import shapes_arrow

# Simulated model output
predictions = np.random.randint(0, 10, (1024, 1024), dtype=np.uint8)
confidence = np.random.random((1024, 1024)) > 0.1  # mask low-confidence

transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 4500000.0)

table = shapes_arrow(
    predictions,
    mask=confidence,
    connectivity=4,
    transform=transform,
)
pq.write_table(table, "predictions.parquet")
```

---

## Tutorials

Full walkthrough notebooks with visualizations. Each tutorial is rendered from an executed Jupyter notebook.

| Tutorial | Description |
|---|---|
| [Quickstart](tutorials/quickstart.md) | Core API tour — shapes, contours, Arrow, masks, transforms, GeoPandas, benchmarks |
| [DEM Contours](tutorials/dem_contour.md) | Synthetic DEM isoband extraction with `contours_arrow` |
| [Tiled CDL Polygonization](tutorials/cdl_tiled_polygonize.md) | Tile-based polygonization, cross-tile merge, and side-by-side visualization |
| [TorchGeo FTW Polygonize](tutorials/torchgeo_ftw_polygonize.md) | Run a segmentation model and polygonize field boundaries |

All notebook source files live in [`examples/`](https://github.com/isaaccorley/contourrs/tree/main/examples).
