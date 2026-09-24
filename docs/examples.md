# Examples

## Quick snippets

### Basic polygonization

```python
import numpy as np
from contourrs import shapes

raster = np.array(
    [
        [1, 1, 2],
        [1, 2, 2],
        [3, 3, 3],
    ],
    dtype=np.uint8,
)

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

Arrow variants return a `pyarrow.Table` with WKB geometry and GeoParquet metadata.
Write the table directly to Parquet:

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

Both Arrow functions attach GeoArrow WKB field metadata, which lets GeoPandas read the geometry column:

```python
import geopandas as gpd
import numpy as np
from contourrs import shapes_arrow, contours_arrow

raster = np.random.randint(0, 5, (256, 256), dtype=np.uint8)
gdf = gpd.GeoDataFrame.from_arrow(shapes_arrow(raster))

dem = np.random.default_rng(42).random((256, 256)).astype(np.float32)
gdf = gpd.GeoDataFrame.from_arrow(contours_arrow(dem, thresholds=[0.25, 0.5, 0.75]))
```

### Using `nodata`

Exclude pixels with a known nodata value without building a mask manually:

```python
import numpy as np
from contourrs import shapes

raster = np.array(
    [
        [0, 1, 1],
        [0, 2, 2],
        [3, 3, 3],
    ],
    dtype=np.uint8,
)

results = shapes(raster, nodata=0, connectivity=4)
```

### Using a mask

Exclude pixels from processing (e.g. nodata regions):

```python
import numpy as np
from contourrs import shapes

raster = np.array(
    [
        [0, 1, 1],
        [0, 2, 2],
        [3, 3, 3],
    ],
    dtype=np.uint8,
)

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

from affine import Affine

# 10m pixel, UTM origin
transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4500000.0)

results = shapes(raster, connectivity=8, transform=transform)
results = contours(dem, thresholds=[0.25, 0.5, 0.75], transform=transform)
```

### 8-connectivity

Use 8-connectivity to merge diagonally-adjacent pixels:

```python
import numpy as np
from contourrs import shapes

raster = np.array(
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ],
    dtype=np.uint8,
)

# 4-connectivity: each "1" pixel is a separate region
results_4 = shapes(raster, connectivity=4)

# 8-connectivity: diagonal "1" pixels merge into one region
results_8 = shapes(raster, connectivity=8)
```

### Mask + transform + Arrow (full pipeline)

Mask low-confidence pixels and transform the polygon coordinates before export:

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

The tutorials are generated from Jupyter notebooks.
The quickstart, DEM, and CDL notebooks execute during the docs build, with cells tagged `skip_ci` removed.
The TorchGeo tutorial is converted without execution because it requires model weights and imagery.

| Tutorial | Description |
|---|---|
| [Quickstart](tutorials/quickstart.md) | Core API tour — shapes, contours, Arrow, masks, transforms, GeoPandas, benchmarks |
| [DEM Contours](tutorials/dem_contour.md) | Synthetic DEM isoband extraction with `contours_arrow` |
| [Tiled CDL Polygonization](tutorials/cdl_tiled_polygonize.md) | Tile-based polygonization, cross-tile merge, and side-by-side visualization |
| [TorchGeo FTW Polygonize](tutorials/torchgeo_ftw_polygonize.md) | Run a segmentation model and polygonize field boundaries |

All notebook source files live in [`examples/`](https://github.com/isaaccorley/contourrs/tree/main/examples).

## Figure sources

Run `uv run --extra dev python scripts/generate_readme_plots.py` to regenerate the synthetic overview figures.
The script exports 5.5-inch PDF and SVG figures plus 300-dpi PNG previews to `assets/` and `docs/assets/`.
It uses the same seeded four-class raster and three-Gaussian elevation field as the original examples; these are illustrations, not benchmark measurements.
The continuous field and contour bands share one value scale, with each band colored at its midpoint.

The real-data figures come from `examples/cdl_tiled_polygonize.py`, `examples/dem_contour.py`, and the TorchGeo notebook.
The elevation-bin figure uses categorical polygonization after quantile binning; it is distinct from interpolated marching-squares isobands in the DEM notebook.

With the cached example rasters available, regenerate the real-data figures with:

```bash
uv run --extra docs python examples/dem_contour.py
uv run --extra docs python examples/cdl_tiled_polygonize.py --raster examples/data/cdl_2023_polk_512.tif --tile-size 128
```

The CDL comparison uses the raster's class color table when available.
For crops without a color table, the generator assigns one display color per class and uses that mapping in both panels.
Dense geometry layers are rasterized inside PDF/SVG exports; labels and axes remain vector.
