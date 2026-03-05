# contourrs

Fast raster polygonization and contouring in pure Rust with Python bindings.
Drop-in replacement for `rasterio.features.shapes` with no GDAL dependency.

Full docs: https://isaac.earth/contourrs/

## Example outputs

![USDA CDL tiled polygonization](assets/cdl_polygonize.png)

![Mount Rainier DEM elevation bins](assets/contours_mt_rainier.png)

## Install

```bash
pip install contourrs
```

## Quick start

```python
import geopandas as gpd
import numpy as np
from contourrs import contours, shapes, shapes_arrow

raster = np.array([[1, 1, 2], [1, 2, 2], [3, 3, 3]], dtype=np.uint8)
dem = np.random.default_rng(42).random((256, 256)).astype(np.float32)

# GeoJSON-style output
polygons = shapes(raster, connectivity=4)
isobands = contours(dem, thresholds=[0.25, 0.5, 0.75])

# Arrow/GeoParquet output
table = shapes_arrow(raster, connectivity=4)
gdf = gpd.GeoDataFrame.from_arrow(table)
```

## Real-world examples

```bash
# USDA Cropland Data Layer (tiled polygonization + merge)
python examples/cdl_tiled_polygonize.py --year 2023 --fips 19153 --tile-size 1024

# Synthetic + real DEM visualization
python examples/dem_contour.py
```

## Development

```bash
git clone https://github.com/isaaccorley/contourrs.git
cd contourrs
uv sync --extra dev
uv run maturin develop --release
uv run pytest tests/ -v
uv run pre-commit run --all-files
```

## More

- Examples: `docs/examples.md`
- Performance: `docs/performance.md`
- Architecture: `docs/architecture.md`
