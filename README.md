# contourrs

<img src="https://raw.githubusercontent.com/isaaccorley/contourrs/main/assets/logo.png" alt="contourrs — a coral contour island inside a charcoal outline" width="600">

[![PyPI](https://img.shields.io/pypi/v/contourrs)](https://pypi.org/project/contourrs/)
[![DOI](https://zenodo.org/badge/1171138064.svg)](https://doi.org/10.5281/zenodo.22949665)

contourrs converts NumPy rasters into polygons with a Rust core and Python bindings.
Use it to trace land-cover classes and segmentation masks, or extract filled contour bands from elevation and probability grids.
The Python package requires no GDAL installation.

[Documentation](https://isaac.earth/contourrs/) · [API reference](https://isaac.earth/contourrs/api/) · [PyPI](https://pypi.org/project/contourrs/) · [Benchmarks](https://isaac.earth/contourrs/performance/)

## Install

```bash
pip install contourrs
```

Wheels are available for Python 3.12–3.14 on Linux (x86_64 and ARM64), macOS (Apple Silicon), and Windows (x86_64).
NumPy and PyArrow are installed as dependencies.

## Polygonize a raster

Each connected region of equal-valued pixels becomes a polygon.

```python
import numpy as np
from contourrs import shapes

labels = np.array([[1, 1, 2], [1, 2, 2], [3, 3, 3]], dtype=np.uint8)

for geometry, value in shapes(labels, connectivity=4):
    print(value, geometry["type"])
# 1.0 Polygon
# 2.0 Polygon
# 3.0 Polygon
```

Use `connectivity=8` to join regions that touch diagonally.
Like `rasterio.features.shapes`, each result pairs a GeoJSON geometry dictionary with its raster value.
`shapes()` returns a list rather than an iterator.

![USDA Cropland Data Layer raster and extracted polygons](https://raw.githubusercontent.com/isaaccorley/contourrs/main/assets/cdl_polygonize.png)

*A 512 × 512 crop of the 2023 USDA Cropland Data Layer for Polk County, Iowa.
The right panel shows polygons extracted in tiles and merged across tile boundaries.*

## Export to Arrow and GeoParquet

The Arrow variants return a `pyarrow.Table` with a WKB `geometry` column and a float64 `value` column.
They avoid constructing a Python geometry dictionary for every polygon and transfer buffers through the Arrow C Data Interface without copying them.

```python
import pyarrow.parquet as pq
from contourrs import shapes_arrow

table = shapes_arrow(labels, connectivity=4)
pq.write_table(table, "polygons.parquet")
```

The table includes GeoParquet metadata.
An affine transform sets output coordinates, but the CRS remains unknown until you assign it; see [georeferenced export](https://isaac.earth/contourrs/api/#coordinate-reference-systems).

## Extract contour bands

`contours()` interpolates boundaries between raster samples and returns filled polygons between consecutive thresholds.
Each result's value is the lower threshold of its band.

```python
import numpy as np
from contourrs import contours

axis = np.linspace(-2, 2, 128)
x, y = np.meshgrid(axis, axis)
elevation = np.exp(-(x**2 + y**2))

bands = contours(elevation, thresholds=[0.1, 0.3, 0.5, 0.7, 1.0])
```

This extracts the bands `[0.1, 0.3)`, `[0.3, 0.5)`, `[0.5, 0.7)`, and `[0.7, 1.0)`.
Values outside those intervals are excluded.

![A synthetic elevation field and its interpolated contour bands](https://raw.githubusercontent.com/isaaccorley/contourrs/main/assets/contours.png)

*Interpolated bands from a synthetic terrain with three peaks.
The raster and filled bands share a color scale; white areas fall below the first threshold.*

Both operations offer the same two output formats:

| Input | GeoJSON geometry/value pairs | Arrow table |
|---|---|---|
| Categorical raster | `shapes()` | `shapes_arrow()` |
| Continuous raster | `contours()` | `contours_arrow()` |

## Masks and map coordinates

All four functions accept `mask`, `nodata`, and `transform`:

```python
from contourrs import shapes_arrow

table = shapes_arrow(
    labels,
    nodata=0,
    transform=(10, 0, 500000, 0, -10, 4500000),
)
```

- `True` entries in `mask` include samples. Use `nodata=0` to exclude zeros or `nodata=np.nan` to exclude NaNs.
- `transform` accepts an `affine.Affine` or the six coefficients `(a, b, c, d, e, f)` used by rasterio.
- Inputs and masks must be two-dimensional and C-contiguous. Use `np.ascontiguousarray()` after slicing or transposing when needed.
- Accepted dtypes are `uint8`, `uint16`, `uint32`, `int16`, `int32`, `float32`, and `float64`.

Polygonization follows pixel edges; contours interpolate between samples at integer `(column, row)` coordinates.
See the [coordinate conventions](https://isaac.earth/contourrs/api/#contour-coordinates) when applying a raster transform.
Read raster files with rasterio or another loader, then pass the arrays to contourrs.

## Tutorials

- [Quickstart](https://isaac.earth/contourrs/tutorials/quickstart/) covers masks, transforms, and both output formats.
- [DEM contours](https://isaac.earth/contourrs/tutorials/dem_contour/) extracts elevation bands from synthetic terrain and a Mount Rainier DEM.
- [Tiled CDL polygonization](https://isaac.earth/contourrs/tutorials/cdl_tiled_polygonize/) polygonizes land cover in tiles and merges their boundaries.
- [TorchGeo field segmentation](https://isaac.earth/contourrs/tutorials/torchgeo_ftw_polygonize/) converts Fields of the World model predictions into field polygons.

Notebook sources are in [`examples/`](https://github.com/isaaccorley/contourrs/tree/main/examples).
[Performance notes](https://isaac.earth/contourrs/performance/) describe the benchmark inputs, output costs, and memory measurements.

## Development

Install [uv](https://docs.astral.sh/uv/) and [Rust](https://rustup.rs/), then:

```bash
git clone https://github.com/isaaccorley/contourrs.git
cd contourrs
make install
make test
make check
```

Run `make build` after changing Rust code.
The checks include Rust formatting and Clippy, Ruff, ty, Pyrefly, and the Rust test suite.
`make test` also runs the Python tests, including comparisons against rasterio.
See [development and docs setup](https://isaac.earth/contourrs/getting-started/#development-setup) and the [Rust architecture](https://isaac.earth/contourrs/architecture/) for more.

## License

[Apache-2.0](https://github.com/isaaccorley/contourrs/blob/main/LICENSE).

## Citation

Cite the [v0.8.1 Zenodo archive](https://doi.org/10.5281/zenodo.22949666):

```bibtex
@software{isaac_corley_2026_22949666,
  author       = {Isaac Corley},
  title        = {isaaccorley/contourrs: v0.8.1},
  month        = sep,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v0.8.1},
  doi          = {10.5281/zenodo.22949666},
  url          = {https://doi.org/10.5281/zenodo.22949666},
}
```
