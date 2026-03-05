# Getting Started

## Installation

=== "pip"

    ```bash
    pip install contourrs
    ```

=== "pip (CUDA extras)"

    ```bash
    pip install "contourrs[cuda]"
    ```

=== "pip (CUDA package)"

    ```bash
    pip install contourrs-cuda
    ```

=== "uv"

    ```bash
    uv add contourrs
    ```

Pre-built CPU wheels are available for Linux, macOS, and Windows on Python 3.12+.
CUDA runtime dependencies are optional via the `cuda` extra.
CUDA-enabled wheels are published separately under `contourrs-cuda`.

## Development setup

```bash
git clone https://github.com/isaaccorley/contourrs.git
cd contourrs
uv sync --extra dev
uv run maturin develop --release
```

Run tests:

```bash
uv run pytest tests/ -v
```

Pre-commit hooks:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## First usage

### Polygonize a categorical raster

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

Each `geojson` is a standard GeoJSON geometry dict (Polygon or MultiPolygon). The `value` is the pixel value for that region.

### Contour a continuous raster

```python
import numpy as np
from contourrs import contours

dem = np.random.default_rng(42).random((256, 256)).astype(np.float32)

for geojson, value in contours(dem, thresholds=[0.25, 0.5, 0.75]):
    print(f"band={value}, rings={len(geojson['coordinates'])}")
```

Thresholds define the break values between bands. Each band covers the interval `[lo, hi)` between consecutive thresholds.

### Arrow output (zero-copy)

For best performance and GeoParquet export, use the Arrow variants:

```python
from contourrs import shapes_arrow, contours_arrow

table = shapes_arrow(raster, connectivity=4)
# table.schema: geometry (binary/WKB), value (float64)

import pyarrow.parquet as pq
pq.write_table(table, "output.parquet")
```
