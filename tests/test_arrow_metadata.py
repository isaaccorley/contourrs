"""Ensure Arrow exports do not invent a CRS for raster coordinates."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest
from contourrs import contours_arrow, shapes_arrow


@pytest.mark.parametrize("kind", ["shapes", "contours"])
@pytest.mark.parametrize("transform", [None, (30, 0, 500000, 0, -30, 4200000)])
def test_geoparquet_preserves_unknown_crs(tmp_path, kind, transform):
    if kind == "shapes":
        table = shapes_arrow(np.ones((3, 3), dtype=np.uint8), transform=transform)
    else:
        table = contours_arrow(
            np.ones((3, 3), dtype=np.float64),
            thresholds=[0, 2],
            transform=transform,
        )
    path = tmp_path / "polygons.parquet"
    pq.write_table(table, path)
    restored = pq.read_table(path)
    metadata = json.loads(restored.schema.metadata[b"geo"])
    geometry = metadata["columns"]["geometry"]
    assert "crs" in geometry
    assert geometry["crs"] is None
    assert restored.num_rows == table.num_rows
