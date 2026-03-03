"""Unit tests for contourrs — no rasterio/shapely required."""

import numpy as np
import pyarrow as pa
from contourrs import contours, contours_arrow, shapes, shapes_arrow

# ── shapes ──────────────────────────────────────────────────────────────


def test_shapes_uniform():
    data = np.ones((4, 4), dtype=np.uint8)
    result = shapes(data)
    assert len(result) == 1
    geom, val = result[0]
    assert val == 1.0
    assert geom["type"] in ("Polygon", "MultiPolygon")


def test_shapes_two_values():
    data = np.zeros((4, 6), dtype=np.uint8)
    data[:, :3] = 1
    data[:, 3:] = 2
    result = shapes(data)
    values = {v for _, v in result}
    assert values == {1.0, 2.0}


def test_shapes_single_pixel():
    data = np.array([[42]], dtype=np.uint8)
    result = shapes(data)
    assert len(result) == 1
    assert result[0][1] == 42.0
    coords = result[0][0]["coordinates"][0]
    assert len(coords) == 5  # closed ring


def test_shapes_mask():
    data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    mask = np.array([[True, False], [True, False]], dtype=bool)
    result = shapes(data, mask=mask)
    values = {v for _, v in result}
    assert 2.0 not in values
    assert 4.0 not in values


def test_shapes_all_masked():
    data = np.ones((3, 3), dtype=np.uint8)
    mask = np.zeros((3, 3), dtype=bool)
    result = shapes(data, mask=mask)
    assert len(result) == 0


def test_shapes_connectivity_8():
    # 8-conn should produce fewer or equal regions than 4-conn
    data = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    result_4 = shapes(data, connectivity=4)
    result_8 = shapes(data, connectivity=8)
    count_4 = sum(1 for _, v in result_4 if v == 1.0)
    count_8 = sum(1 for _, v in result_8 if v == 1.0)
    assert count_8 <= count_4


def test_shapes_transform():
    data = np.array([[1]], dtype=np.uint8)
    t = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
    result = shapes(data, transform=t)
    coords = result[0][0]["coordinates"][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    assert min(xs) >= 100.0
    assert max(xs) <= 110.0
    assert min(ys) >= 190.0
    assert max(ys) <= 200.0


def test_shapes_hole():
    data = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=np.uint8)
    result = shapes(data)
    for geom, val in result:
        if val == 1.0:
            assert len(geom["coordinates"]) == 2  # exterior + 1 hole


def test_shapes_float32():
    data = np.array([[1.5, 2.5], [1.5, 2.5]], dtype=np.float32)
    result = shapes(data)
    values = {v for _, v in result}
    assert 1.5 in values
    assert 2.5 in values


def test_shapes_dtypes():
    for dtype in [
        np.uint8,
        np.uint16,
        np.uint32,
        np.int16,
        np.int32,
        np.float32,
        np.float64,
    ]:
        data = np.array([[1, 2], [3, 4]], dtype=dtype)
        result = shapes(data)
        assert len(result) == 4


# ── shapes_arrow ────────────────────────────────────────────────────────


def test_shapes_arrow_returns_table():
    data = np.ones((4, 4), dtype=np.uint8)
    table = shapes_arrow(data)
    assert isinstance(table, pa.Table)
    assert "geometry" in table.column_names
    assert "value" in table.column_names
    assert table.num_rows == 1


def test_shapes_arrow_schema():
    data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    table = shapes_arrow(data)
    assert table.schema.field("geometry").type == pa.binary()
    assert table.schema.field("value").type == pa.float64()
    assert table.num_rows == 4


def test_shapes_arrow_geoparquet_metadata():
    data = np.ones((2, 2), dtype=np.uint8)
    table = shapes_arrow(data)
    meta = table.schema.metadata
    assert b"geo" in meta


# ── contours ────────────────────────────────────────────────────────────


def test_contours_basic():
    data = np.array([[0.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    result = contours(data, thresholds=[0.25, 0.75])
    assert len(result) >= 1
    for geom, val in result:
        assert geom["type"] in ("Polygon", "MultiPolygon")
        assert val == 0.25


def test_contours_multiple_bands():
    # Gradient from 0 to 1 — should produce polygons in both bands
    data = np.linspace(0, 1, 64).reshape(8, 8).astype(np.float32)
    result = contours(data, thresholds=[0.25, 0.5, 0.75])
    band_values = {v for _, v in result}
    assert 0.25 in band_values
    assert 0.5 in band_values


def test_contours_flat_below():
    data = np.zeros((4, 4), dtype=np.float32)
    result = contours(data, thresholds=[0.5, 1.0])
    assert len(result) == 0


def test_contours_flat_inside_band():
    data = np.full((4, 4), 0.75, dtype=np.float32)
    result = contours(data, thresholds=[0.5, 1.0])
    assert len(result) == 1
    assert result[0][1] == 0.5


def test_contours_with_mask():
    data = np.full((4, 4), 0.75, dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    result = contours(data, thresholds=[0.5, 1.0], mask=mask)
    assert len(result) == 0


def test_contours_with_transform():
    data = np.full((4, 4), 0.75, dtype=np.float32)
    t = (10.0, 0.0, 500.0, 0.0, -10.0, 4500.0)
    result = contours(data, thresholds=[0.5, 1.0], transform=t)
    assert len(result) >= 1
    coords = result[0][0]["coordinates"][0]
    xs = [c[0] for c in coords]
    assert min(xs) >= 490.0  # transformed coords


def test_contours_gaussian():
    """Regression test: smooth Gaussian field must produce contours."""
    y, x = np.mgrid[-3:3:64j, -3:3:64j]
    dem = np.exp(-(x**2 + y**2)).astype(np.float32)
    result = contours(dem, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9])
    assert len(result) >= 3
    band_values = {v for _, v in result}
    assert 0.1 in band_values
    assert 0.3 in band_values
    assert 0.5 in band_values


def test_contours_geopandas_roundtrip():
    """Arrow table → GeoPandas GeoDataFrame via from_arrow."""
    import geopandas as gpd

    data = np.linspace(0, 1, 64).reshape(8, 8).astype(np.float32)
    table = contours_arrow(data, thresholds=[0.25, 0.5, 0.75])
    gdf = gpd.GeoDataFrame.from_arrow(table)
    assert len(gdf) > 0
    assert "geometry" in gdf.columns
    assert "value" in gdf.columns


# ── contours_arrow ──────────────────────────────────────────────────────


def test_contours_arrow_returns_table():
    rng = np.random.default_rng(42)
    data = rng.random((16, 16)).astype(np.float32)
    table = contours_arrow(data, thresholds=[0.25, 0.5, 0.75])
    assert isinstance(table, pa.Table)
    assert "geometry" in table.column_names
    assert "value" in table.column_names
    assert table.num_rows > 0


def test_contours_arrow_geoparquet_metadata():
    data = np.full((4, 4), 0.75, dtype=np.float32)
    table = contours_arrow(data, thresholds=[0.5, 1.0])
    meta = table.schema.metadata
    assert b"geo" in meta
