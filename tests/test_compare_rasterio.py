"""Compare contourrs output against rasterio.features.shapes.

Integration tests — require rasterio + shapely.
Run with: pytest -m integration --extra test
"""

import pytest
import numpy as np
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape

from contourrs import shapes as rust_shapes

pytestmark = pytest.mark.integration


def normalize_polygon(geom_dict):
    """Convert a GeoJSON geometry dict to a Shapely geometry for comparison."""
    return shape(geom_dict)


def compare_results(rust_result, rio_result, tolerance=1e-10):
    """Compare rust and rasterio results as sets of (geometry, value) pairs."""
    # Group by value
    rust_by_val = {}
    for geom, val in rust_result:
        rust_by_val.setdefault(val, []).append(normalize_polygon(geom))

    rio_by_val = {}
    for geom, val in rio_result:
        rio_by_val.setdefault(val, []).append(shape(geom))

    assert set(rust_by_val.keys()) == set(rio_by_val.keys()), (
        f"Value mismatch: rust={sorted(rust_by_val.keys())}, "
        f"rio={sorted(rio_by_val.keys())}"
    )

    for val in rust_by_val:
        rust_polys = rust_by_val[val]
        rio_polys = rio_by_val[val]

        # Merge all polygons for each value and compare total area/shape
        from shapely.ops import unary_union

        rust_union = unary_union(rust_polys)
        rio_union = unary_union(rio_polys)

        # Check symmetric difference is near zero
        sym_diff = rust_union.symmetric_difference(rio_union)
        assert sym_diff.area < tolerance, (
            f"Geometry mismatch for value {val}: "
            f"symmetric_difference area = {sym_diff.area}"
        )


def to_rasterio_transform(transform_tuple):
    """Convert (a,b,c,d,e,f) to rasterio Affine."""
    from affine import Affine

    a, b, c, d, e, f = transform_tuple
    return Affine(a, b, c, d, e, f)


def test_uniform_grid():
    """All pixels same value."""
    data = np.ones((4, 4), dtype=np.uint8)
    rust = rust_shapes(data)
    rio = list(rio_shapes(data))
    compare_results(rust, rio)
    print("  PASS: uniform grid")


def test_two_values():
    """Left half = 1, right half = 2."""
    data = np.zeros((4, 6), dtype=np.uint8)
    data[:, :3] = 1
    data[:, 3:] = 2
    rust = rust_shapes(data)
    rio = list(rio_shapes(data))
    compare_results(rust, rio)
    print("  PASS: two values")


def test_checkerboard():
    """Checkerboard pattern — many small regions."""
    data = np.zeros((4, 4), dtype=np.uint8)
    for r in range(4):
        for c in range(4):
            data[r, c] = (r + c) % 2
    rust = rust_shapes(data, connectivity=4)
    rio = list(rio_shapes(data, connectivity=4))
    compare_results(rust, rio)
    print("  PASS: checkerboard (4-conn)")


def test_checkerboard_8conn():
    """Checkerboard with 8-connectivity."""
    data = np.zeros((4, 4), dtype=np.uint8)
    for r in range(4):
        for c in range(4):
            data[r, c] = (r + c) % 2
    rust = rust_shapes(data, connectivity=8)
    rio = list(rio_shapes(data, connectivity=8))
    compare_results(rust, rio)
    print("  PASS: checkerboard (8-conn)")


def test_single_pixel():
    """1x1 grid."""
    data = np.array([[42]], dtype=np.uint8)
    rust = rust_shapes(data)
    rio = list(rio_shapes(data))
    compare_results(rust, rio)
    print("  PASS: single pixel")


def test_with_mask():
    """Mask out some pixels."""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    mask = np.array([[True, False, True], [True, True, False]], dtype=bool)
    rust = rust_shapes(data, mask=mask)
    rio = list(rio_shapes(data, mask=mask))
    compare_results(rust, rio)
    print("  PASS: with mask")


def test_all_masked():
    """Everything masked out."""
    data = np.ones((3, 3), dtype=np.uint8)
    mask = np.zeros((3, 3), dtype=bool)
    rust = rust_shapes(data, mask=mask)
    rio = list(rio_shapes(data, mask=mask))
    assert len(rust) == 0 and len(rio) == 0
    print("  PASS: all masked")


def test_with_transform():
    """Apply an affine transform."""
    data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    t = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
    rust = rust_shapes(data, transform=t)
    rio = list(rio_shapes(data, transform=to_rasterio_transform(t)))
    compare_results(rust, rio)
    print("  PASS: with transform")


def test_hole():
    """Region with a hole (center pixel different)."""
    data = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=np.uint8)
    rust = rust_shapes(data)
    rio = list(rio_shapes(data))
    compare_results(rust, rio)

    # Check that value=1 polygon has a hole
    for geom, val in rust:
        if val == 1.0:
            assert len(geom["coordinates"]) == 2, "Expected exterior + 1 hole"
    print("  PASS: hole")


def test_float32():
    """Float32 data."""
    data = np.array([[1.5, 2.5], [1.5, 2.5]], dtype=np.float32)
    rust = rust_shapes(data)
    rio = list(rio_shapes(data))
    compare_results(rust, rio)
    print("  PASS: float32")


def test_random_mask():
    """Random data with random mask."""
    rng = np.random.default_rng(42)
    data = rng.integers(0, 5, size=(10, 10), dtype=np.uint8)
    mask = rng.choice([True, False], size=(10, 10))
    rust = rust_shapes(data, mask=mask)
    rio = list(rio_shapes(data, mask=mask))
    compare_results(rust, rio)
    print("  PASS: random mask")


def test_larger_grid():
    """Larger grid with multiple regions."""
    rng = np.random.default_rng(123)
    data = rng.integers(0, 3, size=(32, 32), dtype=np.uint8)
    rust = rust_shapes(data, connectivity=4)
    rio = list(rio_shapes(data, connectivity=4))
    compare_results(rust, rio)
    print("  PASS: larger grid (32x32)")


if __name__ == "__main__":
    tests = [
        test_uniform_grid,
        test_two_values,
        test_checkerboard,
        test_checkerboard_8conn,
        test_single_pixel,
        test_with_mask,
        test_all_masked,
        test_with_transform,
        test_hole,
        test_float32,
        test_random_mask,
        test_larger_grid,
    ]
    print(f"Running {len(tests)} comparison tests...")
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    if failed > 0:
        exit(1)
