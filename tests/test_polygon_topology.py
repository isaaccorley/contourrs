"""Raster corner contacts must produce valid polygons without losing pixels."""

import numpy as np
import pytest
from affine import Affine
from contourrs import shapes, shapes_arrow
from rasterio.features import shapes as rio_shapes
from shapely import from_wkb, union_all
from shapely.geometry import shape


def polygons(source, output, **kwargs):
    if output == "geojson":
        return [
            (shape(geometry), value) for geometry, value in shapes(source, **kwargs)
        ]
    table = shapes_arrow(source, **kwargs)
    return list(
        zip(
            from_wkb(table.column("geometry").to_numpy(zero_copy_only=False)),
            table.column("value").to_pylist(),
            strict=True,
        ),
    )


@pytest.mark.parametrize("output", ["geojson", "arrow"])
@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize(
    "transform",
    [Affine.identity(), Affine(10, 2, 500000, 1, -10, 4500000)],
)
def test_hole_touches_shell_at_pixel_corner(output, connectivity, rotation, transform):
    source = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    source = np.ascontiguousarray(np.rot90(source, rotation))
    result = polygons(source, output, connectivity=connectivity, transform=transform)
    assert len(result) == 3
    assert all(geometry.is_valid for geometry, _ in result)
    foreground = [geometry for geometry, value in result if value == 1]
    assert len(foreground) == 1
    assert len(foreground[0].interiors) == 1
    assert foreground[0].area == pytest.approx(7 * abs(transform.determinant))
    # Four-connectivity gives valid reference rings even when comparing the
    # coverage of eight-connected regions with several polygon parts.
    reference = list(rio_shapes(source, connectivity=4, transform=transform))
    for value in (0, 1):
        actual = union_all([geometry for geometry, v in result if v == value])
        expected = union_all(
            [shape(geometry) for geometry, v in reference if v == value]
        )
        assert actual.equals(expected)


@pytest.mark.parametrize("output", ["geojson", "arrow"])
@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("rotation", range(4))
def test_corner_touching_hole_with_multiple_shells(output, connectivity, rotation):
    source = np.array([[2, 2, 2, 1], [2, 1, 2, 0], [2, 2, 0, 2]], dtype=np.uint8)
    source = np.ascontiguousarray(np.rot90(source, rotation))
    result = polygons(source, output, connectivity=connectivity)
    assert all(geometry.is_valid for geometry, _ in result)
    parts = [geometry for geometry, value in result if value == 2]
    assert len(parts) == 2
    assert sum(len(geometry.interiors) for geometry in parts) == 1
    assert sum(geometry.area for geometry in parts) == 8


@pytest.mark.parametrize("output", ["geojson", "arrow"])
@pytest.mark.parametrize("connectivity", [4, 8])
def test_random_rasters_preserve_valid_class_coverage(output, connectivity):
    for seed in range(10):
        rng = np.random.default_rng(seed)
        source = rng.integers(0, 3, size=(20, 21), dtype=np.uint8)
        mask = rng.random(source.shape) > 0.1
        result = polygons(source, output, connectivity=connectivity, mask=mask)
        assert all(geometry.is_valid for geometry, _ in result), seed
        reference = list(rio_shapes(source, mask=mask, connectivity=4))
        for value in range(3):
            parts = [geometry for geometry, v in result if v == value]
            actual = union_all(parts)
            expected = union_all(
                [shape(geometry) for geometry, v in reference if v == value]
            )
            assert actual.equals(expected), (seed, value)
            assert sum(geometry.area for geometry in parts) == np.count_nonzero(
                (source == value) & mask,
            )
