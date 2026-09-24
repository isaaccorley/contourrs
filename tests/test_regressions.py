"""Input layout and geometric precision regressions."""

import numpy as np
import pytest
from contourrs import (
    _resolve_source_and_mask,
    contours,
    contours_arrow,
    shapes,
    shapes_arrow,
)


@pytest.mark.parametrize("function", [shapes, shapes_arrow, contours, contours_arrow])
@pytest.mark.parametrize("layout", ["source", "mask"])
def test_fortran_layout_rejected(function, layout):
    source = np.array([[1, 1, 2], [1, 2, 2]], dtype=np.float64)
    mask = np.array([[True, False, True], [True, True, False]])
    if layout == "source":
        source = np.asfortranarray(source)
    else:
        mask = np.asfortranarray(mask)
    kwargs = {"thresholds": [0.5, 1.5]} if "contours" in function.__name__ else {}
    with pytest.raises(ValueError, match=f"{layout} must be C-contiguous"):
        function(source, mask=mask, **kwargs)


def test_nodata_mask_does_not_copy_or_mutate_source():
    source = np.array([[1.0, np.nan], [2.0, 3.0]])
    resolved, mask = _resolve_source_and_mask(source, nodata=np.nan)
    assert resolved is source
    assert np.isnan(source[0, 1])
    np.testing.assert_array_equal(mask, [[True, False], [True, True]])


def test_contour_bands_partition_sample_extent():
    shapely = pytest.importorskip("shapely")
    for seed in range(5):
        source = np.random.default_rng(seed).random((8, 9))
        polygons = [
            shapely.geometry.shape(geometry)
            for geometry, _ in contours(source, [-1, 0.25, 0.5, 0.75, 2])
        ]
        assert all(polygon.is_valid for polygon in polygons)
        total_area = sum(polygon.area for polygon in polygons)
        assert total_area == pytest.approx(56.0, abs=1e-6)
        assert shapely.union_all(polygons).area == pytest.approx(total_area, abs=1e-6)
