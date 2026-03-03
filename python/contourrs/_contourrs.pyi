from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pyarrow

def shapes(
    source: NDArray,
    mask: Optional[NDArray[np.bool_]] = None,
    connectivity: int = 4,
    transform: Optional[tuple[float, ...]] = None,
) -> list[tuple[dict, float]]:
    """Extract polygon shapes from a raster array.

    Parameters
    ----------
    source : NDArray
        2D numpy array of raster values (uint8/16/32, int16/32, float32/64).
    mask : NDArray[np.bool_], optional
        2D boolean array. True = include pixel, False = exclude.
    connectivity : int, optional
        Pixel neighborhood connectivity, 4 or 8. Default is 4.
    transform : tuple[float, ...], optional
        Affine transform as (a, b, c, d, e, f). Default is identity.

    Returns
    -------
    list[tuple[dict, float]]
        List of (GeoJSON geometry dict, pixel value) tuples.
    """
    ...

def shapes_arrow(
    source: NDArray,
    mask: Optional[NDArray[np.bool_]] = None,
    connectivity: int = 4,
    transform: Optional[tuple[float, ...]] = None,
) -> pyarrow.Table:
    """Extract polygon shapes as a PyArrow Table with WKB geometry.

    Zero-copy from Rust via Arrow C Data Interface. Schema includes
    GeoParquet-compatible metadata for direct parquet export.

    Parameters
    ----------
    source : NDArray
        2D numpy array of raster values (uint8/16/32, int16/32, float32/64).
    mask : NDArray[np.bool_], optional
        2D boolean array. True = include pixel, False = exclude.
    connectivity : int, optional
        Pixel neighborhood connectivity, 4 or 8. Default is 4.
    transform : tuple[float, ...], optional
        Affine transform as (a, b, c, d, e, f). Default is identity.

    Returns
    -------
    pyarrow.Table
        Table with columns: ``geometry`` (binary/WKB), ``value`` (float64).
    """
    ...

def contours(
    source: NDArray,
    thresholds: list[float],
    mask: Optional[NDArray[np.bool_]] = None,
    transform: Optional[tuple[float, ...]] = None,
) -> list[tuple[dict, float]]:
    """Generate filled contour (isoband) polygons from a continuous raster.

    Uses marching squares to produce polygons between consecutive threshold
    pairs. Returns the same format as ``shapes()`` for compatibility.

    Parameters
    ----------
    source : NDArray
        2D numpy array of raster values (uint8/16/32, int16/32, float32/64).
    thresholds : list[float]
        Break values. Bands are formed from consecutive pairs.
        At least 2 thresholds required.
    mask : NDArray[np.bool_], optional
        2D boolean array. True = include pixel, False = exclude.
    transform : tuple[float, ...], optional
        Affine transform as (a, b, c, d, e, f). Default is identity.

    Returns
    -------
    list[tuple[dict, float]]
        List of (GeoJSON geometry dict, band lower threshold) tuples.
    """
    ...

def contours_arrow(
    source: NDArray,
    thresholds: list[float],
    mask: Optional[NDArray[np.bool_]] = None,
    transform: Optional[tuple[float, ...]] = None,
) -> pyarrow.Table:
    """Generate filled contour polygons as a PyArrow Table with WKB geometry.

    Same as ``contours()`` but returns an Arrow Table via zero-copy FFI.
    Schema includes GeoParquet-compatible metadata.

    Parameters
    ----------
    source : NDArray
        2D numpy array of raster values (uint8/16/32, int16/32, float32/64).
    thresholds : list[float]
        Break values. Bands are formed from consecutive pairs.
        At least 2 thresholds required.
    mask : NDArray[np.bool_], optional
        2D boolean array. True = include pixel, False = exclude.
    transform : tuple[float, ...], optional
        Affine transform as (a, b, c, d, e, f). Default is identity.

    Returns
    -------
    pyarrow.Table
        Table with columns: ``geometry`` (binary/WKB), ``value`` (float64).
    """
    ...
