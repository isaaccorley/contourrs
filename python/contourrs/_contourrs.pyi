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

    Returns list of (geojson_geometry, value) tuples.
    """
    ...

def shapes_arrow(
    source: NDArray,
    mask: Optional[NDArray[np.bool_]] = None,
    connectivity: int = 4,
    transform: Optional[tuple[float, ...]] = None,
) -> pyarrow.Table:
    """Extract polygon shapes as a PyArrow Table with WKB geometry.

    Columns: geometry (binary/WKB), value (float64).
    Zero-copy from Rust. GeoParquet-compatible metadata included.
    """
    ...
