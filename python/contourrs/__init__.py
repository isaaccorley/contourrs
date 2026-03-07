"""contourrs — fast raster polygonization with Arrow export."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from contourrs._contourrs import contours as _contours
from contourrs._contourrs import contours_arrow as _contours_arrow
from contourrs._contourrs import shapes as _shapes
from contourrs._contourrs import shapes_arrow as _shapes_arrow


def _is_nan_scalar(value: object) -> bool:
    return isinstance(value, float | np.floating) and math.isnan(float(value))


def _resolve_source_and_mask(
    source: Any,
    mask: Any = None,
    nodata: object = None,
) -> tuple[np.ndarray[Any, Any], Any]:
    source_arr = np.asarray(source)

    if nodata is None:
        return source_arr, mask

    nodata_mask = (
        ~np.isnan(source_arr) if _is_nan_scalar(nodata) else source_arr != nodata
    )
    nodata_mask = np.asarray(nodata_mask, dtype=np.bool_)

    # Replace nodata cells with a neutral value (0) so NaN/sentinel values don't
    # corrupt comparisons inside the Rust algorithm.
    if not nodata_mask.all():
        source_arr = source_arr.copy()
        if np.issubdtype(source_arr.dtype, np.floating):
            source_arr[~nodata_mask] = 0.0
        else:
            source_arr[~nodata_mask] = 0

    if mask is None:
        return source_arr, nodata_mask

    mask_arr = np.asarray(mask)
    if mask_arr.dtype.kind != "b":
        msg = "mask must be a 2D bool array"
        raise TypeError(msg)
    if mask_arr.shape != source_arr.shape:
        msg = (
            f"mask shape {mask_arr.shape} does not match"
            f" source shape {source_arr.shape}"
        )
        raise ValueError(msg)

    return source_arr, np.logical_and(mask_arr, nodata_mask)


def _normalize_transform(transform: object) -> tuple[float, ...] | None:
    if transform is None:
        return None

    # Check for Affine-like objects (e.g. affine.Affine) before plain-tuple check,
    # since affine.Affine is a tuple subclass with 9 elements but exposes the
    # standard 6 GDAL coefficients via .a/.b/.c/.d/.e/.f attributes.
    attrs = ("a", "b", "c", "d", "e", "f")
    if all(hasattr(transform, attr) for attr in attrs):
        return tuple(float(getattr(transform, attr)) for attr in attrs)

    if isinstance(transform, tuple):
        return cast("tuple[float, ...]", transform)

    msg = "transform must be a 6-tuple or an object with a, b, c, d, e, f attributes"
    raise TypeError(msg)


def shapes(
    source: Any,
    mask: Any = None,
    connectivity: int = 4,
    transform: object = None,
    nodata: object = None,
):
    source_arr, resolved_mask = _resolve_source_and_mask(
        source, mask=mask, nodata=nodata
    )
    resolved_transform: tuple[float, ...] | None = _normalize_transform(transform)
    return _shapes(
        source_arr,
        mask=resolved_mask,
        connectivity=connectivity,
        transform=resolved_transform,
    )


def shapes_arrow(
    source: Any,
    mask: Any = None,
    connectivity: int = 4,
    transform: object = None,
    nodata: object = None,
):
    source_arr, resolved_mask = _resolve_source_and_mask(
        source, mask=mask, nodata=nodata
    )
    resolved_transform: tuple[float, ...] | None = _normalize_transform(transform)
    return _shapes_arrow(
        source_arr,
        mask=resolved_mask,
        connectivity=connectivity,
        transform=resolved_transform,
    )


def contours(
    source: Any,
    thresholds: list[float],
    mask: Any = None,
    transform: object = None,
    nodata: object = None,
):
    source_arr, resolved_mask = _resolve_source_and_mask(
        source, mask=mask, nodata=nodata
    )
    resolved_transform: tuple[float, ...] | None = _normalize_transform(transform)
    return _contours(
        source_arr,
        thresholds,
        mask=resolved_mask,
        transform=resolved_transform,
    )


def contours_arrow(
    source: Any,
    thresholds: list[float],
    mask: Any = None,
    transform: object = None,
    nodata: object = None,
):
    source_arr, resolved_mask = _resolve_source_and_mask(
        source, mask=mask, nodata=nodata
    )
    resolved_transform: tuple[float, ...] | None = _normalize_transform(transform)
    return _contours_arrow(
        source_arr,
        thresholds,
        mask=resolved_mask,
        transform=resolved_transform,
    )


__all__ = ["contours", "contours_arrow", "shapes", "shapes_arrow"]
