# Architecture

## Workspace layout

```
contourrs           Pure Rust library, returns geo_types::Polygon<f64>
contourrs-python    PyO3/maturin Python bindings
```

## Polygonize pipeline

Two-pass algorithm in the same family as GDAL's `GDALPolygonize`:

### Pass 1 — Region labeling

Connected-component labeling via union-find with path compression. Supports 4- and 8-connectivity with optional mask.

### Pass 2 — Boundary tracing

Boundary tracing over labeled regions, followed by ring normalization and affine transform application on output coordinates.

## Contours pipeline

Two-isoline marching squares decomposition:

1. For each band `[lo, hi)`, run standard 16-case marching squares at both thresholds
2. Set-difference the resulting rings: lo-exteriors become isoband boundaries, hi-exteriors become holes
3. Interpolation along cell edges places boundaries at sub-pixel precision

## Optimizations

### Algorithmic

| Optimization | Impact | Description |
|---|---|---|
| **Rayon-parallel marching squares** | HIGH | Rows processed in parallel via rayon. 4x speedup on 1024x1024 contours. Kicks in above 128x128 grids |
| **Threshold ring caching** | HIGH | `hi_rings` from iteration k reused as `lo_rings` for k+1. Halves grid scans from 2(n-1) to n for n thresholds |
| **BBox spatial indexing** | MEDIUM | Exterior bounding boxes pre-computed; holes skip expensive `point_in_ring` ray-cast when outside bbox |
| **BBox pre-check in `point_in_ring`** | MEDIUM | Min/max bbox scan before O(n) ray-cast. Early return if point outside ring bounds |
| **Bulk WKB coordinate writes** | MEDIUM | On little-endian targets, `&[Coord<f64>]` reinterpreted as bytes for single `extend_from_slice` per ring |
| **Borrowed f64 raster path** | MEDIUM | When contour input is already `f64`, `Cow::Borrowed` avoids allocating a conversion buffer |

### Micro-optimizations

| Optimization | Location | Description |
|---|---|---|
| `#[inline]` on union-find `find`/`union` | union_find.rs | Called per-pixel; eliminates cross-module call overhead |
| Union-find path halving | union_find.rs | Local parent/grandparent vars; fewer array indexings |
| `#[inline]` on geometry functions | geometry.rs | `signed_area`, `point_in_ring` — hot loops |
| Pre-allocated segments Vec | contour.rs | `Vec::with_capacity(w * 2)` avoids early reallocations |
| Pre-allocated HashMap | contour.rs | `HashMap::with_capacity(segments.len())` in `chain_segments` |
| Pre-computed hole areas | contour.rs | Hole areas cached before multi-exterior assignment loop |
| Hoisted `det` computation | contour.rs | Transform determinant computed once before threshold loop |

## Feature flags (Rust crate)

| Flag | Default | Description |
|---|---|---|
| `arrow` | off (on in Python bindings) | Arrow RecordBatch export with WKB geometry + GeoParquet metadata |
| `cuda` | off | Experimental GPU connected-component labeling path for the Rust crate only |

## CUDA support (experimental)

Current pipeline: accept GPU pointer -> run CCL kernel on device -> transfer u32 label grid to CPU -> boundary tracing on CPU.

This path exists behind the `cuda` feature in the Rust crate, but it is not exposed in the Python package and should be treated as experimental rather than production-ready.
