# Features & Performance

## Optimizations

### Algorithmic

| Optimization | Impact | Description |
|---|---|---|
| **Rayon-parallel marching squares** | HIGH | `march_isoline` rows processed in parallel via rayon. 4x speedup on 1024x1024 contours. Kicks in above 128x128 grids. |
| **Threshold ring caching** | HIGH | `hi_rings` from iteration k reused as `lo_rings` for iteration k+1. Halves grid scans from 2(n-1) to n for n thresholds. Move (no clone) at end of loop. |
| **BBox spatial indexing** | MEDIUM | Exterior bounding boxes pre-computed; holes skip expensive `point_in_ring` ray-cast when outside bbox. Applied in both contour.rs and trace.rs. |
| **BBox pre-check in `point_in_ring`** | MEDIUM | Min/max bbox scan before O(n) ray-cast. Early return if point outside ring bounds. |
| **Bulk WKB coordinate writes** | MEDIUM | On little-endian targets, `&[Coord<f64>]` reinterpreted as bytes for single `extend_from_slice` per ring. Falls back to per-coord on big-endian. |
| **Zero-copy f64 raster path** | MEDIUM | When `T` is already `f64`, `Cow::Borrowed` avoids allocating a conversion buffer in contour.rs. |

### Micro-optimizations

| Optimization | Location | Description |
|---|---|---|
| `#[inline]` on union-find `find`/`union` | union_find.rs | Called per-pixel in label loop; eliminates cross-module call overhead. |
| Union-find path halving | union_find.rs | Local parent/grandparent vars; fewer array indexings per iteration. |
| `#[inline]` on geometry functions | geometry.rs | `signed_area`, `point_in_ring` — called per ring in classification loops. |
| `#[inline]` on `normalize_polygon` | polygon.rs | Called per output polygon. |
| `#[inline]` on `write_ring`, `quantize`, `apply_transform` | arrow.rs, contour.rs | Hot-path helpers. |
| Hoisted `det` computation | contour.rs | Transform determinant computed once before threshold loop. |
| Pre-allocated segments Vec | contour.rs | `Vec::with_capacity(w * 2)` in `march_row` avoids early reallocations. |
| Pre-allocated HashMap | contour.rs | `HashMap::with_capacity(segments.len())` in `chain_segments`. |
| Pre-computed hole areas | contour.rs | Hole areas cached before multi-exterior assignment loop. |
| `extract_mask!` macro | lib.rs (Python) | Deduplicates 4 identical 8-line mask extraction blocks. |

## Supported Data Types

| Input dtype | Python | Precision |
|---|---|---|
| `uint8` | `np.uint8` | Lossless → f64 |
| `uint16` | `np.uint16` | Lossless → f64 |
| `uint32` | `np.uint32` | Lossless → f64 |
| `int16` | `np.int16` | Lossless → f64 |
| `int32` | `np.int32` | Lossless → f64 |
| `float32` | `np.float32` | Promoted to f64 |
| `float64` | `np.float64` | Native (zero-copy in contours) |

All output coordinates are `f64`. WKB encoding uses IEEE 754 little-endian doubles per OGC spec.

## Feature Flags (Rust)

| Flag | Default | Description |
|---|---|---|
| `arrow` | off (on in Python bindings) | Arrow RecordBatch export with WKB geometry + GeoParquet metadata |
| `cuda` | off | GPU-accelerated connected-component labeling (scaffolded, not yet compiled) |

## Benchmark Results

**Machine**: Apple M-series, release build, median of 5 runs after 2 warmup.

### Polygonize — `shapes_arrow()` vs rasterio (5 categorical values)

| Grid | `shapes()` | `arrow()` | rasterio | Speedup | Arrow Spd |
|---|---|---|---|---|---|
| 64x64 | 2.0ms | 544us | 4.1ms | 2.0x | **7.5x** |
| 128x128 | 8.8ms | 3.0ms | 16.1ms | 1.8x | **5.4x** |
| 256x256 | 52ms | 14.2ms | 76.4ms | 1.5x | **5.4x** |
| 512x512 | 268ms | 57.6ms | 330ms | 1.2x | **5.7x** |
| 1024x1024 | 1.08s | 226ms | 1.33s | 1.2x | **5.9x** |
| 2048x2048 | 4.37s | 898ms | 5.35s | 1.2x | **6.0x** |

### Contours — marching squares isobands (5 thresholds, random float32)

| Grid | `contours()` | `arrow()` |
|---|---|---|
| 64x64 | 2.4ms | 1.8ms |
| 128x128 | 13.6ms | 11.4ms |
| 256x256 | 104ms | 95.7ms |
| 512x512 | 1.09s | 1.03s |
| 1024x1024 | 14.5s | 14.2s |

Note: random float data is worst-case for isobands (maximizes ring count). Real-world continuous fields (DEMs, heatmaps) are much smoother and faster.

### Memory — Python-side peak allocation (tracemalloc)

| Grid | `shapes()` | `arrow()` | rasterio | Arrow reduction |
|---|---|---|---|---|
| 64x64 | 2.7MB | 2.9MB | 2.6MB | ~0% |
| 256x256 | 41.6MB | <0.1MB | 39.9MB | **~100%** |
| 512x512 | 166MB | <0.1MB | 159MB | **~100%** |
| 1024x1024 | 665MB | <0.1MB | 637MB | **~100%** |
| 2048x2048 | 2.66GB | <0.1MB | 2.55GB | **~100%** |

`shapes_arrow()` achieves near-zero Python-side allocation because all computation happens in Rust and the Arrow C Data Interface transfers ownership without copying.

### Dtype performance (1024x1024, `shapes_arrow`)

All input types perform within 8% of each other — dtype conversion overhead is negligible.

| dtype | time |
|---|---|
| uint8 | 222ms |
| uint16 | 223ms |
| uint32 | 221ms |
| int16 | 223ms |
| int32 | 236ms |
| float32 | 239ms |
| float64 | 238ms |

## CUDA Support (Scaffolded)

Pipeline: accept GPU pointer → run CCL kernel on device → transfer u32 label grid to CPU (4x smaller than f32 input) → boundary tracing on CPU.

Current status: kernel algorithm documented in `kernel.rs`, `cudarc` v0.12 linked, but actual PTX compilation not yet wired. The boundary tracing pass is inherently serial, so only Pass 1 (connected-component labeling) benefits from GPU acceleration.
