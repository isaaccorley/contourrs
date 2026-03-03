# Performance

All benchmarks on Apple M-series, release build, median of 5 runs after 2 warmup.

## Polygonize — shapes_arrow vs rasterio

5 categorical values, random raster.

| Grid | `shapes()` | `shapes_arrow()` | rasterio | Speedup | Arrow Speedup |
|---|---|---|---|---|---|
| 64x64 | 2.0ms | 544us | 4.1ms | 2.0x | **7.5x** |
| 128x128 | 8.8ms | 3.0ms | 16.1ms | 1.8x | **5.4x** |
| 256x256 | 52ms | 14.2ms | 76.4ms | 1.5x | **5.4x** |
| 512x512 | 268ms | 57.6ms | 330ms | 1.2x | **5.7x** |
| 1024x1024 | 1.08s | 226ms | 1.33s | 1.2x | **5.9x** |
| 2048x2048 | 4.37s | 898ms | 5.35s | 1.2x | **6.0x** |

`shapes_arrow()` eliminates Python GeoJSON dict construction — all geometry stays in Rust as WKB.

## Contours — marching squares isobands

5 thresholds, random float32 (worst case — maximizes ring count).

| Grid | `contours()` | `contours_arrow()` |
|---|---|---|
| 64x64 | 2.4ms | 1.8ms |
| 128x128 | 13.6ms | 11.4ms |
| 256x256 | 104ms | 95.7ms |
| 512x512 | 1.09s | 1.03s |
| 1024x1024 | 14.5s | 14.2s |

!!! note
    Random float data is worst-case for isobands (maximizes ring count). Real-world continuous fields (DEMs, heatmaps) are much smoother and faster.

## Memory — Python-side peak allocation

Measured with `tracemalloc`.

| Grid | `shapes()` | `shapes_arrow()` | rasterio | Arrow reduction |
|---|---|---|---|---|
| 64x64 | 2.7MB | 2.9MB | 2.6MB | ~0% |
| 256x256 | 41.6MB | <0.1MB | 39.9MB | **~100%** |
| 512x512 | 166MB | <0.1MB | 159MB | **~100%** |
| 1024x1024 | 665MB | <0.1MB | 637MB | **~100%** |
| 2048x2048 | 2.66GB | <0.1MB | 2.55GB | **~100%** |

`shapes_arrow()` achieves near-zero Python-side allocation because all computation happens in Rust and the Arrow C Data Interface transfers ownership without copying.

## Dtype performance

1024x1024 grid, `shapes_arrow()`. All input types perform within 8% of each other.

| dtype | time |
|---|---|
| uint8 | 222ms |
| uint16 | 223ms |
| uint32 | 221ms |
| int16 | 223ms |
| int32 | 236ms |
| float32 | 239ms |
| float64 | 238ms |
