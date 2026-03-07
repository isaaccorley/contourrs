# Performance

Published numbers below were collected on `Linux x86_64`, `Intel i7-10700K`, release build, `Python 3.13.5`, and `NumPy 2.4.2`. Reproduce with [`scripts/benchmark.py`](https://github.com/isaaccorley/contourrs/blob/main/scripts/benchmark.py).

## Methodology

- **Timing** — median of 5 runs after 2 warmup runs (`3 + 1` for the real-world cases)
- **Synthetic polygonize** — random `uint8` raster with 5 categorical values; intentionally noisy, high-boundary workload
- **Synthetic contours** — random `float32` raster with 5 thresholds; near worst-case for marching-squares ring count
- **Python heap** — peak allocation from `tracemalloc`
- **Process RSS delta** — fresh-subprocess RSS growth above a post-import, post-input-load baseline

!!! warning
    `tracemalloc` measures Python-managed heap only. It does **not** capture Rust heap allocations, Arrow/native buffers, or total process memory. `shapes_arrow()` and `contours_arrow()` dramatically reduce Python object construction, but they still allocate substantial native memory.

## Polygonize timing — synthetic categorical raster

`rasterio.features.shapes` is a thin wrapper around GDAL's `GDALPolygonize`, so this is the most direct public baseline for `shapes()` and `shapes_arrow()`.

| Grid | `shapes()` | `shapes_arrow()` | rasterio | `arrow()` vs `shapes()` | `arrow()` vs rasterio |
|---|---|---|---|---|---|
| 64x64 | 3.8ms | 873us | 8.0ms | **4.4x** | **9.2x** |
| 128x128 | 17.9ms | 3.9ms | 32.3ms | **4.6x** | **8.3x** |
| 256x256 | 88.2ms | 15.8ms | 147.5ms | **5.6x** | **9.4x** |
| 512x512 | 371.7ms | 62.8ms | 622.0ms | **5.9x** | **9.9x** |
| 1024x1024 | 1.50s | 377.4ms | 2.51s | **4.0x** | **6.7x** |
| 2048x2048 | 6.17s | 1.63s | 9.97s | **3.8x** | **6.1x** |

`shapes_arrow()` wins by avoiding Python GeoJSON dict/list construction and handing WKB buffers straight to PyArrow.

## Polygonize timing — real CDL raster

512x512 crop of the 2023 USDA Cropland Data Layer for Polk County, Iowa.

| Dataset | `shapes()` | `shapes_arrow()` | rasterio | output rows |
|---|---|---|---|---|
| CDL 2023 Polk County 512x512 | 102.0ms | **23.1ms** | 174.1ms | 34,130 |

On real land-cover data, `shapes_arrow()` is still about **4.4x** faster than `shapes()` and about **7.5x** faster than rasterio.

## Contour timing — synthetic float32 raster

These are `contours()` vs `contours_arrow()` on random float32 data with 5 thresholds. This is intentionally harsh on isoband extraction because it maximizes tiny rings.

| Grid | `contours()` | `contours_arrow()` | `arrow()` vs `contours()` |
|---|---|---|---|
| 64x64 | 3.8ms | 2.9ms | **1.3x** |
| 128x128 | 21.0ms | 17.4ms | **1.2x** |
| 256x256 | 172.0ms | 154.9ms | **1.1x** |
| 512x512 | 2.08s | 1.87s | **1.1x** |
| 1024x1024 | 33.98s | 32.46s | **1.0x** |

The contour path spends most of its time in marching squares and polygon assembly, so Arrow helps less than it does for categorical polygonization.

## Contour timing — real DEM

2048x2048 Mt. Rainier DEM with 250 m contour bands.

| Dataset | `contours()` | `contours_arrow()` | output rows |
|---|---|---|---|
| Mt. Rainier DEM 2048x2048 | 49.8ms | **47.7ms** | 186 |

Real DEMs are much smoother than random noise, so contour extraction is dramatically faster than the synthetic worst case.

## Polygonize memory — Python heap vs process RSS

### Python heap peak (`tracemalloc`)

| Grid | `shapes()` | `shapes_arrow()` | rasterio | `arrow()` reduction vs rasterio |
|---|---|---|---|---|
| 64x64 | 2.7MB | <0.1MB | 2.6MB | **100%** |
| 256x256 | 41.6MB | <0.1MB | 39.9MB | **100%** |
| 512x512 | 166.4MB | <0.1MB | 159.4MB | **100%** |
| 1024x1024 | 665.1MB | <0.1MB | 637.1MB | **100%** |
| 2048x2048 | 2.60GB | <0.1MB | 2.49GB | **100%** |

This is the source of the earlier `<0.1MB` claim: Arrow output nearly eliminates Python-side object churn.

### Process RSS delta (fresh subprocess)

| Grid | `shapes()` | `shapes_arrow()` | rasterio | `arrow()` reduction vs rasterio |
|---|---|---|---|---|
| 64x64 | 5.5MB | 2.7MB | 12.7MB | **79%** |
| 256x256 | 90.2MB | 10.6MB | 72.0MB | **85%** |
| 512x512 | 358.5MB | 184.8MB | 256.9MB | **28%** |
| 1024x1024 | 1.39GB | 714.0MB | 1002.8MB | **29%** |
| 2048x2048 | 5.58GB | 2.79GB | 3.90GB | **28%** |

This is the more important caveat: `shapes_arrow()` is **not** near-zero total memory. It still needs large native allocations for labeling, polygon tracing, WKB serialization, and Arrow buffers. The win is real, but it is closer to **~28-29% lower process RSS than rasterio** on larger grids, not ~100%.

## Contour memory — Python heap vs process RSS

### Python heap peak (`tracemalloc`)

| Grid | `contours()` | `contours_arrow()` | `arrow()` reduction |
|---|---|---|---|
| 64x64 | 0.9MB | <0.1MB | **100%** |
| 128x128 | 3.6MB | <0.1MB | **100%** |
| 256x256 | 14.5MB | <0.1MB | **100%** |
| 512x512 | 57.4MB | <0.1MB | **100%** |
| 1024x1024 | 228.4MB | <0.1MB | **100%** |

### Process RSS delta (fresh subprocess)

| Grid | `contours()` | `contours_arrow()` | `arrow()` reduction |
|---|---|---|---|
| 64x64 | 2.0MB | 1.8MB | **8%** |
| 128x128 | 8.6MB | 5.5MB | **36%** |
| 256x256 | 31.6MB | 15.6MB | **51%** |
| 512x512 | 118.6MB | 54.8MB | **54%** |
| 1024x1024 | 405.8MB | 216.6MB | **47%** |

For contours, Arrow reduces Python overhead and still lowers process memory, but the end-to-end savings are smaller than the `tracemalloc` table alone suggests.

## Dtype timing

1024x1024 grid, `shapes_arrow()`.

| dtype | time |
|---|---|
| uint8 | 253.0ms |
| uint16 | 255.8ms |
| uint32 | 253.6ms |
| int16 | 253.4ms |
| int32 | 255.2ms |
| float32 | 256.0ms |
| float64 | 254.1ms |

Input dtype has little effect on the Arrow polygonize path in this benchmark; all tested dtypes fall within about 1% of each other.
