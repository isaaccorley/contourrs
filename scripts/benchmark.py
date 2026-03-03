"""Comprehensive benchmark: contourrs vs rasterio.

Measures wall time (median of N runs) and peak RSS memory for both
polygonize (discrete) and contour (continuous) workloads across grid sizes.

Usage:
    uv run python scripts/benchmark.py
"""

import gc
import resource
import sys
import time

import numpy as np


def log(msg=""):
    """Flush-safe print for non-TTY output."""
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRID_SIZES = [64, 128, 256, 512, 1024, 2048]
N_VALUES = 5  # categorical values for polygonize
N_THRESHOLDS = [0.1, 0.25, 0.5, 0.75, 0.9]  # isoband thresholds
MS_PER_SEC = 1000
WARMUP = 2
REPEATS = 5


def peak_rss_mb():
    """Current peak RSS in MiB (macOS/Linux)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports bytes, Linux reports KiB
    import sys

    if sys.platform == "darwin":
        return ru.ru_maxrss / (1024 * 1024)
    return ru.ru_maxrss / 1024


def bench(fn, warmup=WARMUP, repeats=REPEATS):
    """Return (median_ms, peak_rss_mb) for fn()."""
    gc.collect()
    for _ in range(warmup):
        fn()

    gc.collect()
    rss_before = peak_rss_mb()
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    rss_after = peak_rss_mb()
    median = sorted(times)[len(times) // 2]
    return median, rss_after - rss_before


def fmt_ms(ms):
    if ms < 1:
        return f"{ms * MS_PER_SEC:.0f}us"
    if ms < MS_PER_SEC:
        return f"{ms:.1f}ms"
    return f"{ms / MS_PER_SEC:.2f}s"


# ---------------------------------------------------------------------------
# Polygonize benchmark
# ---------------------------------------------------------------------------


def bench_polygonize():
    from contourrs import shapes, shapes_arrow

    try:
        from rasterio.features import shapes as rio_shapes

        has_rio = True
    except ImportError:
        has_rio = False
        log("  (rasterio not installed — skipping comparison)\n")

    log("=" * 80)
    log("POLYGONIZE BENCHMARK (discrete/categorical rasters)")
    log("=" * 80)

    parts = [f"{'Size':>10}", f"{'shapes()':>10}", f"{'arrow()':>10}"]
    if has_rio:
        parts += [f"{'rasterio':>10}", f"{'Speedup':>8}", f"{'Arrow Spd':>9}"]
    header = " | ".join(parts)
    log(header)
    log("-" * len(header))

    rng = np.random.default_rng(42)

    for i, size in enumerate(GRID_SIZES):
        log(f"  [{i + 1}/{len(GRID_SIZES)}] Benchmarking {size}x{size}...")
        data = rng.integers(0, N_VALUES, size=(size, size), dtype=np.uint8)

        ms_shapes, _ = bench(lambda d=data: shapes(d, connectivity=4))
        ms_arrow, _ = bench(lambda d=data: shapes_arrow(d, connectivity=4))

        sz = f"{size:>5}x{size:<4}"
        s = f"{sz} | {fmt_ms(ms_shapes):>10} | {fmt_ms(ms_arrow):>10}"
        if has_rio:
            ms_rio, _ = bench(
                lambda d=data: list(rio_shapes(d, connectivity=4)),
            )
            spd = ms_rio / ms_shapes if ms_shapes > 0 else float("inf")
            aspd = ms_rio / ms_arrow if ms_arrow > 0 else float("inf")
            s += f" | {fmt_ms(ms_rio):>10} | {spd:>7.1f}x | {aspd:>8.1f}x"
        log(s)

    log()


# ---------------------------------------------------------------------------
# Contour benchmark
# ---------------------------------------------------------------------------


def bench_contours():
    from contourrs import contours, contours_arrow

    log("=" * 80)
    log("CONTOUR BENCHMARK (continuous rasters, marching squares isobands)")
    log("=" * 80)

    header = f"{'Size':>10} | {'contours()':>12} | {'arrow()':>12}"
    log(header)
    log("-" * len(header))

    rng = np.random.default_rng(42)

    for i, size in enumerate(GRID_SIZES):
        log(f"  [{i + 1}/{len(GRID_SIZES)}] Benchmarking {size}x{size}...")
        data = rng.random((size, size)).astype(np.float32)

        ms_contours, _ = bench(lambda d=data: contours(d, thresholds=N_THRESHOLDS))
        ms_arrow, _ = bench(lambda d=data: contours_arrow(d, thresholds=N_THRESHOLDS))
        log(f"{size:>5}x{size:<4} | {fmt_ms(ms_contours):>12} | {fmt_ms(ms_arrow):>12}")

    log()


# ---------------------------------------------------------------------------
# Memory benchmark
# ---------------------------------------------------------------------------


def bench_memory():
    import tracemalloc

    from contourrs import shapes, shapes_arrow

    try:
        from rasterio.features import shapes as rio_shapes

        has_rio = True
    except ImportError:
        has_rio = False

    log("=" * 80)
    log("MEMORY BENCHMARK (peak Python-side allocation in MiB)")
    log("=" * 80)

    parts = [f"{'Size':>10}", f"{'shapes()':>10}", f"{'arrow()':>10}"]
    if has_rio:
        parts += [f"{'rasterio':>10}", f"{'Reduction':>10}"]
    header = " | ".join(parts)
    log(header)
    log("-" * len(header))

    rng = np.random.default_rng(42)

    mem_sizes = [64, 256, 512, 1024, 2048]
    for i, size in enumerate(mem_sizes):
        log(f"  [{i + 1}/{len(mem_sizes)}] Measuring {size}x{size}...")
        data = rng.integers(0, N_VALUES, size=(size, size), dtype=np.uint8)

        # Measure contourrs shapes()
        gc.collect()
        tracemalloc.start()
        _ = shapes(data, connectivity=4)
        _, peak_shapes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Measure contourrs shapes_arrow()
        gc.collect()
        tracemalloc.start()
        _ = shapes_arrow(data, connectivity=4)
        _, peak_arrow = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_shapes_mb = peak_shapes / (1024 * 1024)
        peak_arrow_mb = peak_arrow / (1024 * 1024)

        if has_rio:
            gc.collect()
            tracemalloc.start()
            _ = list(rio_shapes(data, connectivity=4))
            _, peak_rio = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_rio_mb = peak_rio / (1024 * 1024)

            reduction = (
                (1 - peak_arrow_mb / peak_rio_mb) * 100 if peak_rio_mb > 0 else 0
            )
            sz = f"{size:>5}x{size:<4}"
            log(
                f"{sz} | {peak_shapes_mb:>8.1f}MB"
                f" | {peak_arrow_mb:>8.1f}MB"
                f" | {peak_rio_mb:>8.1f}MB"
                f" | {reduction:>8.0f}%"
            )
        else:
            sz = f"{size:>5}x{size:<4}"
            log(f"{sz} | {peak_shapes_mb:>8.1f}MB | {peak_arrow_mb:>8.1f}MB")

    log()


# ---------------------------------------------------------------------------
# Dtype benchmark
# ---------------------------------------------------------------------------


def bench_dtypes():
    from contourrs import shapes_arrow

    log("=" * 80)
    log("DTYPE BENCHMARK (1024x1024, shapes_arrow)")
    log("=" * 80)

    header = f"{'dtype':>10} | {'time':>10}"
    log(header)
    log("-" * len(header))

    rng = np.random.default_rng(42)
    base = rng.integers(0, N_VALUES, size=(1024, 1024))

    dtypes = [
        np.uint8,
        np.uint16,
        np.uint32,
        np.int16,
        np.int32,
        np.float32,
        np.float64,
    ]
    for i, dtype in enumerate(dtypes):
        log(f"  [{i + 1}/{len(dtypes)}] {dtype.__name__}...")
        data = base.astype(dtype)
        ms, _ = bench(lambda d=data: shapes_arrow(d, connectivity=4))
        log(f"{dtype.__name__!s:>10} | {fmt_ms(ms):>10}")

    log()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log(f"\ncontourrs benchmark — {time.strftime('%Y-%m-%d %H:%M')}")
    log(f"NumPy {np.__version__}")
    log()

    bench_polygonize()
    bench_contours()
    bench_memory()
    bench_dtypes()

    log("Done.")
