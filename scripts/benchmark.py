"""Benchmark contourrs timing and memory behavior.

Compares GeoJSON vs Arrow output for both polygonize and contour workloads.
Reports both Python-heap (`tracemalloc`) and process-peak RSS deltas.

Usage:
    uv run --python 3.13 --extra all python scripts/benchmark.py
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, cast

import numpy as np

POLYGONIZE_SIZES = [64, 128, 256, 512, 1024, 2048]
CONTOUR_SIZES = [64, 128, 256, 512, 1024]
POLYGONIZE_MEMORY_SIZES = [64, 256, 512, 1024, 2048]
CONTOUR_MEMORY_SIZES = [64, 128, 256, 512, 1024]
N_VALUES = 5
N_THRESHOLDS = [0.1, 0.25, 0.5, 0.75, 0.9]
WARMUP = 2
REPEATS = 5
REAL_WARMUP = 1
REAL_REPEATS = 3
MS_PER_SEC = 1000
MIN_VISIBLE_MB = 0.1
MB_PER_GB = 1024
CDL_PATH = Path("examples/data/cdl_2023_polk_512.tif")
DEM_PATH = Path("examples/data/mt_rainier_dem_2048.tif")


def log(msg: str = "") -> None:
    sys.stdout.write(f"{msg}\n")
    sys.stdout.flush()


def fmt_ms(ms: float) -> str:
    if ms < 1:
        return f"{ms * MS_PER_SEC:.0f}us"
    if ms < MS_PER_SEC:
        return f"{ms:.1f}ms"
    return f"{ms / MS_PER_SEC:.2f}s"


def fmt_mb(mb: float) -> str:
    if mb < MIN_VISIBLE_MB:
        return f"<{MIN_VISIBLE_MB:.1f}MB"
    if mb < MB_PER_GB:
        return f"{mb:.1f}MB"
    return f"{mb / MB_PER_GB:.2f}GB"


def bench(fn, *, warmup: int = WARMUP, repeats: int = REPEATS) -> tuple[float, object]:
    """Return median wall time in ms and the last result."""
    gc.collect()
    last = None
    for _ in range(warmup):
        last = fn()
        del last

    gc.collect()
    times = []
    result = None
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return sorted(times)[len(times) // 2], result


def python_heap_peak_mb(fn) -> float:
    gc.collect()
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return peak / (1024 * 1024)


def build_polygonize_data(size: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, N_VALUES, size=(size, size), dtype=np.uint8)


def build_contour_data(size: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((size, size)).astype(np.float32)


def load_real_cdl() -> np.ndarray:
    import rasterio

    with rasterio.open(CDL_PATH) as src:
        return src.read(1)


def load_real_dem() -> tuple[np.ndarray, list[float]]:
    import rasterio

    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1)
        nodata = src.nodata

    mask = np.ones(dem.shape, dtype=bool)
    if nodata is not None:
        mask &= dem != nodata
    vmin = float(dem[mask].min())
    vmax = float(dem[mask].max())
    thresholds = [float(x) for x in np.arange(np.ceil(vmin / 250) * 250, vmax, 250)]
    return dem, thresholds


def make_polygonize_fn(impl: str, data: np.ndarray):
    from contourrs import shapes, shapes_arrow

    if impl == "shapes":
        return lambda: shapes(data, connectivity=4)
    if impl == "shapes_arrow":
        return lambda: shapes_arrow(data, connectivity=4)
    if impl == "rasterio":
        from rasterio.features import shapes as rio_shapes

        return lambda: list(rio_shapes(data, connectivity=4))
    msg = f"Unsupported polygonize impl: {impl}"
    raise ValueError(msg)


def make_contour_fn(impl: str, data: np.ndarray, thresholds: list[float]):
    from contourrs import contours, contours_arrow

    if impl == "contours":
        return lambda: contours(data, thresholds=thresholds)
    if impl == "contours_arrow":
        return lambda: contours_arrow(data, thresholds=thresholds)
    msg = f"Unsupported contour impl: {impl}"
    raise ValueError(msg)


def result_size(result: object) -> int:
    if hasattr(result, "num_rows"):
        return int(cast("Any", result).num_rows)
    return len(cast("list[object]", result))


def synthetic_process_fn(workload: str, impl: str, size: int):
    if workload == "polygonize":
        data = build_polygonize_data(size)
        return make_polygonize_fn(impl, data)
    if workload == "contours":
        data = build_contour_data(size)
        return make_contour_fn(impl, data, N_THRESHOLDS)
    msg = f"Unsupported synthetic workload: {workload}"
    raise ValueError(msg)


def real_process_fn(workload: str, impl: str):
    if workload == "polygonize":
        data = load_real_cdl()
        return make_polygonize_fn(impl, data)
    if workload == "contours":
        data, thresholds = load_real_dem()
        return make_contour_fn(impl, data, thresholds)
    msg = f"Unsupported real workload: {workload}"
    raise ValueError(msg)


def load_process_fn(workload: str, impl: str, dataset: str, size: int | None):
    if dataset == "synthetic":
        if size is None:
            msg = f"size is required for synthetic {workload}"
            raise ValueError(msg)
        return synthetic_process_fn(workload, impl, size)
    if dataset == "real":
        return real_process_fn(workload, impl)
    msg = f"Unsupported dataset: {dataset}"
    raise ValueError(msg)


def measure_process_peak_child(
    *,
    workload: str,
    impl: str,
    dataset: str,
    size: int | None,
) -> dict[str, float | int | str]:
    """Measure process RSS delta in a fresh interpreter."""
    import psutil

    fn = load_process_fn(workload, impl, dataset, size)

    process = psutil.Process()
    gc.collect()
    rss_before = process.memory_info().rss / (1024 * 1024)
    result = fn()
    rss_after = process.memory_info().rss / (1024 * 1024)
    delta = max(0.0, rss_after - rss_before)
    rows = result_size(result)
    return {
        "workload": workload,
        "impl": impl,
        "dataset": dataset,
        "size": 0 if size is None else size,
        "rows": rows,
        "rss_delta_mb": delta,
    }


def measure_process_peak_subprocess(
    *,
    workload: str,
    impl: str,
    dataset: str,
    size: int | None,
) -> dict[str, float | int | str]:
    cmd = [
        sys.executable,
        __file__,
        "--measure-process-peak",
        "--workload",
        workload,
        "--impl",
        impl,
        "--dataset",
        dataset,
    ]
    if size is not None:
        cmd.extend(["--size", str(size)])
    proc = subprocess.run(  # noqa: S603
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def has_rasterio() -> bool:
    try:
        import rasterio  # noqa: F401
    except ImportError:
        return False
    return True


def bench_polygonize_timings() -> list[dict[str, object]]:
    rows = []
    include_rasterio = has_rasterio()

    log("=" * 80)
    log("POLYGONIZE TIMING (synthetic categorical raster)")
    log("=" * 80)

    parts = [f"{'Size':>10}", f"{'shapes()':>10}", f"{'arrow()':>10}"]
    if include_rasterio:
        parts.append(f"{'rasterio':>10}")
    parts.extend([f"{'vs shapes':>10}", f"{'vs rasterio':>11}"])
    header = " | ".join(parts)
    log(header)
    log("-" * len(header))

    for size in POLYGONIZE_SIZES:
        data = build_polygonize_data(size)
        ms_shapes, _shapes_result = bench(
            lambda d=data: make_polygonize_fn("shapes", d)(),
        )
        ms_arrow, arrow_result = bench(
            lambda d=data: make_polygonize_fn("shapes_arrow", d)(),
        )
        row = {
            "size": size,
            "shapes_ms": ms_shapes,
            "arrow_ms": ms_arrow,
            "rows": result_size(arrow_result),
        }

        sz = f"{size:>5}x{size:<4}"
        line = f"{sz} | {fmt_ms(ms_shapes):>10} | {fmt_ms(ms_arrow):>10}"
        if include_rasterio:
            ms_rio, rio_result = bench(
                lambda d=data: make_polygonize_fn("rasterio", d)(),
            )
            row["rasterio_ms"] = ms_rio
            row["rasterio_rows"] = result_size(rio_result)
            line += f" | {fmt_ms(ms_rio):>10}"
            line += f" | {ms_shapes / ms_arrow:>9.1f}x"
            line += f" | {ms_rio / ms_arrow:>10.1f}x"
        else:
            line += f" | {ms_shapes / ms_arrow:>9.1f}x | {'n/a':>10}"
        log(line)
        rows.append(row)

    log()
    return rows


def bench_contour_timings() -> list[dict[str, object]]:
    rows = []

    log("=" * 80)
    log("CONTOUR TIMING (synthetic float32 isobands)")
    log("=" * 80)
    header = (
        f"{'Size':>10} | {'contours()':>12} | {'arrow()':>12} | {'vs contours':>11}"
    )
    log(header)
    log("-" * len(header))

    for size in CONTOUR_SIZES:
        data = build_contour_data(size)
        ms_contours, contour_result = bench(
            lambda d=data: make_contour_fn("contours", d, N_THRESHOLDS)(),
        )
        ms_arrow, arrow_result = bench(
            lambda d=data: make_contour_fn("contours_arrow", d, N_THRESHOLDS)(),
        )
        log(
            f"{size:>5}x{size:<4} | {fmt_ms(ms_contours):>12}"
            f" | {fmt_ms(ms_arrow):>12} | {ms_contours / ms_arrow:>10.1f}x",
        )
        rows.append(
            {
                "size": size,
                "contours_ms": ms_contours,
                "arrow_ms": ms_arrow,
                "contours_rows": result_size(contour_result),
                "arrow_rows": result_size(arrow_result),
            },
        )

    log()
    return rows


def bench_polygonize_memory() -> tuple[
    list[dict[str, object]], list[dict[str, object]]
]:
    python_rows = []
    process_rows = []
    include_rasterio = has_rasterio()

    log("=" * 80)
    log("POLYGONIZE MEMORY (synthetic categorical raster)")
    log("=" * 80)
    log("Python heap peak (`tracemalloc`)")
    parts = [f"{'Size':>10}", f"{'shapes()':>10}", f"{'arrow()':>10}"]
    if include_rasterio:
        parts.extend([f"{'rasterio':>10}", f"{'Arrow Red.':>10}"])
    header = " | ".join(parts)
    log(header)
    log("-" * len(header))

    for size in POLYGONIZE_MEMORY_SIZES:
        data = build_polygonize_data(size)
        heap_shapes = python_heap_peak_mb(
            lambda d=data: make_polygonize_fn("shapes", d)(),
        )
        heap_arrow = python_heap_peak_mb(
            lambda d=data: make_polygonize_fn("shapes_arrow", d)(),
        )
        row = {
            "size": size,
            "shapes_mb": heap_shapes,
            "arrow_mb": heap_arrow,
        }
        line = (
            f"{size:>5}x{size:<4}"
            f" | {fmt_mb(heap_shapes):>10}"
            f" | {fmt_mb(heap_arrow):>10}"
        )
        if include_rasterio:
            heap_rio = python_heap_peak_mb(
                lambda d=data: make_polygonize_fn("rasterio", d)(),
            )
            reduction = 100 * (1 - heap_arrow / heap_rio) if heap_rio > 0 else 0.0
            row["rasterio_mb"] = heap_rio
            row["arrow_reduction_pct"] = reduction
            line += f" | {fmt_mb(heap_rio):>10} | {reduction:>9.0f}%"
        log(line)
        python_rows.append(row)

    log()
    log("Process RSS delta (fresh subprocess, post-import/post-data baseline)")
    parts = [f"{'Size':>10}", f"{'shapes()':>10}", f"{'arrow()':>10}"]
    if include_rasterio:
        parts.extend([f"{'rasterio':>10}", f"{'Arrow Red.':>10}"])
    header = " | ".join(parts)
    log(header)
    log("-" * len(header))

    for size in POLYGONIZE_MEMORY_SIZES:
        proc_shapes = measure_process_peak_subprocess(
            workload="polygonize",
            impl="shapes",
            dataset="synthetic",
            size=size,
        )
        proc_arrow = measure_process_peak_subprocess(
            workload="polygonize",
            impl="shapes_arrow",
            dataset="synthetic",
            size=size,
        )
        row = {
            "size": size,
            "shapes_mb": proc_shapes["rss_delta_mb"],
            "arrow_mb": proc_arrow["rss_delta_mb"],
        }
        line = (
            f"{size:>5}x{size:<4}"
            f" | {fmt_mb(float(proc_shapes['rss_delta_mb'])):>10}"
            f" | {fmt_mb(float(proc_arrow['rss_delta_mb'])):>10}"
        )
        if include_rasterio:
            proc_rio = measure_process_peak_subprocess(
                workload="polygonize",
                impl="rasterio",
                dataset="synthetic",
                size=size,
            )
            rio_mb = float(proc_rio["rss_delta_mb"])
            arrow_mb = float(proc_arrow["rss_delta_mb"])
            reduction = 100 * (1 - arrow_mb / rio_mb) if rio_mb > 0 else 0.0
            row["rasterio_mb"] = rio_mb
            row["arrow_reduction_pct"] = reduction
            line += f" | {fmt_mb(rio_mb):>10} | {reduction:>9.0f}%"
        log(line)
        process_rows.append(row)

    log()
    return python_rows, process_rows


def bench_contour_memory() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    python_rows = []
    process_rows = []

    log("=" * 80)
    log("CONTOUR MEMORY (synthetic float32 isobands)")
    log("=" * 80)
    log("Python heap peak (`tracemalloc`)")
    header = f"{'Size':>10} | {'contours()':>12} | {'arrow()':>12} | {'Arrow Red.':>10}"
    log(header)
    log("-" * len(header))

    for size in CONTOUR_MEMORY_SIZES:
        data = build_contour_data(size)
        heap_contours = python_heap_peak_mb(
            lambda d=data: make_contour_fn("contours", d, N_THRESHOLDS)(),
        )
        heap_arrow = python_heap_peak_mb(
            lambda d=data: make_contour_fn("contours_arrow", d, N_THRESHOLDS)(),
        )
        reduction = 100 * (1 - heap_arrow / heap_contours) if heap_contours > 0 else 0.0
        log(
            f"{size:>5}x{size:<4} | {fmt_mb(heap_contours):>12}"
            f" | {fmt_mb(heap_arrow):>12} | {reduction:>9.0f}%",
        )
        python_rows.append(
            {
                "size": size,
                "contours_mb": heap_contours,
                "arrow_mb": heap_arrow,
                "arrow_reduction_pct": reduction,
            },
        )

    log()
    log("Process RSS delta (fresh subprocess, post-import/post-data baseline)")
    header = f"{'Size':>10} | {'contours()':>12} | {'arrow()':>12} | {'Arrow Red.':>10}"
    log(header)
    log("-" * len(header))

    for size in CONTOUR_MEMORY_SIZES:
        proc_contours = measure_process_peak_subprocess(
            workload="contours",
            impl="contours",
            dataset="synthetic",
            size=size,
        )
        proc_arrow = measure_process_peak_subprocess(
            workload="contours",
            impl="contours_arrow",
            dataset="synthetic",
            size=size,
        )
        contour_mb = float(proc_contours["rss_delta_mb"])
        arrow_mb = float(proc_arrow["rss_delta_mb"])
        reduction = 100 * (1 - arrow_mb / contour_mb) if contour_mb > 0 else 0.0
        log(
            f"{size:>5}x{size:<4} | {fmt_mb(contour_mb):>12}"
            f" | {fmt_mb(arrow_mb):>12} | {reduction:>9.0f}%",
        )
        process_rows.append(
            {
                "size": size,
                "contours_mb": contour_mb,
                "arrow_mb": arrow_mb,
                "arrow_reduction_pct": reduction,
            },
        )

    log()
    return python_rows, process_rows


def bench_real_world() -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    include_rasterio = has_rasterio()

    log("=" * 80)
    log("REAL-WORLD WORKLOADS")
    log("=" * 80)

    if CDL_PATH.exists():
        cdl = load_real_cdl()
        ms_shapes, shapes_result = bench(
            lambda d=cdl: make_polygonize_fn("shapes", d)(),
            warmup=REAL_WARMUP,
            repeats=REAL_REPEATS,
        )
        ms_arrow, arrow_result = bench(
            lambda d=cdl: make_polygonize_fn("shapes_arrow", d)(),
            warmup=REAL_WARMUP,
            repeats=REAL_REPEATS,
        )
        row = {
            "dataset": "CDL 2023 Polk County 512x512",
            "shapes_ms": ms_shapes,
            "arrow_ms": ms_arrow,
            "rows": result_size(arrow_result),
        }
        if include_rasterio:
            ms_rio, rio_result = bench(
                lambda d=cdl: make_polygonize_fn("rasterio", d)(),
                warmup=REAL_WARMUP,
                repeats=REAL_REPEATS,
            )
            row["rasterio_ms"] = ms_rio
            row["rasterio_rows"] = result_size(rio_result)
            log(
                "CDL polygonize: "
                f"shapes={fmt_ms(ms_shapes)}, "
                f"arrow={fmt_ms(ms_arrow)}, "
                f"rasterio={fmt_ms(ms_rio)}, "
                f"rows={result_size(shapes_result):,}",
            )
        else:
            log(
                "CDL polygonize: "
                f"shapes={fmt_ms(ms_shapes)}, "
                f"arrow={fmt_ms(ms_arrow)}, "
                f"rows={result_size(shapes_result):,}",
            )
        results["cdl_polygonize"] = row

    if DEM_PATH.exists():
        dem, thresholds = load_real_dem()
        ms_contours, contour_result = bench(
            lambda d=dem: make_contour_fn("contours", d, thresholds)(),
            warmup=REAL_WARMUP,
            repeats=REAL_REPEATS,
        )
        ms_arrow, arrow_result = bench(
            lambda d=dem: make_contour_fn("contours_arrow", d, thresholds)(),
            warmup=REAL_WARMUP,
            repeats=REAL_REPEATS,
        )
        log(
            "Mt. Rainier contours: "
            f"contours={fmt_ms(ms_contours)}, "
            f"arrow={fmt_ms(ms_arrow)}, "
            f"rows={result_size(contour_result):,}",
        )
        results["rainier_contours"] = {
            "dataset": "Mt. Rainier DEM 2048x2048",
            "contours_ms": ms_contours,
            "arrow_ms": ms_arrow,
            "rows": result_size(arrow_result),
        }

    log()
    return results


def bench_dtypes() -> list[dict[str, object]]:
    from contourrs import shapes_arrow

    rows = []

    log("=" * 80)
    log("DTYPE TIMING (1024x1024, shapes_arrow)")
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
    for dtype in dtypes:
        data = base.astype(dtype)
        ms, _ = bench(lambda d=data: shapes_arrow(d, connectivity=4))
        log(f"{dtype.__name__!s:>10} | {fmt_ms(ms):>10}")
        rows.append({"dtype": dtype.__name__, "time_ms": ms})

    log()
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure-process-peak", action="store_true")
    parser.add_argument("--workload", choices=["polygonize", "contours"])
    parser.add_argument(
        "--impl",
        choices=["shapes", "shapes_arrow", "rasterio", "contours", "contours_arrow"],
    )
    parser.add_argument("--dataset", choices=["synthetic", "real"])
    parser.add_argument("--size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.measure_process_peak:
        payload = measure_process_peak_child(
            workload=args.workload,
            impl=args.impl,
            dataset=args.dataset,
            size=args.size,
        )
        sys.stdout.write(json.dumps(payload))
        return

    log(f"contourrs benchmark — {time.strftime('%Y-%m-%d %H:%M')}")
    log(f"Python {sys.version.split()[0]}")
    log(f"NumPy {np.__version__}")
    log()
    log("Methodology:")
    log("- timing: median of 5 runs after 2 warmup (real-world: 3 runs after 1 warmup)")
    log("- Python heap: tracemalloc peak")
    log("- process RSS: fresh subprocess delta above post-import/post-data baseline")
    log()

    bench_polygonize_timings()
    bench_contour_timings()
    bench_polygonize_memory()
    bench_contour_memory()
    bench_real_world()
    bench_dtypes()
    log("Done.")


if __name__ == "__main__":
    main()
