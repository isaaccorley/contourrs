"""Benchmark raster workloads and optionally save measurements as JSON.

Run with ``uv run --extra dev python scripts/benchmark.py --output results.json``.
"""

import argparse
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from benchmark_support import (
    CDL_PATH,
    CONTOUR_MEMORY_SIZES,
    CONTOUR_SIZES,
    DEM_PATH,
    N_THRESHOLDS,
    N_VALUES,
    POLYGONIZE_MEMORY_SIZES,
    POLYGONIZE_SIZES,
    REAL_REPEATS,
    REAL_WARMUP,
    bench,
    build_contour_data,
    build_polygonize_data,
    fmt_mb,
    fmt_ms,
    has_rasterio,
    load_real_cdl,
    load_real_dem,
    log,
    make_contour_fn,
    make_polygonize_fn,
    measure_process_rss_child,
    measure_process_rss_subprocess,
    python_heap_peak_mb,
    result_size,
)


def bench_polygonize_timings() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
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
            make_polygonize_fn("shapes", data),
        )
        ms_arrow, arrow_result = bench(
            make_polygonize_fn("shapes_arrow", data),
        )
        row: dict[str, object] = {
            "size": size,
            "shapes_ms": ms_shapes,
            "arrow_ms": ms_arrow,
            "rows": result_size(arrow_result),
        }

        sz = f"{size:>5}x{size:<4}"
        line = f"{sz} | {fmt_ms(ms_shapes):>10} | {fmt_ms(ms_arrow):>10}"
        if include_rasterio:
            ms_rio, rio_result = bench(
                make_polygonize_fn("rasterio", data),
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
    rows: list[dict[str, object]] = []

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
            make_contour_fn("contours", data, N_THRESHOLDS),
        )
        ms_arrow, arrow_result = bench(
            make_contour_fn("contours_arrow", data, N_THRESHOLDS),
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
    python_rows: list[dict[str, object]] = []
    process_rows: list[dict[str, object]] = []
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
            make_polygonize_fn("shapes", data),
        )
        heap_arrow = python_heap_peak_mb(
            make_polygonize_fn("shapes_arrow", data),
        )
        row: dict[str, object] = {
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
                make_polygonize_fn("rasterio", data),
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
        proc_shapes = measure_process_rss_subprocess(
            workload="polygonize",
            impl="shapes",
            dataset="synthetic",
            size=size,
        )
        proc_arrow = measure_process_rss_subprocess(
            workload="polygonize",
            impl="shapes_arrow",
            dataset="synthetic",
            size=size,
        )
        row: dict[str, object] = {
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
            proc_rio = measure_process_rss_subprocess(
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
    python_rows: list[dict[str, object]] = []
    process_rows: list[dict[str, object]] = []

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
            make_contour_fn("contours", data, N_THRESHOLDS),
        )
        heap_arrow = python_heap_peak_mb(
            make_contour_fn("contours_arrow", data, N_THRESHOLDS),
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
        proc_contours = measure_process_rss_subprocess(
            workload="contours",
            impl="contours",
            dataset="synthetic",
            size=size,
        )
        proc_arrow = measure_process_rss_subprocess(
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
            make_polygonize_fn("shapes", cdl),
            warmup=REAL_WARMUP,
            repeats=REAL_REPEATS,
        )
        ms_arrow, arrow_result = bench(
            make_polygonize_fn("shapes_arrow", cdl),
            warmup=REAL_WARMUP,
            repeats=REAL_REPEATS,
        )
        row: dict[str, object] = {
            "dataset": "CDL 2023 Polk County 512x512",
            "shapes_ms": ms_shapes,
            "arrow_ms": ms_arrow,
            "rows": result_size(arrow_result),
        }
        if include_rasterio:
            ms_rio, rio_result = bench(
                make_polygonize_fn("rasterio", cdl),
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
            make_contour_fn("contours", dem, thresholds),
            warmup=REAL_WARMUP,
            repeats=REAL_REPEATS,
        )
        ms_arrow, arrow_result = bench(
            make_contour_fn("contours_arrow", dem, thresholds),
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

    rows: list[dict[str, object]] = []

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
    parser.add_argument(
        "--measure-process-rss",
        "--measure-process-peak",
        dest="measure_process_rss",
        action="store_true",
    )
    parser.add_argument(
        "--output", type=Path, help="Save measurements and environment as JSON"
    )
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
    if args.measure_process_rss:
        payload = measure_process_rss_child(
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

    results = {
        "environment": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "platform": platform.platform(),
            "python": sys.version,
            "versions": {
                name: version(name) for name in ("contourrs", "numpy", "pyarrow")
            },
            "timing": "median; 2 warmups + 5 repeats (real: 1 + 3)",
            "rss": (
                "retained RSS delta, not peak; "
                "fresh process after imports and input loading"
            ),
        },
        "polygonize_timings": bench_polygonize_timings(),
        "contour_timings": bench_contour_timings(),
        "polygonize_memory": bench_polygonize_memory(),
        "contour_memory": bench_contour_memory(),
        "real_world": bench_real_world(),
        "dtypes": bench_dtypes(),
    }
    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    log("Done.")


if __name__ == "__main__":
    main()
