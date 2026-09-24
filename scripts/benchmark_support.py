"""Data and measurement helpers for the benchmark runner.

RSS deltas measure memory retained after a call, including its live result.
They do not measure the peak native-memory footprint during computation.
"""

import gc
import json
import statistics
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
    if warmup < 0 or repeats < 1:
        msg = "warmup must be nonnegative and repeats must be positive"
        raise ValueError(msg)
    gc.collect()
    last = None
    for _ in range(warmup):
        last = fn()
        del last

    gc.collect()
    times = []
    result = None
    for _ in range(repeats):
        # Release the previous output before timing another allocation.
        result = None
        gc.collect()
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return float(statistics.median(times)), result


def python_heap_peak_mb(fn) -> float:
    gc.collect()
    tracemalloc.start()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
        del result
    finally:
        tracemalloc.stop()
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

    mask = np.isfinite(dem)
    if nodata is not None:
        mask &= dem != nodata
    if not mask.any():
        msg = "DEM contains no finite, valid elevations"
        raise ValueError(msg)
    dem = np.where(mask, dem, np.nan)
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
        return lambda: contours(data, thresholds=thresholds, nodata=np.nan)
    if impl == "contours_arrow":
        return lambda: contours_arrow(data, thresholds=thresholds, nodata=np.nan)
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


def measure_process_rss_child(
    *,
    workload: str,
    impl: str,
    dataset: str,
    size: int | None,
) -> dict[str, float | int | str]:
    """Measure process RSS delta in a fresh interpreter."""
    import psutil

    fn = load_process_fn(workload, impl, dataset, size)
    if impl.endswith("_arrow"):
        # Bindings import PyArrow lazily; exclude that import from the baseline.
        import pyarrow  # noqa: F401

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


def measure_process_rss_subprocess(
    *,
    workload: str,
    impl: str,
    dataset: str,
    size: int | None,
) -> dict[str, float | int | str]:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("benchmark.py")),
        "--measure-process-rss",
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
