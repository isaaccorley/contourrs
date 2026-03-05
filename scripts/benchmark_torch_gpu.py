"""CPU vs CUDA tensor polygonize benchmark.

This measures end-to-end latency for:

1) CPU baseline: `contourrs.shapes()` on a NumPy int32 array
2) GPU path: `contourrs.shapes_cuda()` on a CUDA tensor (direct tensor)
3) GPU DLPack path: `contourrs.shapes_cuda()` on a DLPack capsule

The CUDA path runs GPU connected-component labeling and CPU boundary tracing.

Usage:
    uv run --extra cuda python scripts/benchmark_torch_gpu.py
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch
from torch.utils import dlpack as torch_dlpack
from contourrs import shapes, shapes_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark contourrs CPU vs torch GPU->CPU pipeline",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="Square grid sizes to benchmark (default: 128 256 512 1024)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Timed repeats per size (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup runs per size (default: 2)",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        default=5,
        help="Number of categorical classes (default: 5)",
    )
    return parser.parse_args()


def bench(fn, warmup: int, repeats: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)

    return statistics.median(times), min(times), max(times)


def main() -> None:
    args = parse_args()

    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available in torch runtime")

    header = "size,cpu_ms,gpu_tensor_ms,gpu_dlpack_ms,tensor_ratio,dlpack_ratio"
    print(header)

    for n in args.sizes:
        rng = np.random.default_rng(42)
        arr_cpu = rng.integers(0, args.n_values, size=(n, n), dtype=np.int32)

        gpu_tensor = torch.from_numpy(arr_cpu).to("cuda")
        torch.cuda.synchronize()

        cpu_median, _, _ = bench(
            lambda: shapes(arr_cpu, connectivity=4),
            warmup=args.warmup,
            repeats=args.repeats,
        )

        def gpu_tensor_path() -> None:
            _ = shapes_cuda(gpu_tensor, connectivity=4)
            torch.cuda.synchronize()

        gpu_tensor_median, _, _ = bench(
            gpu_tensor_path,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        def gpu_dlpack_path() -> None:
            capsule = torch_dlpack.to_dlpack(gpu_tensor)
            _ = shapes_cuda(capsule, connectivity=4)
            torch.cuda.synchronize()

        gpu_dlpack_median, _, _ = bench(
            gpu_dlpack_path,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        ratio_tensor = gpu_tensor_median / cpu_median if cpu_median > 0 else float("inf")
        ratio_dlpack = gpu_dlpack_median / cpu_median if cpu_median > 0 else float("inf")
        print(
            f"{n}x{n},{cpu_median:.2f},{gpu_tensor_median:.2f},{gpu_dlpack_median:.2f},"
            f"{ratio_tensor:.2f}x,{ratio_dlpack:.2f}x",
        )


if __name__ == "__main__":
    main()
