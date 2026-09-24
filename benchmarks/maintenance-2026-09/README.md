# Maintenance measurements, September 2026

The union-find change reduced requested peak live heap on a uniform 1024 × 1024 raster from 9.000 to 5.127 MiB (43%).
Median execution time was 6.779 versus 7.243 ms.
The noisy categorical case showed no memory improvement.

Correct contour topology increased computation and output size.
In the final comparison at 256 × 256, noisy contours took 92.274 ms before correction and 131.730 ms after correction and indexing.
Requested peak live heap increased from 12.235 to 30.164 MiB, and polygon count increased from 9,644 to 19,254.
The baseline output omitted geometry, so these timings do not compare equivalent outputs.

The spatial index showed no convincing improvement in these bounded cases.
At 512 × 512, the corrected implementation took 834.974 ms without indexing and 826.320 ms with indexing, with 77,538 polygons in both outputs.
Indexing is retained to avoid scanning every exterior for every hole; these measurements do not establish a practical speedup.

## Measurement setup

Baseline source came from `6727cb73529bede6ebb1eeb0445d06964f687d1f`.
Both baseline and revised sources used the upgraded Cargo manifests and lockfile from this maintenance change, isolating source changes from dependency upgrades.
The host ran macOS 27.0 on arm64 with Cargo 1.98.1.
All runs used release builds, four Rayon threads, two warmups, and seven measured calls; reported values are medians.
Calls ran sequentially on the same host, without CPU isolation, so small timing changes should not be interpreted as improvements.

The harness uses float64 rasters, identity transforms, and four-connected polygonization.
Uniform rasters contain ones; categorical rasters have five labels.
A deterministic linear congruential generator starts at seed 42 for categorical and noisy inputs.
The smooth input is `(sin(8x) + cos(6y) + 2) / 4` for normalized raster coordinates.
Contour thresholds are `[0.1, 0.25, 0.5, 0.75, 0.9]`.
No Python conversion or Arrow export is included.

Timing runs use the system allocator without instrumentation.
Separate memory runs count allocator-requested live heap bytes above allocations live before each call.
These measurements exclude allocator metadata, stack memory, and thread allocations performed during warmup; they are not process RSS.
Memory-instrumented execution times must not be substituted for the separate timing results.

## Reproduction

Run `bash benchmarks/maintenance-2026-09/run.sh noise 256 timing` for timing or replace `timing` with `heap` for memory.
Cases are `uniform`, `categorical`, `smooth`, and `noise`; the second argument is raster side length.
The script creates isolated temporary snapshots, uses current dependency manifests for each, and preserves build outputs for inspection.
It compares the recorded baseline with current repository sources and with the current spatial index disabled.
The disabled-index variant uses the same linear containment search as the intermediate corrected implementation, though compiler output can differ from that historical snapshot.
Dependency downloads may be needed on the first run.

## Raw results

- `timings.csv` contains the initial uninstrumented baseline and corrected comparisons.
- `results.csv` contains the corresponding instrumented timings and peak heap.
- `indexed-results.csv` compares baseline, corrected (`current`), and corrected with indexing (`indexed`).
- `scaling-results.csv` compares corrected and indexed noisy contours at 512 × 512.
- `uniform-1024.csv` records the larger uniform polygonization comparison.

The `current` label in the saved CSVs refers to the corrected implementation before spatial indexing.
Timing rows have a zero or unused heap field because the memory counter was disabled.
