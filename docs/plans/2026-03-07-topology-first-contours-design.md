# Topology-first contour assembly

## Goal

Replace float-quantized segment stitching with integer grid-edge topology so contour assembly scales better on noisy rasters and avoids correctness risk from coordinate hashing.

## Current bottleneck

`contour.rs` currently:

1. runs marching squares and materializes interpolated float endpoints for every segment
2. quantizes those floats into hash keys
3. reconnects rings through a float-derived adjacency map

That creates extra CPU and memory pressure in the hottest contour path and makes ring closure depend on rounding policy.

## Proposed shape

- represent each crossing vertex by the grid edge it lies on, not by its interpolated float coordinate
- keep interpolated coordinates only as payload for final output points
- chain segments by exact integer edge keys (`top`, `right`, `bottom`, `left`) from the marching-squares cell topology

Concretely:

- add a small `VertexKey` enum for horizontal and vertical grid edges
- have `march_row()` emit `EdgeSegment { start, end }` where each endpoint carries both `VertexKey` and `Coord<f64>`
- rewrite `chain_segments()` to hash `VertexKey` directly instead of quantized `(i64, i64)` float coordinates

## Expected wins

- remove float quantization from the hot chain path
- reduce hash cost and branchiness during ring assembly
- make closure exact for shared cell edges
- create a cleaner base for a future streamed/tiled contour builder that never needs global float endpoint maps

## Risk / scope

- low API risk: output polygons stay unchanged
- medium implementation risk: saddle cases and ring orientation must still match current behavior
- validation: existing contour regression suite plus targeted synthetic contour benchmarks at larger grid sizes
