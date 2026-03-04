//! Block-based (distributed) marching squares contouring.
//!
//! Tiles the raster into `block_size × block_size` blocks, marches isolines
//! within each block in parallel, then merges cross-block fragments into
//! closed rings. Improves cache locality and scaling for large rasters.

use std::borrow::Cow;
use std::collections::HashMap;

use geo_types::{Coord, LineString, Polygon};
use rayon::prelude::*;

use crate::contour::{apply_transform, grid_val, interp, quantize, BBox, EdgeSegment, PointKey};
use crate::geometry::{point_in_ring, signed_area};
use crate::polygon::normalize_polygon;
use crate::raster::{RasterGrid, RasterValue};
use crate::transform::AffineTransform;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type BlockResults = Vec<(Vec<LineString<f64>>, Vec<Fragment>)>;

// ---------------------------------------------------------------------------
// Block extent
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BlockExtent {
    row_start: i32,
    row_end: i32,
    col_start: i32,
    col_end: i32,
}

// ---------------------------------------------------------------------------
// Fragment: an open chain that touches a block boundary
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Fragment {
    coords: Vec<Coord<f64>>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate filled contour (isoband) polygons using block-based parallelism.
///
/// Same output as [`crate::contours`] but tiles the grid into blocks for
/// better cache locality and parallel scaling.
pub fn contours_blocked<T: RasterValue>(
    grid: &RasterGrid<T>,
    thresholds: &[f64],
    mask: Option<&[bool]>,
    transform: AffineTransform,
    block_size: usize,
) -> Vec<(Polygon<f64>, f64)> {
    if grid.width < 2 || grid.height < 2 || thresholds.len() < 2 {
        return Vec::new();
    }

    let mut thresholds: Vec<f64> = thresholds.to_vec();
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    thresholds.dedup();
    if thresholds.len() < 2 {
        return Vec::new();
    }

    // Pre-convert to f64
    let f64_data: Cow<[f64]> = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let ptr = grid.data.as_ptr() as *const f64;
        let len = grid.data.len();
        Cow::Borrowed(unsafe { std::slice::from_raw_parts(ptr, len) })
    } else {
        Cow::Owned(grid.data.iter().map(|v| v.to_f64_value()).collect())
    };
    let w = grid.width;
    let h = grid.height;
    let blocks = compute_blocks(w, h, block_size);

    let det = transform.a * transform.e - transform.b * transform.d;
    let mut result = Vec::new();

    // Cache hi-threshold block results for reuse as lo in next pair
    let mut cached_block_results: Option<BlockResults> = None;

    for pair in thresholds.windows(2) {
        let lo = pair[0];
        let hi = pair[1];

        // --- lo rings ---
        let lo_rings = match cached_block_results.take() {
            Some(block_results) => assemble_rings(block_results),
            None => {
                let block_results = march_all_blocks(&f64_data, w, h, &blocks, lo, mask);
                assemble_rings(block_results)
            }
        };

        // --- hi rings (march + cache raw block results) ---
        let hi_block_results = march_all_blocks(&f64_data, w, h, &blocks, hi, mask);
        let hi_rings = assemble_rings_from_ref(&hi_block_results);
        cached_block_results = Some(hi_block_results);

        if lo_rings.is_empty() {
            continue;
        }

        // Classify rings — same logic as contour.rs
        let mut exteriors: Vec<LineString<f64>> = Vec::new();
        let mut holes: Vec<LineString<f64>> = Vec::new();

        for ring in &lo_rings {
            let ring = apply_transform(ring, &transform);
            let area = signed_area(&ring);
            if area.abs() < f64::EPSILON {
                continue;
            }
            let is_iso_exterior = if det >= 0.0 { area < 0.0 } else { area > 0.0 };
            if is_iso_exterior {
                exteriors.push(ring);
            } else {
                holes.push(ring);
            }
        }

        for ring in &hi_rings {
            let ring = apply_transform(ring, &transform);
            let area = signed_area(&ring);
            if area.abs() < f64::EPSILON {
                continue;
            }
            let is_iso_exterior = if det >= 0.0 { area < 0.0 } else { area > 0.0 };
            if is_iso_exterior {
                holes.push(ring);
            }
        }

        if exteriors.is_empty() {
            continue;
        }

        // Assign holes to exteriors
        if exteriors.len() == 1 {
            let ext = exteriors.remove(0);
            let ext_area = signed_area(&ext).abs();
            let hole_area: f64 = holes.iter().map(|h| signed_area(h).abs()).sum();
            let net_area = ext_area - hole_area;
            if net_area > f64::EPSILON {
                let poly = Polygon::new(ext, holes);
                result.push((normalize_polygon(poly), lo));
            }
        } else {
            let hole_areas: Vec<f64> = holes.iter().map(|h| signed_area(h).abs()).collect();
            let ext_bboxes: Vec<BBox> = exteriors.iter().map(BBox::from_ring).collect();
            for (j, ext) in exteriors.iter().enumerate() {
                let mut my_holes = Vec::new();
                let mut my_hole_area = 0.0_f64;
                for (i, hole) in holes.iter().enumerate() {
                    if !hole.0.is_empty() {
                        let hp = &hole.0[0];
                        if ext_bboxes[j].contains_point(hp) && point_in_ring(hp, ext) {
                            my_holes.push(hole.clone());
                            my_hole_area += hole_areas[i];
                        }
                    }
                }
                let ext_area = signed_area(ext).abs();
                let net_area = ext_area - my_hole_area;
                if net_area > f64::EPSILON {
                    let poly = Polygon::new(ext.clone(), my_holes);
                    result.push((normalize_polygon(poly), lo));
                }
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Block tiling
// ---------------------------------------------------------------------------

fn compute_blocks(w: usize, h: usize, block_size: usize) -> Vec<BlockExtent> {
    // Extended grid spans rows -1..h and cols -1..w
    let row_start = -1_i32;
    let row_end = h as i32;
    let col_start = -1_i32;
    let col_end = w as i32;
    let bs = block_size as i32;

    let mut blocks = Vec::new();
    let mut r = row_start;
    while r < row_end {
        let r_end = (r + bs).min(row_end);
        let mut c = col_start;
        while c < col_end {
            let c_end = (c + bs).min(col_end);
            blocks.push(BlockExtent {
                row_start: r,
                row_end: r_end,
                col_start: c,
                col_end: c_end,
            });
            c = c_end;
        }
        r = r_end;
    }
    blocks
}

// ---------------------------------------------------------------------------
// March a single block
// ---------------------------------------------------------------------------

fn march_block(
    data: &[f64],
    w: usize,
    h: usize,
    block: &BlockExtent,
    threshold: f64,
    mask: Option<&[bool]>,
) -> Vec<EdgeSegment> {
    let mut segments = Vec::new();

    for row in block.row_start..block.row_end {
        let mut prev_tr = f64::NAN;
        let mut prev_br = f64::NAN;

        for col in block.col_start..block.col_end {
            let tl;
            let bl;
            if col == block.col_start {
                tl = grid_val(data, w, h, col, row, mask);
                bl = grid_val(data, w, h, col, row + 1, mask);
            } else {
                tl = prev_tr;
                bl = prev_br;
            }
            let tr = grid_val(data, w, h, col + 1, row, mask);
            let br = grid_val(data, w, h, col + 1, row + 1, mask);
            prev_tr = tr;
            prev_br = br;

            if tl.is_nan() || tr.is_nan() || bl.is_nan() || br.is_nan() {
                continue;
            }

            let tl_bit = u8::from(tl >= threshold);
            let tr_bit = u8::from(tr >= threshold);
            let br_bit = u8::from(br >= threshold);
            let bl_bit = u8::from(bl >= threshold);
            let code = tl_bit * 8 + tr_bit * 4 + br_bit * 2 + bl_bit;

            if code == 0 || code == 15 {
                continue;
            }

            let cx = col as f64;
            let cy = row as f64;

            let top = || Coord {
                x: cx + interp(tl, tr, threshold),
                y: cy,
            };
            let right = || Coord {
                x: cx + 1.0,
                y: cy + interp(tr, br, threshold),
            };
            let bottom = || Coord {
                x: cx + 1.0 - interp(br, bl, threshold),
                y: cy + 1.0,
            };
            let left = || Coord {
                x: cx,
                y: cy + 1.0 - interp(bl, tl, threshold),
            };

            match code {
                1 => segments.push(EdgeSegment {
                    start: bottom(),
                    end: left(),
                }),
                2 => segments.push(EdgeSegment {
                    start: right(),
                    end: bottom(),
                }),
                3 => segments.push(EdgeSegment {
                    start: right(),
                    end: left(),
                }),
                4 => segments.push(EdgeSegment {
                    start: top(),
                    end: right(),
                }),
                5 => {
                    let center = (tl + tr + br + bl) * 0.25;
                    if center >= threshold {
                        segments.push(EdgeSegment {
                            start: top(),
                            end: left(),
                        });
                        segments.push(EdgeSegment {
                            start: right(),
                            end: bottom(),
                        });
                    } else {
                        segments.push(EdgeSegment {
                            start: top(),
                            end: right(),
                        });
                        segments.push(EdgeSegment {
                            start: bottom(),
                            end: left(),
                        });
                    }
                }
                6 => segments.push(EdgeSegment {
                    start: top(),
                    end: bottom(),
                }),
                7 => segments.push(EdgeSegment {
                    start: top(),
                    end: left(),
                }),
                8 => segments.push(EdgeSegment {
                    start: left(),
                    end: top(),
                }),
                9 => segments.push(EdgeSegment {
                    start: bottom(),
                    end: top(),
                }),
                10 => {
                    let center = (tl + tr + br + bl) * 0.25;
                    if center >= threshold {
                        segments.push(EdgeSegment {
                            start: left(),
                            end: bottom(),
                        });
                        segments.push(EdgeSegment {
                            start: right(),
                            end: top(),
                        });
                    } else {
                        segments.push(EdgeSegment {
                            start: left(),
                            end: top(),
                        });
                        segments.push(EdgeSegment {
                            start: right(),
                            end: bottom(),
                        });
                    }
                }
                11 => segments.push(EdgeSegment {
                    start: right(),
                    end: top(),
                }),
                12 => segments.push(EdgeSegment {
                    start: left(),
                    end: right(),
                }),
                13 => segments.push(EdgeSegment {
                    start: bottom(),
                    end: right(),
                }),
                14 => segments.push(EdgeSegment {
                    start: left(),
                    end: bottom(),
                }),
                _ => {}
            }
        }
    }

    segments
}

// ---------------------------------------------------------------------------
// Chain segments within a block, partitioning into closed rings + fragments
// ---------------------------------------------------------------------------

fn chain_segments_partitioned(
    segments: &[EdgeSegment],
    _block: &BlockExtent,
) -> (Vec<LineString<f64>>, Vec<Fragment>) {
    if segments.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut endpoint_map: HashMap<PointKey, Vec<usize>> = HashMap::with_capacity(segments.len());
    for (i, seg) in segments.iter().enumerate() {
        endpoint_map
            .entry(quantize(&seg.start))
            .or_default()
            .push(i);
    }

    let mut used = vec![false; segments.len()];
    let mut rings = Vec::new();
    let mut fragments = Vec::new();

    for start_idx in 0..segments.len() {
        if used[start_idx] {
            continue;
        }

        let mut coords = Vec::new();
        used[start_idx] = true;
        coords.push(segments[start_idx].start);
        coords.push(segments[start_idx].end);

        loop {
            let end_key = quantize(coords.last().unwrap());
            let start_key = quantize(&coords[0]);

            if coords.len() > 2 && end_key == start_key {
                *coords.last_mut().unwrap() = coords[0];
                break;
            }

            let next = endpoint_map
                .get(&end_key)
                .and_then(|candidates| candidates.iter().find(|&&idx| !used[idx]).copied());

            match next {
                Some(idx) => {
                    used[idx] = true;
                    coords.push(segments[idx].end);
                }
                None => break,
            }
        }

        if coords.len() < 2 {
            continue;
        }

        let first = coords[0];
        let last = *coords.last().unwrap();
        if coords.len() >= 4 && quantize(&first) == quantize(&last) {
            // Closed ring
            *coords.last_mut().unwrap() = first;
            rings.push(LineString(coords));
        } else if coords.len() >= 2 {
            // Open fragment — needs cross-block merging
            fragments.push(Fragment { coords });
        }
    }

    (rings, fragments)
}

// ---------------------------------------------------------------------------
// Merge fragments across block boundaries
// ---------------------------------------------------------------------------

fn merge_fragments(mut fragments: Vec<Fragment>) -> Vec<LineString<f64>> {
    if fragments.is_empty() {
        return Vec::new();
    }

    // Build map: quantized start point → fragment index
    let mut start_map: HashMap<PointKey, Vec<usize>> = HashMap::with_capacity(fragments.len());
    for (i, frag) in fragments.iter().enumerate() {
        let key = quantize(&frag.coords[0]);
        start_map.entry(key).or_default().push(i);
    }

    let mut used = vec![false; fragments.len()];
    let mut rings = Vec::new();

    for seed in 0..fragments.len() {
        if used[seed] {
            continue;
        }
        used[seed] = true;

        // Start with this fragment's coords
        let mut chain: Vec<Coord<f64>> = std::mem::take(&mut fragments[seed].coords);

        loop {
            let end_key = quantize(chain.last().unwrap());
            let start_key = quantize(&chain[0]);

            // Check closure
            if chain.len() > 2 && end_key == start_key {
                *chain.last_mut().unwrap() = chain[0];
                break;
            }

            // Find fragment whose start matches our end
            let next = start_map
                .get(&end_key)
                .and_then(|candidates| candidates.iter().find(|&&idx| !used[idx]).copied());

            match next {
                Some(idx) => {
                    used[idx] = true;
                    let other = std::mem::take(&mut fragments[idx].coords);
                    // Skip the first coord of `other` (it matches our end)
                    chain.extend_from_slice(&other[1..]);
                }
                None => break,
            }
        }

        if chain.len() >= 4 {
            let first = chain[0];
            let last = *chain.last().unwrap();
            if quantize(&first) == quantize(&last) {
                *chain.last_mut().unwrap() = first;
                rings.push(LineString(chain));
            }
        }
    }

    rings
}

// ---------------------------------------------------------------------------
// Orchestration helpers
// ---------------------------------------------------------------------------

/// March all blocks in parallel, return per-block (closed_rings, fragments).
fn march_all_blocks(
    data: &[f64],
    w: usize,
    h: usize,
    blocks: &[BlockExtent],
    threshold: f64,
    mask: Option<&[bool]>,
) -> BlockResults {
    blocks
        .par_iter()
        .map(|block| {
            let segments = march_block(data, w, h, block, threshold, mask);
            chain_segments_partitioned(&segments, block)
        })
        .collect()
}

/// Assemble rings from block results (consuming).
fn assemble_rings(block_results: BlockResults) -> Vec<LineString<f64>> {
    let mut all_rings = Vec::new();
    let mut all_fragments = Vec::new();

    for (rings, fragments) in block_results {
        all_rings.extend(rings);
        all_fragments.extend(fragments);
    }

    let merged = merge_fragments(all_fragments);
    all_rings.extend(merged);
    all_rings
}

/// Assemble rings from block results (borrowing — re-chains from segments).
/// Used when we need to keep the block results for caching.
fn assemble_rings_from_ref(
    block_results: &[(Vec<LineString<f64>>, Vec<Fragment>)],
) -> Vec<LineString<f64>> {
    let mut all_rings = Vec::new();
    let mut all_fragments = Vec::new();

    for (rings, fragments) in block_results {
        all_rings.extend(rings.iter().cloned());
        // Fragments need to be cloned for merging
        all_fragments.extend(fragments.iter().map(|f| Fragment {
            coords: f.coords.clone(),
        }));
    }

    let merged = merge_fragments(all_fragments);
    all_rings.extend(merged);
    all_rings
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contour::contours;

    fn make_grid(data: &[f64], w: usize, h: usize) -> RasterGrid<'_, f64> {
        RasterGrid::new(data, w, h)
    }

    /// Sort polygons by (value, area) for deterministic comparison.
    fn sorted_polys(mut polys: Vec<(Polygon<f64>, f64)>) -> Vec<(Polygon<f64>, f64)> {
        polys.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let aa = signed_area(a.0.exterior()).abs();
                    let ab = signed_area(b.0.exterior()).abs();
                    aa.partial_cmp(&ab).unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        polys
    }

    fn assert_equivalent(a: &[(Polygon<f64>, f64)], b: &[(Polygon<f64>, f64)], label: &str) {
        assert_eq!(
            a.len(),
            b.len(),
            "{}: polygon count mismatch: {} vs {}",
            label,
            a.len(),
            b.len()
        );

        for (i, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(pa.1, pb.1, "{}: polygon {} value mismatch", label, i);
            let area_a = signed_area(pa.0.exterior()).abs();
            let area_b = signed_area(pb.0.exterior()).abs();
            let diff = (area_a - area_b).abs();
            assert!(
                diff < 1e-6,
                "{}: polygon {} area mismatch: {} vs {} (diff={})",
                label,
                i,
                area_a,
                area_b,
                diff
            );
            assert_eq!(
                pa.0.interiors().len(),
                pb.0.interiors().len(),
                "{}: polygon {} hole count mismatch",
                label,
                i
            );
        }
    }

    #[test]
    fn test_compute_blocks() {
        let blocks = compute_blocks(10, 10, 4);
        // Extended grid: -1..10 rows, -1..10 cols → 11 units each
        // 4-wide blocks → ceil(11/4) = 3 per axis → 9 blocks
        assert_eq!(blocks.len(), 9);
        assert_eq!(blocks[0].row_start, -1);
        assert_eq!(blocks[0].col_start, -1);
    }

    #[test]
    fn test_blocked_matches_flat_inside_band() {
        let data = vec![5.0; 9];
        let grid = make_grid(&data, 3, 3);
        let thresholds = &[3.0, 7.0];
        let expected = contours(&grid, thresholds, None, AffineTransform::identity());
        let blocked = contours_blocked(&grid, thresholds, None, AffineTransform::identity(), 2);
        let expected = sorted_polys(expected);
        let blocked = sorted_polys(blocked);
        assert_equivalent(&expected, &blocked, "flat_inside_band");
    }

    #[test]
    fn test_blocked_matches_linear_gradient() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0];
        let grid = make_grid(&data, 4, 2);
        let thresholds = &[0.5, 1.5, 2.5];
        let expected = contours(&grid, thresholds, None, AffineTransform::identity());
        let blocked = contours_blocked(&grid, thresholds, None, AffineTransform::identity(), 2);
        let expected = sorted_polys(expected);
        let blocked = sorted_polys(blocked);
        assert_equivalent(&expected, &blocked, "linear_gradient");
    }

    #[test]
    fn test_blocked_matches_donut() {
        #[rustfmt::skip]
        let data = vec![
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 2.0, 2.0, 0.0,
            0.0, 2.0, 5.0, 2.0, 0.0,
            0.0, 2.0, 2.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let grid = make_grid(&data, 5, 5);
        let thresholds = &[1.0, 3.0];
        let expected = contours(&grid, thresholds, None, AffineTransform::identity());
        let blocked = contours_blocked(&grid, thresholds, None, AffineTransform::identity(), 2);
        let expected = sorted_polys(expected);
        let blocked = sorted_polys(blocked);
        assert_equivalent(&expected, &blocked, "donut");
    }

    #[test]
    fn test_blocked_matches_center_peak() {
        let data = vec![0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0];
        let grid = make_grid(&data, 3, 3);
        let thresholds = &[2.0, 8.0];
        let expected = contours(&grid, thresholds, None, AffineTransform::identity());
        let blocked = contours_blocked(&grid, thresholds, None, AffineTransform::identity(), 2);
        let expected = sorted_polys(expected);
        let blocked = sorted_polys(blocked);
        assert_equivalent(&expected, &blocked, "center_peak");
    }

    #[test]
    fn test_blocked_matches_gaussian() {
        // Larger grid to exercise multi-block merging
        let size = 16;
        let mut data = vec![0.0; size * size];
        let center = size as f64 / 2.0;
        for r in 0..size {
            for c in 0..size {
                let dx = c as f64 - center;
                let dy = r as f64 - center;
                data[r * size + c] = (-0.05 * (dx * dx + dy * dy)).exp();
            }
        }
        let grid = make_grid(&data, size, size);
        let thresholds = &[0.1, 0.3, 0.5, 0.7, 0.9];

        for bs in [2, 4, 8] {
            let expected = contours(&grid, thresholds, None, AffineTransform::identity());
            let blocked =
                contours_blocked(&grid, thresholds, None, AffineTransform::identity(), bs);
            let expected = sorted_polys(expected);
            let blocked = sorted_polys(blocked);
            assert_equivalent(&expected, &blocked, &format!("gaussian_bs{}", bs));
        }
    }

    #[test]
    fn test_blocked_empty_grid() {
        let data = vec![1.0; 4];
        let grid = make_grid(&data, 2, 2);
        let result = contours_blocked(&grid, &[5.0, 10.0], None, AffineTransform::identity(), 2);
        assert!(result.is_empty());
    }

    #[test]
    fn test_blocked_with_transform() {
        let data = vec![0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0];
        let grid = make_grid(&data, 3, 3);
        let transform = AffineTransform::new(10.0, 0.0, 100.0, 0.0, -10.0, 200.0);
        let expected = contours(&grid, &[2.0, 8.0], None, transform);
        let blocked = contours_blocked(&grid, &[2.0, 8.0], None, transform, 2);
        let expected = sorted_polys(expected);
        let blocked = sorted_polys(blocked);
        assert_equivalent(&expected, &blocked, "with_transform");
    }
}
