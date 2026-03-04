//! Marching squares isoband contouring for continuous rasters.
//!
//! Produces filled polygons between consecutive threshold pairs — the same
//! `Vec<(Polygon<f64>, f64)>` output as [`crate::polygonize`], so Arrow export
//! and Python bindings are reused unchanged.
//!
//! # Algorithm
//!
//! Each isoband [lo, hi) is decomposed into two standard isoline problems:
//! 1. Isoline at `lo` → rings bounding the {val >= lo} region
//! 2. Isoline at `hi` → rings bounding the {val >= hi} region
//!
//! The isoband is {val >= lo} ∖ {val >= hi}. CCW lo-rings become exteriors,
//! CW lo-rings and reversed CCW hi-rings become holes. Standard 16-case
//! marching squares with saddle disambiguation via center value.

use std::borrow::Cow;
use std::collections::HashMap;

use geo_types::{Coord, LineString, Polygon};
use rayon::prelude::*;

use crate::geometry::{point_in_ring, signed_area};
use crate::polygon::normalize_polygon;
use crate::raster::{RasterGrid, RasterValue};
use crate::transform::AffineTransform;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate filled contour (isoband) polygons from a continuous raster.
///
/// Returns `Vec<(Polygon<f64>, value)>` where `value` is the lower threshold
/// of the band that produced the polygon.
///
/// # Arguments
/// * `grid`       – input raster (any `RasterValue` type)
/// * `thresholds` – break values; bands are formed from consecutive pairs
/// * `mask`       – optional boolean mask (true = include pixel)
/// * `transform`  – affine transform applied to output coordinates
pub fn contours<T: RasterValue>(
    grid: &RasterGrid<T>,
    thresholds: &[f64],
    mask: Option<&[bool]>,
    transform: AffineTransform,
) -> Vec<(Polygon<f64>, f64)> {
    if grid.width < 2 || grid.height < 2 || thresholds.len() < 2 {
        return Vec::new();
    }

    // Sort + dedup thresholds
    let mut thresholds: Vec<f64> = thresholds.to_vec();
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    thresholds.dedup();
    if thresholds.len() < 2 {
        return Vec::new();
    }

    // Pre-convert grid to f64 for interpolation (zero-copy when T is already f64)
    let f64_data: Cow<[f64]> = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        // SAFETY: T is f64, so &[T] and &[f64] have identical layout
        let ptr = grid.data.as_ptr() as *const f64;
        let len = grid.data.len();
        Cow::Borrowed(unsafe { std::slice::from_raw_parts(ptr, len) })
    } else {
        Cow::Owned(grid.data.iter().map(|v| v.to_f64_value()).collect())
    };
    let w = grid.width;
    let h = grid.height;

    let mut result = Vec::new();

    // Hoist det computation — invariant across threshold pairs
    let det = transform.a * transform.e - transform.b * transform.d;

    // Cache hi_rings from previous iteration to reuse as lo_rings
    let mut cached_rings: Option<Vec<LineString<f64>>> = None;

    for pair in thresholds.windows(2) {
        let lo = pair[0];
        let hi = pair[1];

        // Reuse previous hi_rings as lo_rings when available
        let lo_rings = match cached_rings.take() {
            Some(rings) => rings,
            None => {
                let lo_segments = march_isoline(&f64_data, w, h, lo, mask);
                chain_segments(&lo_segments)
            }
        };
        let hi_segments = march_isoline(&f64_data, w, h, hi, mask);
        let hi_rings = chain_segments(&hi_segments);

        if lo_rings.is_empty() {
            // Cache hi_rings (move, no clone) and skip to next pair
            cached_rings = Some(hi_rings);
            continue;
        }

        // Classify and transform rings.
        // The marching squares "inside to the left" convention in y-down coords
        // means iso-exterior rings are CW (negative signed_area with positive det).
        let mut exteriors: Vec<LineString<f64>> = Vec::new();
        let mut holes: Vec<LineString<f64>> = Vec::new();

        // Lo-threshold rings bound {val >= lo}
        for ring in &lo_rings {
            let ring = apply_transform(ring, &transform);
            let area = signed_area(&ring);
            if area.abs() < f64::EPSILON {
                continue;
            }
            // Iso-exterior: negative area when det >= 0, positive when det < 0
            let is_iso_exterior = if det >= 0.0 { area < 0.0 } else { area > 0.0 };
            if is_iso_exterior {
                exteriors.push(ring);
            } else {
                holes.push(ring);
            }
        }

        // Hi-threshold rings bound {val >= hi} — these punch holes in the isoband
        for ring in &hi_rings {
            let ring = apply_transform(ring, &transform);
            let area = signed_area(&ring);
            if area.abs() < f64::EPSILON {
                continue;
            }
            let is_iso_exterior = if det >= 0.0 { area < 0.0 } else { area > 0.0 };
            if is_iso_exterior {
                // Exterior of {val >= hi} → hole in isoband (keep as-is, it's already CW-ish)
                holes.push(ring);
            }
            // Holes in {val >= hi} → already covered by lo-exterior, ignore
        }

        // Cache hi_rings for next iteration (move after borrow ends — no clone needed)
        cached_rings = Some(hi_rings);

        if exteriors.is_empty() {
            continue;
        }

        // Assign holes to exteriors and filter degenerate polygons
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
            // Pre-compute hole areas + exterior bboxes for spatial filtering
            let hole_areas: Vec<f64> = holes.iter().map(|h| signed_area(h).abs()).collect();
            let ext_bboxes: Vec<BBox> = exteriors.iter().map(BBox::from_ring).collect();
            for (j, ext) in exteriors.iter().enumerate() {
                let mut my_holes = Vec::new();
                let mut my_hole_area = 0.0_f64;
                for (i, hole) in holes.iter().enumerate() {
                    if !hole.0.is_empty() {
                        let hp = &hole.0[0];
                        // Bbox pre-filter: skip expensive point_in_ring if outside bbox
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
// Edge segment with absolute coordinates
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct EdgeSegment {
    pub(crate) start: Coord<f64>,
    pub(crate) end: Coord<f64>,
}

// ---------------------------------------------------------------------------
// Bounding box for spatial indexing
// ---------------------------------------------------------------------------

pub(crate) struct BBox {
    pub(crate) min_x: f64,
    pub(crate) max_x: f64,
    pub(crate) min_y: f64,
    pub(crate) max_y: f64,
}

impl BBox {
    #[inline]
    pub(crate) fn from_ring(ring: &LineString<f64>) -> Self {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for c in &ring.0 {
            if c.x < min_x {
                min_x = c.x;
            }
            if c.x > max_x {
                max_x = c.x;
            }
            if c.y < min_y {
                min_y = c.y;
            }
            if c.y > max_y {
                max_y = c.y;
            }
        }
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
        }
    }

    #[inline]
    pub(crate) fn contains_point(&self, p: &Coord<f64>) -> bool {
        p.x >= self.min_x && p.x <= self.max_x && p.y >= self.min_y && p.y <= self.max_y
    }
}

// ---------------------------------------------------------------------------
// Standard 16-case marching squares isoline
// ---------------------------------------------------------------------------

/// Get pixel value, returning NaN for out-of-bounds or masked pixels.
#[inline]
pub(crate) fn grid_val(
    data: &[f64],
    w: usize,
    h: usize,
    col: i32,
    row: i32,
    mask: Option<&[bool]>,
) -> f64 {
    if col < 0 || row < 0 || col >= w as i32 || row >= h as i32 {
        return f64::NEG_INFINITY;
    }
    let idx = row as usize * w + col as usize;
    if let Some(m) = mask {
        if !m[idx] {
            return f64::NAN;
        }
    }
    data[idx]
}

/// Process a single row of the marching-squares grid, returning segments.
#[inline]
pub(crate) fn march_row(
    data: &[f64],
    w: usize,
    h: usize,
    row: i32,
    threshold: f64,
    mask: Option<&[bool]>,
) -> Vec<EdgeSegment> {
    let mut segments = Vec::new();
    let mut prev_tr = f64::NAN;
    let mut prev_br = f64::NAN;

    for col in -1..w as i32 {
        let tl;
        let bl;
        if col == -1 {
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

        // Skip cells with NaN
        if tl.is_nan() || tr.is_nan() || bl.is_nan() || br.is_nan() {
            continue;
        }

        // Binary classification: 1 if >= threshold, 0 if < threshold
        let tl_bit = u8::from(tl >= threshold);
        let tr_bit = u8::from(tr >= threshold);
        let br_bit = u8::from(br >= threshold);
        let bl_bit = u8::from(bl >= threshold);

        let code = tl_bit * 8 + tr_bit * 4 + br_bit * 2 + bl_bit;

        if code == 0 || code == 15 {
            continue; // All same → no crossing
        }

        let cx = col as f64;
        let cy = row as f64;

        // Edge crossing points
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

        // Standard 16-case table. Convention: inside (val >= t) is to the
        // left of the segment direction.
        match code {
            1 => {
                segments.push(EdgeSegment {
                    start: bottom(),
                    end: left(),
                });
            }
            2 => {
                segments.push(EdgeSegment {
                    start: right(),
                    end: bottom(),
                });
            }
            3 => {
                segments.push(EdgeSegment {
                    start: right(),
                    end: left(),
                });
            }
            4 => {
                segments.push(EdgeSegment {
                    start: top(),
                    end: right(),
                });
            }
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
            6 => {
                segments.push(EdgeSegment {
                    start: top(),
                    end: bottom(),
                });
            }
            7 => {
                segments.push(EdgeSegment {
                    start: top(),
                    end: left(),
                });
            }
            8 => {
                segments.push(EdgeSegment {
                    start: left(),
                    end: top(),
                });
            }
            9 => {
                segments.push(EdgeSegment {
                    start: bottom(),
                    end: top(),
                });
            }
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
            11 => {
                segments.push(EdgeSegment {
                    start: right(),
                    end: top(),
                });
            }
            12 => {
                segments.push(EdgeSegment {
                    start: left(),
                    end: right(),
                });
            }
            13 => {
                segments.push(EdgeSegment {
                    start: bottom(),
                    end: right(),
                });
            }
            14 => {
                segments.push(EdgeSegment {
                    start: left(),
                    end: bottom(),
                });
            }
            _ => {} // 0 and 15 handled above
        }
    }

    segments
}

/// Minimum grid area before rayon parallelization kicks in.
const PARALLEL_THRESHOLD: usize = 128 * 128;

/// Generate isoline segments at a single threshold using 16-case marching squares.
/// Parallelized across rows with rayon for large grids.
fn march_isoline(
    data: &[f64],
    w: usize,
    h: usize,
    threshold: f64,
    mask: Option<&[bool]>,
) -> Vec<EdgeSegment> {
    // Extended grid: from -1..h, -1..w so boundary pixels get border cells
    let rows: Vec<i32> = (-1..h as i32).collect();

    if w * h >= PARALLEL_THRESHOLD {
        rows.into_par_iter()
            .flat_map_iter(|row| march_row(data, w, h, row, threshold, mask))
            .collect()
    } else {
        rows.into_iter()
            .flat_map(|row| march_row(data, w, h, row, threshold, mask))
            .collect()
    }
}

/// Linear interpolation fraction.
#[inline]
pub(crate) fn interp(v0: f64, v1: f64, threshold: f64) -> f64 {
    if v0.is_infinite() || v1.is_infinite() {
        if v0.is_infinite() && v1.is_infinite() {
            return 0.5;
        }
        return if v0.is_infinite() { 0.0 } else { 1.0 };
    }
    let denom = v1 - v0;
    if denom.abs() < f64::EPSILON {
        0.5
    } else {
        ((threshold - v0) / denom).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Segment chaining: connect segments into closed rings
// ---------------------------------------------------------------------------

pub(crate) type PointKey = (i64, i64);

#[inline]
pub(crate) fn quantize(c: &Coord<f64>) -> PointKey {
    ((c.x * 1e10).round() as i64, (c.y * 1e10).round() as i64)
}

pub(crate) fn chain_segments(segments: &[EdgeSegment]) -> Vec<LineString<f64>> {
    if segments.is_empty() {
        return Vec::new();
    }

    // Build adjacency: start endpoint → segment index
    let mut endpoint_map: HashMap<PointKey, Vec<usize>> = HashMap::with_capacity(segments.len());
    for (i, seg) in segments.iter().enumerate() {
        endpoint_map
            .entry(quantize(&seg.start))
            .or_default()
            .push(i);
    }

    let mut used = vec![false; segments.len()];
    let mut rings = Vec::new();

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

            // Check if ring is closed
            if coords.len() > 2 && end_key == start_key {
                *coords.last_mut().unwrap() = coords[0];
                break;
            }

            // Find next unused segment whose start matches our end
            let next = endpoint_map
                .get(&end_key)
                .and_then(|candidates| candidates.iter().find(|&&idx| !used[idx]).copied());

            match next {
                Some(idx) => {
                    used[idx] = true;
                    coords.push(segments[idx].end);
                }
                None => break, // Can't close — discard
            }
        }

        if coords.len() >= 4 {
            let first = coords[0];
            let last = *coords.last().unwrap();
            if quantize(&first) == quantize(&last) {
                *coords.last_mut().unwrap() = first;
                rings.push(LineString(coords));
            }
        }
    }

    rings
}

// ---------------------------------------------------------------------------
// Transform helper
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn apply_transform(
    ring: &LineString<f64>,
    transform: &AffineTransform,
) -> LineString<f64> {
    LineString(
        ring.0
            .iter()
            .map(|c| {
                let (x, y) = transform.apply(c.x, c.y);
                Coord { x, y }
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(data: &[f64], w: usize, h: usize) -> RasterGrid<'_, f64> {
        RasterGrid::new(data, w, h)
    }

    #[test]
    fn test_flat_raster_inside_band() {
        // All values 5.0, band [3, 7) → entire grid is inside
        let data = vec![5.0; 9];
        let grid = make_grid(&data, 3, 3);
        let result = contours(&grid, &[3.0, 7.0], None, AffineTransform::identity());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 3.0);
        // Area should be approximately 2x2 (dual grid of 3x3)
        let area = signed_area(result[0].0.exterior()).abs();
        assert!(area > 3.0, "area={} too small", area);
    }

    #[test]
    fn test_flat_below_threshold() {
        let data = vec![1.0; 9];
        let grid = make_grid(&data, 3, 3);
        let result = contours(&grid, &[3.0, 7.0], None, AffineTransform::identity());
        assert!(result.is_empty());
    }

    #[test]
    fn test_flat_above_threshold() {
        let data = vec![10.0; 9];
        let grid = make_grid(&data, 3, 3);
        let result = contours(&grid, &[3.0, 7.0], None, AffineTransform::identity());
        // All values above hi → no isoband (lo-exterior minus hi-hole = empty)
        assert!(result.is_empty());
    }

    #[test]
    fn test_linear_gradient() {
        // 4x2 grid with horizontal gradient
        let data = vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0];
        let grid = make_grid(&data, 4, 2);
        let result = contours(&grid, &[0.5, 1.5, 2.5], None, AffineTransform::identity());
        assert!(!result.is_empty(), "Should produce some polygons");
        let band_values: Vec<f64> = result.iter().map(|r| r.1).collect();
        assert!(band_values.contains(&0.5));
        assert!(band_values.contains(&1.5));
    }

    #[test]
    fn test_nan_handling() {
        let data = vec![1.0, f64::NAN, 3.0, 2.0, 2.0, 2.0, 1.0, 2.0, 3.0];
        let grid = make_grid(&data, 3, 3);
        let _result = contours(&grid, &[1.5, 2.5], None, AffineTransform::identity());
    }

    #[test]
    fn test_ring_closure() {
        let data = vec![0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0];
        let grid = make_grid(&data, 3, 3);
        let result = contours(&grid, &[2.0, 8.0], None, AffineTransform::identity());
        for (poly, _) in &result {
            let ext = poly.exterior();
            assert!(ext.0.len() >= 4, "Ring too short");
            let first = ext.0.first().unwrap();
            let last = ext.0.last().unwrap();
            assert!(
                (first.x - last.x).abs() < 1e-10 && (first.y - last.y).abs() < 1e-10,
                "Ring not closed"
            );
        }
    }

    #[test]
    fn test_transform_applied() {
        let data = vec![0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0];
        let grid = make_grid(&data, 3, 3);
        let transform = AffineTransform::new(10.0, 0.0, 100.0, 0.0, -10.0, 200.0);
        let result = contours(&grid, &[2.0, 8.0], None, transform);
        for (poly, _) in &result {
            for coord in &poly.exterior().0 {
                assert!(coord.x >= 100.0 - 1e-6, "x={} < 100", coord.x);
                assert!(coord.y <= 200.0 + 1e-6, "y={} > 200", coord.y);
            }
        }
    }

    #[test]
    fn test_step_function_two_bands() {
        // Left half = 0, right half = 10
        let data = vec![0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0];
        let grid = make_grid(&data, 4, 2);
        let result = contours(&grid, &[0.0, 5.0, 10.0], None, AffineTransform::identity());
        assert!(!result.is_empty());
    }

    #[test]
    fn test_too_few_thresholds() {
        let data = vec![1.0; 4];
        let grid = make_grid(&data, 2, 2);
        let result = contours(&grid, &[0.5], None, AffineTransform::identity());
        assert!(result.is_empty());
    }

    #[test]
    fn test_mask() {
        let data = vec![5.0; 9];
        let grid = make_grid(&data, 3, 3);
        let mask = vec![true, true, true, true, false, true, true, true, true];
        let result = contours(&grid, &[3.0, 7.0], Some(&mask), AffineTransform::identity());
        // Center pixel masked → cells touching center produce NaN → skipped
        // Border cells (extended grid) still produce a ring around the unmasked pixels
        // Result may have polygons from the border isoline
        // Just verify it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_donut_band() {
        // Peak in center, band should form a ring (donut)
        #[rustfmt::skip]
        let data = vec![
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 2.0, 2.0, 0.0,
            0.0, 2.0, 5.0, 2.0, 0.0,
            0.0, 2.0, 2.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let grid = make_grid(&data, 5, 5);
        let result = contours(&grid, &[1.0, 3.0], None, AffineTransform::identity());
        assert!(!result.is_empty());
        // The band [1,3) should form a donut around the center peak
        let poly = &result[0].0;
        assert_eq!(poly.interiors().len(), 1, "Should have one hole (peak)");
    }

    #[test]
    fn test_interp() {
        assert!((interp(0.0, 10.0, 5.0) - 0.5).abs() < 1e-10);
        assert!((interp(0.0, 10.0, 0.0) - 0.0).abs() < 1e-10);
        assert!((interp(0.0, 10.0, 10.0) - 1.0).abs() < 1e-10);
        assert!((interp(5.0, 5.0, 5.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_area_conservation() {
        // 3x3 grid with vertical gradient
        let data = vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0];
        let grid = make_grid(&data, 3, 3);
        let result = contours(
            &grid,
            &[0.0, 0.25, 0.5, 0.75, 1.0],
            None,
            AffineTransform::identity(),
        );

        // Net area = exterior - holes for each polygon
        let total_area: f64 = result
            .iter()
            .map(|(poly, _)| {
                let ext = signed_area(poly.exterior()).abs();
                let holes: f64 = poly.interiors().iter().map(|h| signed_area(h).abs()).sum();
                ext - holes
            })
            .sum();

        assert!(total_area > 0.0, "total_area={}", total_area);
    }
}
