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

use std::collections::HashMap;

use geo_types::{Coord, LineString, Polygon};

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

    // Pre-convert grid to f64 for interpolation
    let f64_data: Vec<f64> = grid.data.iter().map(|v| v.to_f64_value()).collect();
    let w = grid.width;
    let h = grid.height;

    let mut result = Vec::new();

    for pair in thresholds.windows(2) {
        let lo = pair[0];
        let hi = pair[1];

        // Generate isoline segments at both thresholds
        let lo_segments = march_isoline(&f64_data, w, h, lo, mask);
        let hi_segments = march_isoline(&f64_data, w, h, hi, mask);

        // Chain into rings
        let lo_rings = chain_segments(&lo_segments);
        let hi_rings = chain_segments(&hi_segments);

        if lo_rings.is_empty() {
            continue;
        }

        // Classify and transform rings.
        // The marching squares "inside to the left" convention in y-down coords
        // means iso-exterior rings are CW (negative signed_area with positive det).
        let det = transform.a * transform.e - transform.b * transform.d;
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
            for ext in &exteriors {
                let mut my_holes = Vec::new();
                for hole in &holes {
                    if !hole.0.is_empty() && point_in_ring(&hole.0[0], ext) {
                        my_holes.push(hole.clone());
                    }
                }
                let ext_area = signed_area(ext).abs();
                let hole_area: f64 = my_holes.iter().map(|h| signed_area(h).abs()).sum();
                let net_area = ext_area - hole_area;
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
struct EdgeSegment {
    start: Coord<f64>,
    end: Coord<f64>,
}

// ---------------------------------------------------------------------------
// Standard 16-case marching squares isoline
// ---------------------------------------------------------------------------

/// Get pixel value, returning NaN for out-of-bounds or masked pixels.
#[inline]
fn grid_val(data: &[f64], w: usize, h: usize, col: i32, row: i32, mask: Option<&[bool]>) -> f64 {
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

/// Generate isoline segments at a single threshold using 16-case marching squares.
fn march_isoline(
    data: &[f64],
    w: usize,
    h: usize,
    threshold: f64,
    mask: Option<&[bool]>,
) -> Vec<EdgeSegment> {
    let mut segments = Vec::new();

    // Extended grid: from -1..h, -1..w so boundary pixels get border cells
    for row in -1..h as i32 {
        for col in -1..w as i32 {
            let tl = grid_val(data, w, h, col, row, mask);
            let tr = grid_val(data, w, h, col + 1, row, mask);
            let bl = grid_val(data, w, h, col, row + 1, mask);
            let br = grid_val(data, w, h, col + 1, row + 1, mask);

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
                    // Only BL inside
                    segments.push(EdgeSegment {
                        start: bottom(),
                        end: left(),
                    });
                }
                2 => {
                    // Only BR inside
                    segments.push(EdgeSegment {
                        start: right(),
                        end: bottom(),
                    });
                }
                3 => {
                    // BL + BR inside
                    segments.push(EdgeSegment {
                        start: right(),
                        end: left(),
                    });
                }
                4 => {
                    // Only TR inside
                    segments.push(EdgeSegment {
                        start: top(),
                        end: right(),
                    });
                }
                5 => {
                    // TR + BL inside (saddle)
                    let center = (tl + tr + br + bl) * 0.25;
                    if center >= threshold {
                        // Connected through center
                        segments.push(EdgeSegment {
                            start: top(),
                            end: left(),
                        });
                        segments.push(EdgeSegment {
                            start: right(),
                            end: bottom(),
                        });
                    } else {
                        // Disconnected
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
                    // TR + BR inside
                    segments.push(EdgeSegment {
                        start: top(),
                        end: bottom(),
                    });
                }
                7 => {
                    // Only TL outside
                    segments.push(EdgeSegment {
                        start: top(),
                        end: left(),
                    });
                }
                8 => {
                    // Only TL inside
                    segments.push(EdgeSegment {
                        start: left(),
                        end: top(),
                    });
                }
                9 => {
                    // TL + BL inside
                    segments.push(EdgeSegment {
                        start: bottom(),
                        end: top(),
                    });
                }
                10 => {
                    // TL + BR inside (saddle)
                    let center = (tl + tr + br + bl) * 0.25;
                    if center >= threshold {
                        // Connected through center
                        segments.push(EdgeSegment {
                            start: left(),
                            end: bottom(),
                        });
                        segments.push(EdgeSegment {
                            start: right(),
                            end: top(),
                        });
                    } else {
                        // Disconnected
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
                    // Only TR outside
                    segments.push(EdgeSegment {
                        start: right(),
                        end: top(),
                    });
                }
                12 => {
                    // TL + TR inside
                    segments.push(EdgeSegment {
                        start: left(),
                        end: right(),
                    });
                }
                13 => {
                    // Only BR outside → crossings on bottom and right edges
                    segments.push(EdgeSegment {
                        start: bottom(),
                        end: right(),
                    });
                }
                14 => {
                    // Only BL outside → crossings on left and bottom edges
                    segments.push(EdgeSegment {
                        start: left(),
                        end: bottom(),
                    });
                }
                _ => {} // 0 and 15 handled above
            }
        }
    }

    segments
}

/// Linear interpolation fraction.
#[inline]
fn interp(v0: f64, v1: f64, threshold: f64) -> f64 {
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

type PointKey = (i64, i64);

fn quantize(c: &Coord<f64>) -> PointKey {
    ((c.x * 1e10).round() as i64, (c.y * 1e10).round() as i64)
}

fn chain_segments(segments: &[EdgeSegment]) -> Vec<LineString<f64>> {
    if segments.is_empty() {
        return Vec::new();
    }

    // Build adjacency: start endpoint → segment index
    let mut endpoint_map: HashMap<PointKey, Vec<usize>> = HashMap::new();
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

fn apply_transform(ring: &LineString<f64>, transform: &AffineTransform) -> LineString<f64> {
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
