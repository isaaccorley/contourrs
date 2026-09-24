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
//! The isoband is the polygon difference {val >= lo} ∖ {val >= hi}.
//! Each superlevel region includes its interior rings before subtraction.
//! Standard 16-case marching squares uses center-value saddle disambiguation.

use rustc_hash::FxHashMap;
use std::borrow::Cow;

use geo::{BooleanOps, MapCoords};
use geo_types::{Coord, LineString, Polygon};
use rayon::prelude::*;

use crate::contour_geometry::assemble_rings;
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
/// Samples lie at integer `(column, row)` coordinates, so contour bounds
/// fall within `[0, width - 1]` and `[0, height - 1]` before the transform.
/// Masked and non-finite samples lie outside every band; boundaries toward
/// missing samples close at their finite neighbors. Thresholds are sorted,
/// deduplicated, and filtered to finite values.
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

    // Sort + dedup thresholds, filtering non-finite values (NaN, ±Inf)
    let mut thresholds: Vec<f64> = thresholds.to_vec();
    thresholds.retain(|t| t.is_finite());
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
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

    // Cache the upper superlevel region for the next band's lower bound.
    // Subtract actual polygons: merely appending upper rings as holes loses
    // basins and produces invalid holes where boundaries meet the raster edge.
    let mut lower = assemble_rings(chain_segments(&march_isoline(
        &f64_data,
        w,
        h,
        thresholds[0],
        mask,
    )));
    for pair in thresholds.windows(2) {
        let upper = assemble_rings(chain_segments(&march_isoline(
            &f64_data, w, h, pair[1], mask,
        )));
        let band = lower.difference(&upper);
        for polygon in band.0 {
            let polygon = if transform.is_identity() {
                polygon
            } else {
                polygon.map_coords(|c| {
                    let (x, y) = transform.apply(c.x, c.y);
                    Coord { x, y }
                })
            };
            result.push((normalize_polygon(polygon), pair[0]));
        }
        lower = upper;
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

/// Get a sample; missing or non-finite samples lie outside every superlevel.
#[inline]
fn grid_val(data: &[f64], w: usize, h: usize, col: i32, row: i32, mask: Option<&[bool]>) -> f64 {
    if col < 0 || row < 0 || col >= w as i32 || row >= h as i32 {
        return f64::NEG_INFINITY;
    }
    let idx = row as usize * w + col as usize;
    if let Some(m) = mask {
        if m.get(idx).copied() != Some(true) {
            return f64::NEG_INFINITY;
        }
    }
    let value = data[idx];
    if value.is_finite() {
        value
    } else {
        f64::NEG_INFINITY
    }
}

/// Process a single row of the marching-squares grid, returning segments.
#[inline]
fn march_row(
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
        let (tl, bl) = if col == -1 {
            (
                grid_val(data, w, h, col, row, mask),
                grid_val(data, w, h, col, row + 1, mask),
            )
        } else {
            (prev_tr, prev_br)
        };
        let tr = grid_val(data, w, h, col + 1, row, mask);
        let br = grid_val(data, w, h, col + 1, row + 1, mask);
        prev_tr = tr;
        prev_br = br;

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
        // Use the same interpolation direction as top/right so that shared
        // edges between adjacent cells produce bit-identical coordinates.
        let bottom = || Coord {
            x: cx + interp(bl, br, threshold),
            y: cy + 1.0,
        };
        let left = || Coord {
            x: cx,
            y: cy + interp(tl, bl, threshold),
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
                let center = tl * 0.25 + tr * 0.25 + br * 0.25 + bl * 0.25;
                if center >= threshold {
                    // 1s connected (tr-bl). Two 0-islands: tl and br.
                    // Ring around tl(0): enters from cell above through top,
                    // exits to cell-left through left.
                    segments.push(EdgeSegment {
                        start: top(),
                        end: left(),
                    });
                    // Ring around br(0): enters from cell below through bottom,
                    // exits to cell-right through right.
                    segments.push(EdgeSegment {
                        start: bottom(),
                        end: right(),
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
                let center = tl * 0.25 + tr * 0.25 + br * 0.25 + bl * 0.25;
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
    if w * h >= PARALLEL_THRESHOLD {
        (-1..h as i32)
            .into_par_iter()
            .flat_map_iter(|row| march_row(data, w, h, row, threshold, mask))
            .collect()
    } else {
        (-1..h as i32)
            .flat_map(|row| march_row(data, w, h, row, threshold, mask))
            .collect()
    }
}

/// Linear interpolation fraction.
#[inline]
fn interp(v0: f64, v1: f64, threshold: f64) -> f64 {
    if v0.is_infinite() || v1.is_infinite() {
        if v0.is_infinite() && v1.is_infinite() {
            return 0.5;
        }
        // The limiting crossing lies at the finite sample, so padded cells
        // close at the raster edge instead of extending a pixel beyond it.
        return if v0.is_infinite() { 1.0 } else { 0.0 };
    }
    let denom = v1 - v0;
    if denom == 0.0 {
        0.5
    } else if denom.is_infinite() {
        ((threshold * 0.5 - v0 * 0.5) / (v1 * 0.5 - v0 * 0.5)).clamp(0.0, 1.0)
    } else {
        ((threshold - v0) / denom).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Segment chaining: connect segments into closed rings
// ---------------------------------------------------------------------------

type PointKey = (i64, i64);

#[inline]
fn quantize(c: &Coord<f64>) -> PointKey {
    ((c.x * 1e10).round() as i64, (c.y * 1e10).round() as i64)
}

fn chain_segments(segments: &[EdgeSegment]) -> Vec<LineString<f64>> {
    if segments.is_empty() {
        return Vec::new();
    }

    // Build adjacency: start endpoint → segment index
    let mut endpoint_map: FxHashMap<PointKey, Vec<usize>> =
        FxHashMap::with_capacity_and_hasher(segments.len(), Default::default());
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

        let mut coords = Vec::with_capacity(8);
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "contour_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "contour_regressions.rs"]
mod regressions;
