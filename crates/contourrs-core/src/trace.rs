use crate::label::LabelResult;
use crate::transform::AffineTransform;
use geo_types::{Coord, LineString, Polygon};

const DIR_RIGHT: u8 = 0;
const DIR_DOWN: u8 = 1;
const DIR_LEFT: u8 = 2;
const DIR_UP: u8 = 3;

#[inline(always)]
fn step(c: i32, r: i32, dir: u8) -> (i32, i32) {
    match dir {
        DIR_RIGHT => (c + 1, r),
        DIR_DOWN => (c, r + 1),
        DIR_LEFT => (c - 1, r),
        _ => (c, r - 1), // DIR_UP
    }
}

#[inline(always)]
fn is_valid_edge(dir: u8, label: u32, tl: u32, tr: u32, bl: u32, br: u32) -> bool {
    match dir {
        DIR_RIGHT => br == label && tr != label,
        DIR_DOWN => bl == label && br != label,
        DIR_LEFT => tl == label && bl != label,
        _ => tr == label && tl != label, // DIR_UP
    }
}

/// Get label at pixel (col, row), returning u32::MAX for out-of-bounds/masked.
#[inline(always)]
fn get_label(labels: &[u32], w: usize, h: usize, col: i32, row: i32) -> u32 {
    if col < 0 || row < 0 || col >= w as i32 || row >= h as i32 {
        u32::MAX
    } else {
        unsafe { *labels.get_unchecked(row as usize * w + col as usize) }
    }
}

/// Trace polygon boundaries from a label grid using direct contour tracing.
pub fn trace_polygons(
    label_result: &LabelResult,
    values: &[f64],
    transform: &AffineTransform,
) -> Vec<(Polygon<f64>, f64)> {
    let det = transform.a * transform.e - transform.b * transform.d;
    let w = label_result.width;
    let h = label_result.height;
    let labels = &label_result.labels;
    let stride = w + 1;

    // Visited bitfield: 4 bits per vertex (one per direction).
    let num_vertices = stride * (h + 1);
    let mut visited = vec![0u8; num_vertices];

    let mut rings: Vec<(u32, LineString<f64>)> = Vec::new();

    // Scan horizontal boundaries
    for row in 0..=h {
        for col in 0..w {
            let above = get_label(labels, w, h, col as i32, row as i32 - 1);
            let below = get_label(labels, w, h, col as i32, row as i32);
            if above == below {
                continue;
            }

            if below != u32::MAX {
                let vi = row * stride + col;
                if visited[vi] & (1 << DIR_RIGHT) == 0 {
                    let ring = trace_ring(
                        labels,
                        w,
                        h,
                        col as i32,
                        row as i32,
                        DIR_RIGHT,
                        below,
                        &mut visited,
                        stride,
                        transform,
                    );
                    if ring.0.len() >= 4 {
                        rings.push((below, ring));
                    }
                }
            }

            if above != u32::MAX {
                let vi = row * stride + col + 1;
                if visited[vi] & (1 << DIR_LEFT) == 0 {
                    let ring = trace_ring(
                        labels,
                        w,
                        h,
                        col as i32 + 1,
                        row as i32,
                        DIR_LEFT,
                        above,
                        &mut visited,
                        stride,
                        transform,
                    );
                    if ring.0.len() >= 4 {
                        rings.push((above, ring));
                    }
                }
            }
        }
    }

    // Scan vertical boundaries
    for row in 0..h {
        for col in 0..=w {
            let left = get_label(labels, w, h, col as i32 - 1, row as i32);
            let right = get_label(labels, w, h, col as i32, row as i32);
            if left == right {
                continue;
            }

            if left != u32::MAX {
                let vi = row * stride + col;
                if visited[vi] & (1 << DIR_DOWN) == 0 {
                    let ring = trace_ring(
                        labels,
                        w,
                        h,
                        col as i32,
                        row as i32,
                        DIR_DOWN,
                        left,
                        &mut visited,
                        stride,
                        transform,
                    );
                    if ring.0.len() >= 4 {
                        rings.push((left, ring));
                    }
                }
            }

            if right != u32::MAX {
                let vi = (row + 1) * stride + col;
                if visited[vi] & (1 << DIR_UP) == 0 {
                    let ring = trace_ring(
                        labels,
                        w,
                        h,
                        col as i32,
                        row as i32 + 1,
                        DIR_UP,
                        right,
                        &mut visited,
                        stride,
                        transform,
                    );
                    if ring.0.len() >= 4 {
                        rings.push((right, ring));
                    }
                }
            }
        }
    }

    build_polygons(&rings, values, det)
}

/// Trace a single closed ring contour.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn trace_ring(
    labels: &[u32],
    w: usize,
    h: usize,
    start_c: i32,
    start_r: i32,
    start_dir: u8,
    label: u32,
    visited: &mut [u8],
    stride: usize,
    transform: &AffineTransform,
) -> LineString<f64> {
    let mut coords = Vec::new();
    let mut c = start_c;
    let mut r = start_r;
    let mut dir = start_dir;

    loop {
        visited[r as usize * stride + c as usize] |= 1 << dir;

        let (x, y) = transform.apply(c as f64, r as f64);
        coords.push(Coord { x, y });

        let (nc, nr) = step(c, r, dir);
        c = nc;
        r = nr;

        // Choose next direction: right turn > straight > left turn > U-turn
        let tl = get_label(labels, w, h, c - 1, r - 1);
        let tr = get_label(labels, w, h, c, r - 1);
        let bl = get_label(labels, w, h, c - 1, r);
        let br = get_label(labels, w, h, c, r);

        let rt = (dir + 1) & 3;
        let st = dir;
        let lt = (dir + 3) & 3;

        dir = if is_valid_edge(rt, label, tl, tr, bl, br) {
            rt
        } else if is_valid_edge(st, label, tl, tr, bl, br) {
            st
        } else if is_valid_edge(lt, label, tl, tr, bl, br) {
            lt
        } else {
            (dir + 2) & 3
        };

        if c == start_c && r == start_r && dir == start_dir {
            let (x, y) = transform.apply(c as f64, r as f64);
            coords.push(Coord { x, y });
            break;
        }
    }

    LineString(coords)
}

/// Group traced rings by label and assemble into polygons.
fn build_polygons(
    rings: &[(u32, LineString<f64>)],
    values: &[f64],
    det: f64,
) -> Vec<(Polygon<f64>, f64)> {
    // Group by label using sorted indices
    if rings.is_empty() {
        return Vec::new();
    }

    let mut indices: Vec<usize> = (0..rings.len()).collect();
    indices.sort_unstable_by_key(|&i| rings[i].0);

    let mut result = Vec::new();
    let mut i = 0;

    while i < indices.len() {
        let label = rings[indices[i]].0;
        if label == u32::MAX {
            i += 1;
            continue;
        }

        let value = values[label as usize];
        let mut exteriors: Vec<usize> = Vec::new();
        let mut holes: Vec<usize> = Vec::new();

        while i < indices.len() && rings[indices[i]].0 == label {
            let idx = indices[i];
            let area = signed_area(&rings[idx].1);
            if area.abs() >= f64::EPSILON {
                let is_exterior = if det >= 0.0 { area > 0.0 } else { area < 0.0 };
                if is_exterior {
                    exteriors.push(idx);
                } else {
                    holes.push(idx);
                }
            }
            i += 1;
        }

        if exteriors.len() == 1 {
            let polygon = Polygon::new(
                rings[exteriors[0]].1.clone(),
                holes.iter().map(|&h| rings[h].1.clone()).collect(),
            );
            result.push((polygon, value));
        } else {
            for &ext_idx in &exteriors {
                let exterior = &rings[ext_idx].1;
                let mut my_holes = Vec::new();
                for &hole_idx in &holes {
                    if point_in_ring(&rings[hole_idx].1 .0[0], exterior) {
                        my_holes.push(rings[hole_idx].1.clone());
                    }
                }
                let polygon = Polygon::new(exterior.clone(), my_holes);
                result.push((polygon, value));
            }
        }
    }

    result
}

fn signed_area(ring: &LineString<f64>) -> f64 {
    let coords = &ring.0;
    let n = coords.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n - 1 {
        area += coords[i].x * coords[i + 1].y;
        area -= coords[i + 1].x * coords[i].y;
    }
    area * 0.5
}

fn point_in_ring(point: &Coord<f64>, ring: &LineString<f64>) -> bool {
    let coords = &ring.0;
    let n = coords.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        if ((coords[i].y > point.y) != (coords[j].y > point.y))
            && (point.x
                < (coords[j].x - coords[i].x) * (point.y - coords[i].y)
                    / (coords[j].y - coords[i].y)
                    + coords[i].x)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signed_area_ccw() {
        let ring = LineString(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
            Coord { x: 0.0, y: 1.0 },
            Coord { x: 0.0, y: 0.0 },
        ]);
        assert!(signed_area(&ring) > 0.0);
    }

    #[test]
    fn test_signed_area_cw() {
        let ring = LineString(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 0.0, y: 1.0 },
            Coord { x: 1.0, y: 1.0 },
            Coord { x: 1.0, y: 0.0 },
            Coord { x: 0.0, y: 0.0 },
        ]);
        assert!(signed_area(&ring) < 0.0);
    }

    #[test]
    fn test_point_in_ring() {
        let ring = LineString(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 2.0, y: 0.0 },
            Coord { x: 2.0, y: 2.0 },
            Coord { x: 0.0, y: 2.0 },
            Coord { x: 0.0, y: 0.0 },
        ]);
        assert!(point_in_ring(&Coord { x: 1.0, y: 1.0 }, &ring));
        assert!(!point_in_ring(&Coord { x: 3.0, y: 1.0 }, &ring));
    }
}
