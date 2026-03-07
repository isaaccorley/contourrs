use crate::geometry::{point_in_ring_prechecked_bbox, signed_area, BBox};
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
        labels[row as usize * w + col as usize]
    }
}

/// Trace polygon boundaries from a label grid using direct contour tracing.
pub fn trace_polygons(
    label_result: &LabelResult,
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

    build_polygons(rings, &label_result.values, det)
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
    let mut coords = Vec::with_capacity(64);
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
    mut rings: Vec<(u32, LineString<f64>)>,
    values: &[f64],
    det: f64,
) -> Vec<(Polygon<f64>, f64)> {
    if rings.is_empty() {
        return Vec::new();
    }

    // Sort rings by label in-place (avoids separate index vec for small ring counts)
    rings.sort_unstable_by_key(|r| r.0);

    let mut result = Vec::new();
    let mut i = 0;

    while i < rings.len() {
        let label = rings[i].0;
        if label == u32::MAX {
            i += 1;
            continue;
        }

        let value = values[label as usize];

        // Find range of rings with this label
        let start = i;
        while i < rings.len() && rings[i].0 == label {
            i += 1;
        }

        // Classify into exteriors/holes by index within the range
        let mut exterior_idxs: Vec<usize> = Vec::new();
        let mut hole_idxs: Vec<usize> = Vec::new();

        for (idx, ring) in rings[start..i].iter().enumerate() {
            let area = signed_area(&ring.1);
            if area.abs() >= f64::EPSILON {
                let is_exterior = if det >= 0.0 { area > 0.0 } else { area < 0.0 };
                if is_exterior {
                    exterior_idxs.push(start + idx);
                } else {
                    hole_idxs.push(start + idx);
                }
            }
        }

        if exterior_idxs.len() == 1 && hole_idxs.is_empty() {
            // Single exterior, no holes — take ownership directly
            let ext = std::mem::replace(&mut rings[exterior_idxs[0]].1, LineString(Vec::new()));
            result.push((Polygon::new(ext, vec![]), value));
        } else if exterior_idxs.len() == 1 {
            // Single exterior with holes — take all
            let ext = std::mem::replace(&mut rings[exterior_idxs[0]].1, LineString(Vec::new()));
            let holes: Vec<LineString<f64>> = hole_idxs
                .iter()
                .map(|&h| std::mem::replace(&mut rings[h].1, LineString(Vec::new())))
                .collect();
            result.push((Polygon::new(ext, holes), value));
        } else {
            // Multiple exteriors — bbox pre-filter + point_in_ring for hole assignment
            let ext_bboxes: Vec<BBox> = exterior_idxs
                .iter()
                .map(|&idx| BBox::from_ring(&rings[idx].1))
                .collect();

            // Assign holes to exteriors via point-in-ring with bbox pre-filter.
            // Holes are moved out of `rings`, and each hole is assigned to at most one exterior.
            let mut ext_holes: Vec<Vec<LineString<f64>>> = vec![Vec::new(); exterior_idxs.len()];
            for &hole_idx in &hole_idxs {
                let hp = &rings[hole_idx].1 .0[0];
                for (j, &ext_idx) in exterior_idxs.iter().enumerate() {
                    if ext_bboxes[j].contains_point(hp)
                        && point_in_ring_prechecked_bbox(hp, &rings[ext_idx].1)
                    {
                        ext_holes[j].push(std::mem::replace(
                            &mut rings[hole_idx].1,
                            LineString(Vec::new()),
                        ));
                        break; // each hole belongs to exactly one exterior
                    }
                }
            }
            for (j, &ext_idx) in exterior_idxs.iter().enumerate() {
                let ext = std::mem::replace(&mut rings[ext_idx].1, LineString(Vec::new()));
                let my_holes = std::mem::take(&mut ext_holes[j]);
                result.push((Polygon::new(ext, my_holes), value));
            }
        }
    }

    result
}
