//! Trace labeled raster boundaries and assemble polygon rings.

use crate::geometry::{point_in_ring_prechecked_bbox, signed_area, BBox};
use crate::label::LabelResult;
use crate::transform::AffineTransform;
use geo_types::{Coord, LineString, Polygon};

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Direction {
    Right,
    Down,
    Left,
    Up,
}

impl Direction {
    fn right(self) -> Self {
        match self {
            Self::Right => Self::Down,
            Self::Down => Self::Left,
            Self::Left => Self::Up,
            Self::Up => Self::Right,
        }
    }

    fn step(self, col: i32, row: i32) -> (i32, i32) {
        match self {
            Self::Right => (col + 1, row),
            Self::Down => (col, row + 1),
            Self::Left => (col - 1, row),
            Self::Up => (col, row - 1),
        }
    }

    fn is_boundary(self, label: u32, tl: u32, tr: u32, bl: u32, br: u32) -> bool {
        match self {
            Self::Right => br == label && tr != label,
            Self::Down => bl == label && br != label,
            Self::Left => tl == label && bl != label,
            Self::Up => tr == label && tl != label,
        }
    }

    fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// Trace polygon boundaries from a label grid using direct contour tracing.
///
/// The label grid must come from [`crate::label::label_regions`], or uphold
/// its dimensions, valid region indices, and `u32::MAX` mask sentinel.
///
/// # Panics
/// Panics if dimensions or region indices are inconsistent with the buffers.
pub fn trace_polygons(
    label_result: &LabelResult,
    transform: &AffineTransform,
) -> Vec<(Polygon<f64>, f64)> {
    let stride = label_result.width + 1;
    let mut tracer = BoundaryTracer {
        grid: label_result,
        transform,
        stride,
        // Four visited bits per vertex, one for each direction.
        visited: vec![0; stride * (label_result.height + 1)],
    };
    let rings = tracer.rings();
    let determinant = transform.a * transform.e - transform.b * transform.d;
    build_polygons(rings, &label_result.values, determinant)
}

struct BoundaryTracer<'grid> {
    grid: &'grid LabelResult,
    transform: &'grid AffineTransform,
    visited: Vec<u8>,
    stride: usize,
}

impl BoundaryTracer<'_> {
    /// Out-of-bounds samples share the masked-pixel sentinel.
    #[inline]
    fn label(&self, col: i32, row: i32) -> u32 {
        if col < 0 || row < 0 || col >= self.grid.width as i32 || row >= self.grid.height as i32 {
            u32::MAX
        } else {
            self.grid.labels[row as usize * self.grid.width + col as usize]
        }
    }

    fn rings(&mut self) -> Vec<(u32, LineString<f64>)> {
        let mut rings = Vec::new();
        // Preserve scan order: horizontal boundaries, then vertical boundaries.
        for row in 0..=self.grid.height as i32 {
            for col in 0..self.grid.width as i32 {
                let above = self.label(col, row - 1);
                let below = self.label(col, row);
                if above != below {
                    if let Some(ring) = self.boundary(col, row, Direction::Right, below) {
                        rings.push((below, ring));
                    }
                    if let Some(ring) = self.boundary(col + 1, row, Direction::Left, above) {
                        rings.push((above, ring));
                    }
                }
            }
        }
        for row in 0..self.grid.height as i32 {
            for col in 0..=self.grid.width as i32 {
                let left = self.label(col - 1, row);
                let right = self.label(col, row);
                if left != right {
                    if let Some(ring) = self.boundary(col, row, Direction::Down, left) {
                        rings.push((left, ring));
                    }
                    if let Some(ring) = self.boundary(col, row + 1, Direction::Up, right) {
                        rings.push((right, ring));
                    }
                }
            }
        }
        rings
    }

    fn boundary(
        &mut self,
        col: i32,
        row: i32,
        direction: Direction,
        label: u32,
    ) -> Option<LineString<f64>> {
        if label == u32::MAX {
            return None;
        }
        let vertex = row as usize * self.stride + col as usize;
        if self.visited[vertex] & direction.bit() != 0 {
            return None;
        }
        let ring = self.trace_ring((col, row), direction, label);
        (ring.0.len() >= 4).then_some(ring)
    }

    #[inline(never)]
    fn trace_ring(
        &mut self,
        start: (i32, i32),
        start_direction: Direction,
        label: u32,
    ) -> LineString<f64> {
        let mut coords = Vec::with_capacity(64);
        let (mut col, mut row) = start;
        let mut direction = start_direction;
        loop {
            self.visited[row as usize * self.stride + col as usize] |= direction.bit();
            let (x, y) = self.transform.apply(f64::from(col), f64::from(row));
            coords.push(Coord { x, y });
            (col, row) = direction.step(col, row);

            let tl = self.label(col - 1, row - 1);
            let tr = self.label(col, row - 1);
            let bl = self.label(col - 1, row);
            let br = self.label(col, row);
            let right = direction.right();
            let reverse = right.right();
            let left = reverse.right();
            // Right turn, straight, left turn, then U-turn.
            direction = if right.is_boundary(label, tl, tr, bl, br) {
                right
            } else if direction.is_boundary(label, tl, tr, bl, br) {
                direction
            } else if left.is_boundary(label, tl, tr, bl, br) {
                left
            } else {
                reverse
            };
            if (col, row) == start && direction == start_direction {
                let (x, y) = self.transform.apply(f64::from(col), f64::from(row));
                coords.push(Coord { x, y });
                break;
            }
        }
        LineString(coords)
    }
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
