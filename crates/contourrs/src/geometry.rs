//! Shared geometry helpers: signed area, point-in-ring, polygon assembly.

use geo_types::{Coord, LineString};

#[derive(Clone, Copy, Debug)]
pub(crate) struct BBox {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
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
    pub(crate) fn contains_point(&self, point: &Coord<f64>) -> bool {
        point.x >= self.min_x
            && point.x <= self.max_x
            && point.y >= self.min_y
            && point.y <= self.max_y
    }
}

/// Signed area of a ring (positive = CCW, negative = CW).
#[inline]
pub fn signed_area(ring: &LineString<f64>) -> f64 {
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

/// Ray-casting point-in-polygon test.
#[inline]
pub fn point_in_ring(point: &Coord<f64>, ring: &LineString<f64>) -> bool {
    let coords = &ring.0;
    let n = coords.len();
    if n < 3 {
        return false;
    }

    if !BBox::from_ring(ring).contains_point(point) {
        return false;
    }

    point_in_ring_prechecked_bbox(point, ring)
}

/// Ray-casting point-in-polygon test for callers that already checked the bbox.
#[inline]
pub(crate) fn point_in_ring_prechecked_bbox(point: &Coord<f64>, ring: &LineString<f64>) -> bool {
    let coords = &ring.0;
    let n = coords.len();
    if n < 3 {
        return false;
    }

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
        // Outside bbox (right)
        assert!(!point_in_ring(&Coord { x: 3.0, y: 1.0 }, &ring));
        // Outside bbox (above)
        assert!(!point_in_ring(&Coord { x: 1.0, y: -1.0 }, &ring));
    }

    #[test]
    fn test_signed_area_degenerate() {
        // Empty ring
        let empty = LineString(vec![]);
        assert_eq!(signed_area(&empty), 0.0);
        // Single point
        let single = LineString(vec![Coord { x: 1.0, y: 1.0 }]);
        assert_eq!(signed_area(&single), 0.0);
        // Two points (not a ring)
        let two = LineString(vec![Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: 1.0 }]);
        assert_eq!(signed_area(&two), 0.0);
    }

    #[test]
    fn test_point_outside_bbox_y() {
        // Point outside bbox vertically (below)
        let ring = LineString(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 2.0, y: 0.0 },
            Coord { x: 2.0, y: 2.0 },
            Coord { x: 0.0, y: 2.0 },
            Coord { x: 0.0, y: 0.0 },
        ]);
        assert!(!point_in_ring(&Coord { x: 1.0, y: 3.0 }, &ring));
    }

    #[test]
    fn test_point_in_empty_ring_is_false() {
        let ring = LineString(vec![]);
        assert!(!point_in_ring(&Coord { x: 0.0, y: 0.0 }, &ring));
    }

    #[test]
    fn test_point_in_ring_prechecked_bbox() {
        let ring = LineString(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 2.0, y: 0.0 },
            Coord { x: 2.0, y: 2.0 },
            Coord { x: 0.0, y: 2.0 },
            Coord { x: 0.0, y: 0.0 },
        ]);
        let bbox = BBox::from_ring(&ring);
        let point = Coord { x: 1.0, y: 1.0 };
        assert!(bbox.contains_point(&point));
        assert!(point_in_ring_prechecked_bbox(&point, &ring));
    }
}
