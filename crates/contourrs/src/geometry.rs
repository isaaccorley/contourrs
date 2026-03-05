//! Shared geometry helpers: signed area, point-in-ring, polygon assembly.

use geo_types::{Coord, LineString};

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

    // Bounding box pre-check
    if n > 0 {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for c in coords {
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
        if point.x < min_x || point.x > max_x || point.y < min_y || point.y > max_y {
            return false;
        }
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
}
