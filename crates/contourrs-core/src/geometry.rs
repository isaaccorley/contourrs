//! Shared geometry helpers: signed area, point-in-ring, polygon assembly.

use geo_types::{Coord, LineString, Polygon};

/// Signed area of a ring (positive = CCW, negative = CW).
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
pub fn point_in_ring(point: &Coord<f64>, ring: &LineString<f64>) -> bool {
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

/// Classify rings as exterior/hole and assemble into polygons.
///
/// - `det`: determinant of affine transform (a*e - b*d)
/// - Positive det → positive area = exterior; negative det → reversed
pub fn build_polygons_from_rings(
    exteriors: &[LineString<f64>],
    holes: &[LineString<f64>],
    value: f64,
) -> Vec<(Polygon<f64>, f64)> {
    if exteriors.is_empty() {
        return Vec::new();
    }

    if exteriors.len() == 1 {
        let polygon = Polygon::new(exteriors[0].clone(), holes.to_vec());
        return vec![(polygon, value)];
    }

    // Multiple exteriors: assign each hole to the containing exterior
    let mut result = Vec::with_capacity(exteriors.len());
    for ext in exteriors {
        let mut my_holes = Vec::new();
        for hole in holes {
            if point_in_ring(&hole.0[0], ext) {
                my_holes.push(hole.clone());
            }
        }
        let polygon = Polygon::new(ext.clone(), my_holes);
        result.push((polygon, value));
    }
    result
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
