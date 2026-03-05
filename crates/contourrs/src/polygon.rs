use crate::geometry::signed_area;
use geo_types::{LineString, Polygon};

/// Ensure polygon follows GeoJSON convention:
/// exterior ring is CCW, interior rings (holes) are CW.
#[inline]
pub fn normalize_polygon(polygon: Polygon<f64>) -> Polygon<f64> {
    let (mut exterior, holes) = polygon.into_inner();

    // Ensure exterior is CCW (positive signed area)
    if signed_area(&exterior) < 0.0 {
        exterior.0.reverse();
    }

    // Ensure holes are CW (negative signed area)
    let holes: Vec<LineString<f64>> = holes
        .into_iter()
        .map(|mut hole| {
            if signed_area(&hole) > 0.0 {
                hole.0.reverse();
            }
            hole
        })
        .collect();

    Polygon::new(exterior, holes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::Coord;

    fn ccw_square() -> LineString<f64> {
        LineString(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
            Coord { x: 0.0, y: 1.0 },
            Coord { x: 0.0, y: 0.0 },
        ])
    }

    fn cw_square() -> LineString<f64> {
        LineString(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 0.0, y: 1.0 },
            Coord { x: 1.0, y: 1.0 },
            Coord { x: 1.0, y: 0.0 },
            Coord { x: 0.0, y: 0.0 },
        ])
    }

    #[test]
    fn test_normalize_cw_exterior() {
        // CW exterior → should be reversed to CCW
        let poly = Polygon::new(cw_square(), vec![]);
        let normalized = normalize_polygon(poly);
        assert!(signed_area(normalized.exterior()) > 0.0);
    }

    #[test]
    fn test_normalize_ccw_hole() {
        // CCW hole → should be reversed to CW
        let hole = LineString(vec![
            Coord { x: 0.2, y: 0.2 },
            Coord { x: 0.8, y: 0.2 },
            Coord { x: 0.8, y: 0.8 },
            Coord { x: 0.2, y: 0.8 },
            Coord { x: 0.2, y: 0.2 },
        ]);
        assert!(signed_area(&hole) > 0.0, "hole should start as CCW");
        let poly = Polygon::new(ccw_square(), vec![hole]);
        let normalized = normalize_polygon(poly);
        assert!(signed_area(normalized.exterior()) > 0.0);
        assert!(signed_area(&normalized.interiors()[0]) < 0.0);
    }

    #[test]
    fn test_normalize_already_correct() {
        // Already correct orientation → no change
        let poly = Polygon::new(ccw_square(), vec![cw_square()]);
        let normalized = normalize_polygon(poly);
        assert!(signed_area(normalized.exterior()) > 0.0);
        assert!(signed_area(&normalized.interiors()[0]) < 0.0);
    }
}
