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
