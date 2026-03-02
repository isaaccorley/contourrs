use geo_types::{LineString, Polygon};

/// Ensure polygon follows GeoJSON convention:
/// exterior ring is CCW, interior rings (holes) are CW.
pub fn normalize_polygon(polygon: Polygon<f64>) -> Polygon<f64> {
    let (mut exterior, holes) = polygon.into_inner();

    // Ensure exterior is CCW (positive signed area)
    if signed_area_ring(&exterior) < 0.0 {
        exterior.0.reverse();
    }

    // Ensure holes are CW (negative signed area)
    let holes: Vec<LineString<f64>> = holes
        .into_iter()
        .map(|mut hole| {
            if signed_area_ring(&hole) > 0.0 {
                hole.0.reverse();
            }
            hole
        })
        .collect();

    Polygon::new(exterior, holes)
}

fn signed_area_ring(ring: &LineString<f64>) -> f64 {
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
    area / 2.0
}
