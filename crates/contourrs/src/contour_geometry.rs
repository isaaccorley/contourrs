//! Assemble marching-squares superlevel rings before isoband subtraction.

use geo_types::{LineString, MultiPolygon, Polygon};
use rstar::primitives::{GeomWithData, Rectangle};
use rstar::{RTree, AABB};

use crate::geometry::{point_in_ring_prechecked_bbox, signed_area, BBox};
use crate::polygon::normalize_polygon;

pub(crate) fn assemble_rings(rings: Vec<LineString<f64>>) -> MultiPolygon<f64> {
    let mut exteriors = Vec::new();
    let mut holes = Vec::new();
    for ring in rings {
        let area = signed_area(&ring);
        // Marching-squares exteriors have negative area in raster coordinates.
        if area < 0.0 {
            exteriors.push((ring, -area));
        } else if area > 0.0 {
            holes.push(ring);
        }
    }
    let bboxes: Vec<_> = exteriors
        .iter()
        .map(|(ring, _)| BBox::from_ring(ring))
        .collect();
    // Bulk indexing avoids scanning every exterior for each hole on noisy
    // rasters. Small ring sets keep the cheaper linear scan.
    let index = if exteriors.len() > 16 && !holes.is_empty() {
        Some(RTree::bulk_load(
            bboxes
                .iter()
                .enumerate()
                .map(|(i, bbox)| {
                    let (lower, upper) = bbox.corners();
                    GeomWithData::new(Rectangle::from_corners(lower, upper), i)
                })
                .collect(),
        ))
    } else {
        None
    };
    let mut assigned: Vec<Vec<LineString<f64>>> = vec![Vec::new(); exteriors.len()];
    for hole in holes {
        let point = &hole.0[0];
        // Nested islands can have multiple containing shells. Choose the
        // smallest shell, not whichever marching row happened to visit first.
        let contains = |&i: &usize| {
            bboxes[i].contains_point(point) && point_in_ring_prechecked_bbox(point, &exteriors[i].0)
        };
        let compare = |&i: &usize, &j: &usize| {
            // Preserve input order on equal areas regardless of R-tree order.
            exteriors[i].1.total_cmp(&exteriors[j].1).then(i.cmp(&j))
        };
        let containing = if let Some(index) = &index {
            index
                .locate_in_envelope_intersecting(&AABB::from_point([point.x, point.y]))
                .map(|entry| entry.data)
                .filter(contains)
                .min_by(compare)
        } else {
            (0..exteriors.len()).filter(contains).min_by(compare)
        };
        if let Some(index) = containing {
            assigned[index].push(hole);
        }
    }
    MultiPolygon(
        exteriors
            .into_iter()
            .zip(assigned)
            .map(|((shell, _), holes)| normalize_polygon(Polygon::new(shell, holes)))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::Coord;

    fn square(lo: f64, hi: f64, exterior: bool) -> LineString<f64> {
        let mut coords = vec![
            Coord { x: lo, y: lo },
            Coord { x: hi, y: lo },
            Coord { x: hi, y: hi },
            Coord { x: lo, y: hi },
            Coord { x: lo, y: lo },
        ];
        if exterior {
            coords.reverse();
        }
        LineString(coords)
    }

    #[test]
    fn indexed_holes_use_smallest_containing_shell() {
        let mut rings = vec![
            square(0.0, 5.0, true),
            square(2.0, 3.0, true),
            square(1.0, 4.0, false),
            square(2.25, 2.75, false),
        ];
        // Trigger the spatial-index path with distant shells and holes.
        for i in 1..20 {
            let base = i as f64 * 10.0;
            rings.push(square(base, base + 5.0, true));
            rings.push(square(base + 1.0, base + 4.0, false));
        }
        let polygons = assemble_rings(rings);
        assert_eq!(polygons.0.len(), 21);
        assert!(polygons
            .0
            .iter()
            .all(|polygon| polygon.interiors().len() == 1));
        assert_eq!(signed_area(&polygons.0[0].interiors()[0]).abs(), 9.0);
        assert_eq!(signed_area(&polygons.0[1].interiors()[0]).abs(), 0.25);
    }
}
