use crate::{polygonize, AffineTransform, Connectivity, RasterGrid};
use geo::{Area, Validation};

#[test]
fn corner_touching_hole_has_separate_ring() {
    let data = [
        0u8, 0, 0, 0, 0, //
        0, 1, 1, 1, 0, //
        0, 1, 0, 1, 0, //
        0, 1, 1, 0, 0, //
        0, 0, 0, 0, 0,
    ];
    let grid = RasterGrid::new(&data, 5, 5);
    for connectivity in [Connectivity::Four, Connectivity::Eight] {
        let polygons = polygonize(&grid, None, connectivity, AffineTransform::identity());
        assert_eq!(polygons.len(), 3);
        assert!(polygons.iter().all(|(polygon, _)| polygon.is_valid()));
        let foreground: Vec<_> = polygons.iter().filter(|(_, value)| *value == 1.0).collect();
        assert_eq!(foreground.len(), 1);
        assert_eq!(foreground[0].0.interiors().len(), 1);
        assert_eq!(foreground[0].0.unsigned_area(), 7.0);
    }
}

#[test]
fn corner_touching_hole_is_assigned_when_region_has_multiple_shells() {
    let data = [2u8, 2, 2, 1, 2, 1, 2, 0, 2, 2, 0, 2];
    let grid = RasterGrid::new(&data, 4, 3);
    let polygons = polygonize(
        &grid,
        None,
        Connectivity::Eight,
        AffineTransform::identity(),
    );
    let parts: Vec<_> = polygons.iter().filter(|(_, value)| *value == 2.0).collect();
    assert_eq!(parts.len(), 2);
    assert_eq!(
        parts
            .iter()
            .map(|(p, _)| p.interiors().len())
            .sum::<usize>(),
        1
    );
    assert_eq!(
        parts.iter().map(|(p, _)| p.unsigned_area()).sum::<f64>(),
        8.0
    );
    assert!(polygons.iter().all(|(polygon, _)| polygon.is_valid()));
}

#[test]
fn all_binary_three_by_three_rasters_have_valid_pixel_areas() {
    for pattern in 0..512u16 {
        let data: Vec<u8> = (0..9).map(|bit| ((pattern >> bit) & 1) as u8).collect();
        let grid = RasterGrid::new(&data, 3, 3);
        for connectivity in [Connectivity::Four, Connectivity::Eight] {
            let polygons = polygonize(&grid, None, connectivity, AffineTransform::identity());
            for (polygon, _) in &polygons {
                assert!(polygon.is_valid(), "pattern {pattern}: {polygon:?}");
            }
            for value in [0u8, 1] {
                let area: f64 = polygons
                    .iter()
                    .filter(|(_, v)| *v == f64::from(value))
                    .map(|(polygon, _)| polygon.unsigned_area())
                    .sum();
                assert_eq!(area, data.iter().filter(|&&v| v == value).count() as f64);
            }
        }
    }
}
