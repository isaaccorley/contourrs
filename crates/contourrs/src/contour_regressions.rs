use super::*;
use geo::{Area, Validation};

#[test]
fn test_basin_inside_upper_superlevel_is_preserved() {
    let mut data = vec![5.0; 49];
    for row in 2..5 {
        for col in 2..5 {
            data[row * 7 + col] = 2.0;
        }
    }
    let grid = RasterGrid::new(&data, 7, 7);
    let polygons = contours(&grid, &[1.0, 3.0], None, AffineTransform::identity());
    assert_eq!(
        polygons.len(),
        1,
        "the lower basin must survive subtraction"
    );
    assert!(polygons[0].0.is_valid());
    assert!(polygons[0].0.unsigned_area() > 4.0);
}

#[test]
fn test_boundary_touching_band_is_valid() {
    let data: Vec<f64> = (0..16).map(|i| (i % 4) as f64).collect();
    let grid = RasterGrid::new(&data, 4, 4);
    let polygons = contours(&grid, &[0.5, 1.5], None, AffineTransform::identity());
    assert_eq!(polygons.len(), 1);
    assert!(
        polygons[0].0.is_valid(),
        "shared raster boundaries must be clipped, not nested holes"
    );
}

#[test]
fn test_translated_band_preserves_area() {
    let data: Vec<f64> = (0..16).map(|i| (i % 4) as f64).collect();
    let grid = RasterGrid::new(&data, 4, 4);
    let transform = AffineTransform::new(1.0, 0.0, 1e9, 0.0, -1.0, 1e9);
    let original = contours(&grid, &[0.5, 1.5], None, AffineTransform::identity());
    let translated = contours(&grid, &[0.5, 1.5], None, transform);
    assert_eq!(original.len(), translated.len());
    assert_eq!(
        original[0].0.unsigned_area(),
        translated[0].0.unsigned_area()
    );
}

#[test]
fn test_flat_contour_stays_within_sample_extent() {
    let data = vec![2.0; 25];
    let grid = RasterGrid::new(&data, 5, 5);
    let polygons = contours(&grid, &[1.0, 3.0], None, AffineTransform::identity());
    assert_eq!(polygons.len(), 1);
    assert_eq!(polygons[0].0.unsigned_area(), 16.0);
    for c in &polygons[0].0.exterior().0 {
        assert!((0.0..=4.0).contains(&c.x));
        assert!((0.0..=4.0).contains(&c.y));
    }
}

#[test]
fn test_mask_and_nonfinite_samples_close_boundaries() {
    let grid_data = vec![2.0; 25];
    for excluded in [0, 12] {
        let mut mask = vec![true; 25];
        mask[excluded] = false;
        let grid = RasterGrid::new(&grid_data, 5, 5);
        let masked = contours(&grid, &[1.0, 3.0], Some(&mask), AffineTransform::identity());
        assert_eq!(masked.len(), 1);
        assert!(masked[0].0.is_valid());
        assert!(masked[0].0.unsigned_area() < 16.0);
        for missing in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut data = grid_data.clone();
            data[excluded] = missing;
            let grid = RasterGrid::new(&data, 5, 5);
            let result = contours(&grid, &[1.0, 3.0], None, AffineTransform::identity());
            assert_eq!(result, masked);
        }
    }
}

#[test]
fn test_interpolation_extreme_scales() {
    assert!((interp(0.0, 1e-20, 2.5e-21) - 0.25).abs() < 1e-12);
    assert!((interp(-1e308, 1e308, 0.0) - 0.5).abs() < 1e-12);
}

#[test]
fn test_bands_partition_sample_extent() {
    // Exercise saddles and exact-threshold vertices with deterministic fields.
    for seed in 0..10 {
        let data: Vec<f64> = (0..72)
            .map(|i| ((i * 37 + seed * 13) % 101) as f64 / 100.0)
            .collect();
        let grid = RasterGrid::new(&data, 9, 8);
        let polygons = contours(
            &grid,
            &[-1.0, 0.25, 0.5, 0.75, 2.0],
            None,
            AffineTransform::identity(),
        );
        assert!(polygons.iter().all(|(p, _)| p.is_valid()));
        let total: f64 = polygons.iter().map(|(p, _)| p.unsigned_area()).sum();
        assert!((total - 56.0).abs() < 1e-6, "seed {seed}: area {total}");
    }
}
