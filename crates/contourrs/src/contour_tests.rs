use super::*;
use crate::geometry::signed_area;

fn make_grid(data: &[f64], w: usize, h: usize) -> RasterGrid<'_, f64> {
    RasterGrid::new(data, w, h)
}

#[test]
fn test_flat_raster_inside_band() {
    // All values 5.0, band [3, 7) → entire grid is inside
    let data = vec![5.0; 9];
    let grid = make_grid(&data, 3, 3);
    let result = contours(&grid, &[3.0, 7.0], None, AffineTransform::identity());
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].1, 3.0);
    // Area should be approximately 2x2 (dual grid of 3x3)
    let area = signed_area(result[0].0.exterior()).abs();
    assert!(area > 3.0, "area={} too small", area);
}

#[test]
fn test_flat_below_threshold() {
    let data = vec![1.0; 9];
    let grid = make_grid(&data, 3, 3);
    let result = contours(&grid, &[3.0, 7.0], None, AffineTransform::identity());
    assert!(result.is_empty());
}

#[test]
fn test_flat_above_threshold() {
    let data = vec![10.0; 9];
    let grid = make_grid(&data, 3, 3);
    let result = contours(&grid, &[3.0, 7.0], None, AffineTransform::identity());
    // All values above hi → no isoband (lo-exterior minus hi-hole = empty)
    assert!(result.is_empty());
}

#[test]
fn test_linear_gradient() {
    // 4x2 grid with horizontal gradient
    let data = vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0];
    let grid = make_grid(&data, 4, 2);
    let result = contours(&grid, &[0.5, 1.5, 2.5], None, AffineTransform::identity());
    assert!(!result.is_empty(), "Should produce some polygons");
    let band_values: Vec<f64> = result.iter().map(|r| r.1).collect();
    assert!(band_values.contains(&0.5));
    assert!(band_values.contains(&1.5));
}

#[test]
fn test_nan_handling() {
    let data = vec![1.0, f64::NAN, 3.0, 2.0, 2.0, 2.0, 1.0, 2.0, 3.0];
    let grid = make_grid(&data, 3, 3);
    let _result = contours(&grid, &[1.5, 2.5], None, AffineTransform::identity());
}

#[test]
fn test_ring_closure() {
    let data = vec![0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0];
    let grid = make_grid(&data, 3, 3);
    let result = contours(&grid, &[2.0, 8.0], None, AffineTransform::identity());
    for (poly, _) in &result {
        let ext = poly.exterior();
        assert!(ext.0.len() >= 4, "Ring too short");
        let first = ext.0.first().unwrap();
        let last = ext.0.last().unwrap();
        assert!(
            (first.x - last.x).abs() < 1e-10 && (first.y - last.y).abs() < 1e-10,
            "Ring not closed"
        );
    }
}

#[test]
fn test_transform_applied() {
    let data = vec![0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0];
    let grid = make_grid(&data, 3, 3);
    let transform = AffineTransform::new(10.0, 0.0, 100.0, 0.0, -10.0, 200.0);
    let result = contours(&grid, &[2.0, 8.0], None, transform);
    for (poly, _) in &result {
        for coord in &poly.exterior().0 {
            assert!(coord.x >= 100.0 - 1e-6, "x={} < 100", coord.x);
            assert!(coord.y <= 200.0 + 1e-6, "y={} > 200", coord.y);
        }
    }
}

#[test]
fn test_step_function_two_bands() {
    // Left half = 0, right half = 10
    let data = vec![0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0];
    let grid = make_grid(&data, 4, 2);
    let result = contours(&grid, &[0.0, 5.0, 10.0], None, AffineTransform::identity());
    assert!(!result.is_empty());
}

#[test]
fn test_too_few_thresholds() {
    let data = vec![1.0; 4];
    let grid = make_grid(&data, 2, 2);
    let result = contours(&grid, &[0.5], None, AffineTransform::identity());
    assert!(result.is_empty());
}

#[test]
fn test_mask() {
    let data = vec![5.0; 9];
    let grid = make_grid(&data, 3, 3);
    let mask = vec![true, true, true, true, false, true, true, true, true];
    let result = contours(&grid, &[3.0, 7.0], Some(&mask), AffineTransform::identity());
    // Center pixel masked → cells touching center produce NaN → skipped
    // Border cells (extended grid) still produce a ring around the unmasked pixels
    // Result may have polygons from the border isoline
    // Just verify it doesn't panic
    let _ = result;
}

#[test]
fn test_donut_band() {
    // Peak in center, band should form a ring (donut)
    #[rustfmt::skip]
    let data = vec![
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 2.0, 2.0, 0.0,
        0.0, 2.0, 5.0, 2.0, 0.0,
        0.0, 2.0, 2.0, 2.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let grid = make_grid(&data, 5, 5);
    let result = contours(&grid, &[1.0, 3.0], None, AffineTransform::identity());
    assert!(!result.is_empty());
    // The band [1,3) should form a donut around the center peak
    let poly = &result[0].0;
    assert_eq!(poly.interiors().len(), 1, "Should have one hole (peak)");
}

#[test]
fn test_interp() {
    assert!((interp(0.0, 10.0, 5.0) - 0.5).abs() < 1e-10);
    assert!((interp(0.0, 10.0, 0.0) - 0.0).abs() < 1e-10);
    assert!((interp(0.0, 10.0, 10.0) - 1.0).abs() < 1e-10);
    assert!((interp(5.0, 5.0, 5.0) - 0.5).abs() < 1e-10);
}

#[test]
fn test_saddle_code5_below_center() {
    // Code 5: TL=1, TR=0, BR=1, BL=0 (diagonal pattern)
    // With center < threshold → should take the "else" branch
    // Need 2x2 cell where center = (tl+tr+br+bl)/4 < threshold
    // tl=0.6, tr=0.1, br=0.6, bl=0.1 → center=0.35, threshold=0.5 → center < threshold
    #[rustfmt::skip]
    let data = vec![
        0.6, 0.1,
        0.1, 0.6,
    ];
    let grid = make_grid(&data, 2, 2);
    let result = contours(&grid, &[0.5, 1.0], None, AffineTransform::identity());
    // Should produce polygons (the saddle is resolved)
    assert!(!result.is_empty());
}

#[test]
fn test_saddle_code5_above_center() {
    // Code 5: same pattern but center >= threshold
    // tl=0.9, tr=0.1, br=0.9, bl=0.1 → center=0.5, threshold=0.5 → center >= threshold
    #[rustfmt::skip]
    let data = vec![
        0.9, 0.1,
        0.1, 0.9,
    ];
    let grid = make_grid(&data, 2, 2);
    let result = contours(&grid, &[0.5, 1.0], None, AffineTransform::identity());
    assert!(!result.is_empty());
}

#[test]
fn test_saddle_code10() {
    // Code 10: TL=0, TR=1, BR=0, BL=1 (opposite diagonal)
    // center < threshold branch
    #[rustfmt::skip]
    let data = vec![
        0.1, 0.6,
        0.6, 0.1,
    ];
    let grid = make_grid(&data, 2, 2);
    let result = contours(&grid, &[0.5, 1.0], None, AffineTransform::identity());
    assert!(!result.is_empty());
}

#[test]
fn test_saddle_code10_above_center() {
    // Code 10 with center >= threshold — need larger grid so interior cells
    // have real neighbors (not boundary NEG_INFINITY)
    #[rustfmt::skip]
    let data = vec![
        0.0, 0.1, 0.9, 0.0,
        0.0, 0.9, 0.1, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    let grid = make_grid(&data, 4, 3);
    let result = contours(&grid, &[0.3, 1.0], None, AffineTransform::identity());
    // Just verify it runs without panic and exercises the saddle code path
    let _ = result;
}

#[test]
fn test_contours_f32_grid() {
    // Exercise the Cow::Owned branch (non-f64 grid)
    let data: Vec<f32> = vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0];
    let grid = RasterGrid::new(&data, 3, 3);
    let result = contours(&grid, &[0.25, 0.75], None, AffineTransform::identity());
    assert!(!result.is_empty());
}

#[test]
fn test_contours_u8_grid() {
    // Another non-f64 type
    let data: Vec<u8> = vec![0, 5, 10, 0, 5, 10, 0, 5, 10];
    let grid = RasterGrid::new(&data, 3, 3);
    let result = contours(&grid, &[2.0, 7.0], None, AffineTransform::identity());
    assert!(!result.is_empty());
}

#[test]
fn test_negative_det_transform() {
    // Transform with negative determinant (y-flip)
    // det = a*e - b*d = 1.0 * 1.0 - 0 * 0 = 1.0 (positive)
    // For negative det: a=1, e=-1 → det = -1
    let data = vec![0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0];
    let grid = make_grid(&data, 3, 3);
    let transform = AffineTransform::new(1.0, 0.0, 0.0, 0.0, -1.0, 3.0);
    let result = contours(&grid, &[2.0, 8.0], None, transform);
    assert!(!result.is_empty());
    // With negative det, ring classification is inverted
    for (poly, _) in &result {
        let ext_area = signed_area(poly.exterior());
        // After normalize_polygon, exterior should be CCW (positive area)
        assert!(
            ext_area > 0.0,
            "exterior area={} should be positive",
            ext_area
        );
    }
}

#[test]
fn test_multiple_exteriors() {
    // Two disconnected peaks → multiple exterior rings in one band
    #[rustfmt::skip]
    let data = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let grid = make_grid(&data, 7, 3);
    let result = contours(&grid, &[2.0, 8.0], None, AffineTransform::identity());
    // Should produce two separate polygons for the two peaks
    assert!(
        result.len() >= 2,
        "expected >=2 polygons, got {}",
        result.len()
    );
}

#[test]
fn test_multiple_exteriors_with_holes() {
    // Two disconnected peaks with higher centers → multiple exteriors each with a hole
    #[rustfmt::skip]
    let data = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 3.0, 3.0, 3.0, 0.0, 3.0, 3.0, 3.0, 0.0,
        0.0, 3.0, 8.0, 3.0, 0.0, 3.0, 8.0, 3.0, 0.0,
        0.0, 3.0, 3.0, 3.0, 0.0, 3.0, 3.0, 3.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let grid = make_grid(&data, 9, 5);
    let result = contours(&grid, &[1.0, 5.0], None, AffineTransform::identity());
    // Band [1,5): the 3.0 rings are exteriors, the 8.0 centers punch holes
    assert!(!result.is_empty());
    let total_holes: usize = result.iter().map(|(p, _)| p.interiors().len()).sum();
    assert!(total_holes >= 2, "expected >=2 holes, got {}", total_holes);
}

#[test]
fn test_empty_bands_caching() {
    // Multiple threshold pairs where some bands are empty (exercise caching)
    let data = vec![5.0; 9];
    let grid = make_grid(&data, 3, 3);
    // Only band [3,7) is non-empty; [1,3) and [7,9) are empty
    let result = contours(
        &grid,
        &[1.0, 3.0, 7.0, 9.0],
        None,
        AffineTransform::identity(),
    );
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].1, 3.0);
}

#[test]
fn test_duplicate_thresholds() {
    // Duplicate thresholds should be deduped
    let data = vec![5.0; 9];
    let grid = make_grid(&data, 3, 3);
    let result = contours(
        &grid,
        &[3.0, 3.0, 7.0, 7.0],
        None,
        AffineTransform::identity(),
    );
    assert_eq!(result.len(), 1);
}

#[test]
fn test_grid_too_small() {
    // 1x1 grid — too small for marching squares
    let data = vec![5.0];
    let grid = make_grid(&data, 1, 1);
    let result = contours(&grid, &[3.0, 7.0], None, AffineTransform::identity());
    assert!(result.is_empty());
}

#[test]
fn test_interp_infinite() {
    // Both infinite
    assert!((interp(f64::NEG_INFINITY, f64::NEG_INFINITY, 5.0) - 0.5).abs() < 1e-10);
    // One infinite
    assert!((interp(f64::NEG_INFINITY, 10.0, 5.0) - 1.0).abs() < 1e-10);
    assert!((interp(0.0, f64::NEG_INFINITY, 5.0) - 0.0).abs() < 1e-10);
}

#[test]
fn test_parallel_large_grid() {
    // Grid >= 128x128 to exercise the rayon parallel path
    let mut data = vec![0.0f64; 128 * 128];
    // Create a gradient
    for row in 0..128 {
        for col in 0..128 {
            data[row * 128 + col] = (col as f64 + row as f64) / 256.0;
        }
    }
    let grid = make_grid(&data, 128, 128);
    let result = contours(
        &grid,
        &[0.1, 0.3, 0.5, 0.7, 0.9],
        None,
        AffineTransform::identity(),
    );
    assert!(!result.is_empty());
}

#[test]
fn test_all_marching_codes() {
    // Larger grid with varied values to exercise all 16 marching squares cases
    // A radial gradient from center hits many cell configurations
    let size = 16;
    let mut data = vec![0.0f64; size * size];
    let center = size as f64 / 2.0;
    for row in 0..size {
        for col in 0..size {
            let dx = col as f64 - center;
            let dy = row as f64 - center;
            data[row * size + col] = 1.0 / (1.0 + (dx * dx + dy * dy).sqrt());
        }
    }
    let grid = make_grid(&data, size, size);
    let result = contours(
        &grid,
        &[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7],
        None,
        AffineTransform::identity(),
    );
    assert!(!result.is_empty());
}

#[test]
fn test_area_conservation() {
    // 3x3 grid with vertical gradient
    let data = vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0];
    let grid = make_grid(&data, 3, 3);
    let result = contours(
        &grid,
        &[0.0, 0.25, 0.5, 0.75, 1.0],
        None,
        AffineTransform::identity(),
    );

    // Net area = exterior - holes for each polygon
    let total_area: f64 = result
        .iter()
        .map(|(poly, _)| {
            let ext = signed_area(poly.exterior()).abs();
            let holes: f64 = poly.interiors().iter().map(|h| signed_area(h).abs()).sum();
            ext - holes
        })
        .sum();

    assert!(total_area > 0.0, "total_area={}", total_area);
}

#[test]
fn test_nan_thresholds_filtered() {
    // NaN and Inf thresholds should be silently filtered
    let data = vec![5.0; 9];
    let grid = make_grid(&data, 3, 3);
    let result = contours(
        &grid,
        &[f64::NAN, 3.0, f64::INFINITY, 7.0, f64::NEG_INFINITY],
        None,
        AffineTransform::identity(),
    );
    // Only [3.0, 7.0] remains after filtering → same as normal case
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].1, 3.0);
}

#[test]
fn test_all_nan_thresholds() {
    let data = vec![5.0; 9];
    let grid = make_grid(&data, 3, 3);
    let result = contours(
        &grid,
        &[f64::NAN, f64::NAN],
        None,
        AffineTransform::identity(),
    );
    assert!(result.is_empty());
}

#[test]
fn test_short_mask_does_not_panic() {
    let data = vec![5.0; 9];
    let grid = make_grid(&data, 3, 3);
    let short_mask: Vec<bool> = vec![];
    let result = contours(
        &grid,
        &[3.0, 7.0],
        Some(&short_mask),
        AffineTransform::identity(),
    );
    assert!(result.is_empty());
}
