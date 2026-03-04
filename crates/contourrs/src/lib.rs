//! Pure Rust raster polygonization.
//!
//! Equivalent to `rasterio.features.shapes` / GDAL's `GDALPolygonize`.
//! Converts a raster grid into vector polygons with their associated values.

#[cfg(feature = "arrow")]
pub mod arrow;
pub mod connectivity;
pub mod contour;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod geometry;
pub mod label;
pub mod polygon;
pub mod raster;
pub mod trace;
pub mod transform;
mod union_find;

pub use connectivity::Connectivity;
pub use contour::contours;
pub use raster::{RasterGrid, RasterValue};
pub use transform::AffineTransform;

use geo_types::Polygon;

/// Polygonize a raster grid into vector polygons.
///
/// Returns a list of `(polygon, value)` pairs where each polygon is a
/// contiguous region of equal values in the input grid.
///
/// # Arguments
/// * `grid` - The input raster grid
/// * `mask` - Optional boolean mask (true = include pixel)
/// * `connectivity` - 4 or 8 connected pixel neighborhoods
/// * `transform` - Affine transform to apply to output coordinates
pub fn polygonize<T: RasterValue>(
    grid: &RasterGrid<T>,
    mask: Option<&[bool]>,
    connectivity: Connectivity,
    transform: AffineTransform,
) -> Vec<(Polygon<f64>, f64)> {
    if grid.width == 0 || grid.height == 0 {
        return Vec::new();
    }

    // Pass 1: Label connected regions
    let label_result = label::label_regions(grid, mask, connectivity);

    // Build value lookup: label -> f64 value
    // Find max label to size the lookup table
    let max_label = label_result
        .labels
        .iter()
        .filter(|&&l| l != u32::MAX)
        .max()
        .copied()
        .unwrap_or(0);

    let mut values = vec![0.0f64; max_label as usize + 1];
    for row in 0..grid.height {
        for col in 0..grid.width {
            let idx = row * grid.width + col;
            let label = label_result.labels[idx];
            if label != u32::MAX {
                values[label as usize] = grid.get(col, row).to_f64_value();
            }
        }
    }

    // Pass 2: Trace boundaries and assemble polygons
    let mut polygons = trace::trace_polygons(&label_result, &values, &transform);

    // Normalize ring orientations (exterior CCW, holes CW)
    polygons = polygons
        .into_iter()
        .map(|(poly, val)| (polygon::normalize_polygon(poly), val))
        .collect();

    polygons
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_value() {
        let data = vec![1u8; 4];
        let grid = RasterGrid::new(&data, 2, 2);
        let result = polygonize(&grid, None, Connectivity::Four, AffineTransform::identity());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 1.0);
    }

    #[test]
    fn test_two_values_vertical() {
        // 1 2
        // 1 2
        let data = vec![1u8, 2, 1, 2];
        let grid = RasterGrid::new(&data, 2, 2);
        let result = polygonize(&grid, None, Connectivity::Four, AffineTransform::identity());
        assert_eq!(result.len(), 2);
        let values: std::collections::HashSet<u64> =
            result.iter().map(|(_, v)| v.to_bits()).collect();
        assert!(values.contains(&1.0f64.to_bits()));
        assert!(values.contains(&2.0f64.to_bits()));
    }

    #[test]
    fn test_single_pixel() {
        let data = vec![42u8];
        let grid = RasterGrid::new(&data, 1, 1);
        let result = polygonize(&grid, None, Connectivity::Four, AffineTransform::identity());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 42.0);
        // Should be a unit square
        let exterior = result[0].0.exterior();
        assert_eq!(exterior.0.len(), 5); // closed ring: 4 corners + repeat first
    }

    #[test]
    fn test_empty_grid() {
        let data: Vec<u8> = vec![];
        let grid = RasterGrid::new(&data, 0, 0);
        let result = polygonize(&grid, None, Connectivity::Four, AffineTransform::identity());
        assert!(result.is_empty());
    }

    #[test]
    fn test_masked_all() {
        let data = vec![1u8; 4];
        let mask = vec![false; 4];
        let grid = RasterGrid::new(&data, 2, 2);
        let result = polygonize(
            &grid,
            Some(&mask),
            Connectivity::Four,
            AffineTransform::identity(),
        );
        assert!(result.is_empty());
    }

    #[test]
    fn test_with_transform() {
        let data = vec![1u8];
        let grid = RasterGrid::new(&data, 1, 1);
        let transform = AffineTransform::new(10.0, 0.0, 100.0, 0.0, -10.0, 200.0);
        let result = polygonize(&grid, None, Connectivity::Four, transform);
        assert_eq!(result.len(), 1);
        // Check that coordinates are transformed
        let exterior = result[0].0.exterior();
        let coords: Vec<(f64, f64)> = exterior.0.iter().map(|c| (c.x, c.y)).collect();
        // Origin (0,0) -> (100, 200), (1,0) -> (110, 200), etc.
        assert!(coords.contains(&(100.0, 200.0)));
        assert!(coords.contains(&(110.0, 200.0)));
    }

    #[test]
    fn test_f32_values() {
        let data = vec![1.5f32, 2.5, 1.5, 2.5];
        let grid = RasterGrid::new(&data, 2, 2);
        let result = polygonize(&grid, None, Connectivity::Four, AffineTransform::identity());
        let values: std::collections::HashSet<u64> =
            result.iter().map(|(_, v)| v.to_bits()).collect();
        assert!(values.contains(&1.5f64.to_bits()));
        assert!(values.contains(&2.5f64.to_bits()));
    }

    #[test]
    fn test_3x3_center_different() {
        // 1 1 1
        // 1 2 1
        // 1 1 1
        let data = vec![1u8, 1, 1, 1, 2, 1, 1, 1, 1];
        let grid = RasterGrid::new(&data, 3, 3);
        let result = polygonize(&grid, None, Connectivity::Four, AffineTransform::identity());
        assert_eq!(result.len(), 2);
        // One polygon for value=1 (with a hole), one for value=2
        let poly_1 = result.iter().find(|(_, v)| *v == 1.0).unwrap();
        let poly_2 = result.iter().find(|(_, v)| *v == 2.0).unwrap();
        // value=1 region should have one hole
        assert_eq!(poly_1.0.interiors().len(), 1);
        // value=2 region should have no holes
        assert_eq!(poly_2.0.interiors().len(), 0);
    }
}
