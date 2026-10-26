//! Raster shape validation and bounds behavior in debug and release builds.

use std::borrow::Cow;

use contourrs::{RasterGrid, RasterValue};

#[test]
fn invalid_dimensions_are_errors() {
    for (width, height) in [(2, 3), (usize::MAX, 2)] {
        let error = RasterGrid::try_new(&[1u8, 2, 3, 4], width, height).unwrap_err();
        assert!(error.to_string().contains(&format!("{width}x{height}")));
    }
    assert!(RasterGrid::<u8>::try_new(&[], 0, 7).is_ok());
    assert!(RasterGrid::<u8>::try_new(&[], 7, 0).is_ok());
}

#[test]
fn checked_access_rejects_invalid_coordinates_and_buffers() {
    let grid = RasterGrid::try_new(&[1u8, 2, 3, 4], 2, 2).unwrap();
    assert_eq!(grid.get_checked(1, 1), Some(4));
    for (col, row) in [(2, 0), (0, 2), (usize::MAX, 0), (0, usize::MAX)] {
        assert_eq!(
            grid.get_checked(col, row),
            None,
            "coordinate ({col}, {row})"
        );
    }
    let malformed = RasterGrid {
        data: &[1u8],
        width: usize::MAX,
        height: 2,
    };
    assert_eq!(malformed.get_checked(1, 1), None);
}

#[test]
fn float64_conversion_borrows_and_other_values_convert_exactly() {
    let values = [1.25, -0.0, f64::NAN];
    let converted = f64::to_f64_slice(&values);
    assert!(matches!(converted, Cow::Borrowed(_)));
    assert_eq!(converted.as_ptr(), values.as_ptr());
    assert_eq!(
        &*u32::to_f64_slice(&[0, u32::MAX]),
        &[0.0, f64::from(u32::MAX)]
    );

    #[derive(Clone, Copy, PartialEq)]
    struct CustomValue(u8);
    impl RasterValue for CustomValue {
        fn to_f64_value(self) -> f64 {
            f64::from(self.0)
        }
    }
    assert_eq!(&*CustomValue::to_f64_slice(&[CustomValue(7)]), &[7.0]);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn column_outside_grid_cannot_read_the_next_row() {
    let grid = RasterGrid::new(&[1u8, 2, 3, 4], 2, 2);
    let _ = grid.get(2, 0);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn overflowing_row_cannot_wrap_into_the_grid() {
    let grid = RasterGrid::new(&[1u8, 2, 3, 4], 2, 2);
    let _ = grid.get(0, usize::MAX / 2 + 1);
}
