//! Borrowed raster buffers and conversion of supported cell values.

use std::borrow::Cow;
use std::error::Error;
use std::fmt;

/// Trait for raster cell values that can be polygonized.
pub trait RasterValue: Copy + PartialEq + Send + Sync + 'static {
    /// Convert a cell to the numeric value used in output polygons.
    fn to_f64_value(self) -> f64;

    /// Convert samples for interpolation, borrowing when the representation permits it.
    fn to_f64_slice(values: &[Self]) -> Cow<'_, [f64]> {
        Cow::Owned(values.iter().map(|value| value.to_f64_value()).collect())
    }
}

macro_rules! impl_raster_value {
    ($($t:ty),+) => {
        $(
            impl RasterValue for $t {
                #[inline]
                fn to_f64_value(self) -> f64 {
                    f64::from(self)
                }
            }
        )+
    };
}

impl_raster_value!(u8, u16, u32, i16, i32, f32);

impl RasterValue for f64 {
    #[inline]
    fn to_f64_value(self) -> f64 {
        self
    }

    fn to_f64_slice(values: &[Self]) -> Cow<'_, [f64]> {
        Cow::Borrowed(values)
    }
}

/// Raster dimensions overflow or do not match the supplied buffer length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RasterShapeError {
    width: usize,
    height: usize,
    data_len: usize,
}

impl fmt::Display for RasterShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            width,
            height,
            data_len,
        } = self;
        match width.checked_mul(*height) {
            Some(expected) => write!(
                f,
                "data length {data_len} != width*height {expected} ({width}x{height})"
            ),
            None => write!(
                f,
                "raster dimensions {width}x{height} overflow usize (data length {data_len})"
            ),
        }
    }
}

impl Error for RasterShapeError {}

/// Read-only 2D view over a raster buffer.
///
/// Public fields remain available for compatibility. Keep `data.len()` equal
/// to `width * height` when constructing or changing a grid directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterGrid<'data, T: RasterValue> {
    /// Row-major samples; its length must equal `width * height`.
    pub data: &'data [T],
    /// Number of columns.
    pub width: usize,
    /// Number of rows.
    pub height: usize,
}

impl<'a, T: RasterValue> RasterGrid<'a, T> {
    /// Create a view over an exactly sized row-major raster buffer.
    ///
    /// # Panics
    /// Panics if dimensions overflow or do not match the buffer length.
    /// Use [`Self::try_new`] to handle invalid dimensions without panicking.
    pub fn new(data: &'a [T], width: usize, height: usize) -> Self {
        Self::try_new(data, width, height).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Validate dimensions and borrow a row-major raster buffer.
    ///
    /// # Examples
    /// ```
    /// use contourrs::RasterGrid;
    /// let grid = RasterGrid::try_new(&[1u8, 2, 3, 4], 2, 2)?;
    /// assert_eq!(grid.get_checked(1, 1), Some(4));
    /// # Ok::<(), contourrs::raster::RasterShapeError>(())
    /// ```
    ///
    /// # Errors
    /// Returns [`RasterShapeError`] if `width * height` overflows or differs
    /// from `data.len()`. Empty rasters are accepted when either dimension is zero.
    pub fn try_new(data: &'a [T], width: usize, height: usize) -> Result<Self, RasterShapeError> {
        if width.checked_mul(height) != Some(data.len()) {
            return Err(RasterShapeError {
                width,
                height,
                data_len: data.len(),
            });
        }
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Read the sample at `(col, row)`.
    ///
    /// # Panics
    /// Panics if either coordinate is outside the grid or its buffer is too short.
    /// Use [`Self::get_checked`] for coordinates that may be out of bounds.
    #[must_use]
    #[inline]
    pub fn get(&self, col: usize, row: usize) -> T {
        self.get_checked(col, row).unwrap_or_else(|| {
            panic!(
                "RasterGrid::get({col}, {row}) out of bounds ({}x{}, {} samples)",
                self.width,
                self.height,
                self.data.len()
            )
        })
    }

    /// Read a sample, returning `None` for out-of-range coordinates or missing data.
    #[must_use]
    #[inline]
    pub fn get_checked(&self, col: usize, row: usize) -> Option<T> {
        if col >= self.width || row >= self.height {
            return None;
        }
        let index = row.checked_mul(self.width)?.checked_add(col)?;
        self.data.get(index).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_access() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let grid = RasterGrid::new(&data, 3, 2);
        assert_eq!(grid.get(0, 0), 1);
        assert_eq!(grid.get(2, 0), 3);
        assert_eq!(grid.get(0, 1), 4);
        assert_eq!(grid.get(2, 1), 6);
    }

    #[test]
    #[should_panic(expected = "data length 3 != width*height 4")]
    fn test_grid_size_mismatch() {
        let data: Vec<u8> = vec![1, 2, 3];
        let _ = RasterGrid::new(&data, 2, 2);
    }
}
