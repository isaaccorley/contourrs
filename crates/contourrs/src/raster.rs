/// Trait for raster cell values that can be polygonized.
pub trait RasterValue: Copy + PartialEq + Send + Sync + 'static {
    fn to_f64_value(self) -> f64;
}

macro_rules! impl_raster_value {
    ($($t:ty),+) => {
        $(
            impl RasterValue for $t {
                #[inline]
                fn to_f64_value(self) -> f64 {
                    self as f64
                }
            }
        )+
    };
}

impl_raster_value!(u8, u16, u32, i16, i32, f32, f64);

/// Read-only 2D view over a raster buffer.
#[derive(Debug)]
pub struct RasterGrid<'a, T: RasterValue> {
    pub data: &'a [T],
    pub width: usize,
    pub height: usize,
}

impl<'a, T: RasterValue> RasterGrid<'a, T> {
    pub fn new(data: &'a [T], width: usize, height: usize) -> Self {
        assert_eq!(
            data.len(),
            width * height,
            "data length {} != width*height {}",
            data.len(),
            width * height,
        );
        Self {
            data,
            width,
            height,
        }
    }

    #[inline]
    pub fn get(&self, col: usize, row: usize) -> T {
        debug_assert!(
            col < self.width && row < self.height,
            "RasterGrid::get({}, {}) out of bounds ({}x{})",
            col,
            row,
            self.width,
            self.height
        );
        self.data[row * self.width + col]
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
    #[should_panic]
    fn test_grid_size_mismatch() {
        let data: Vec<u8> = vec![1, 2, 3];
        let _ = RasterGrid::new(&data, 2, 2);
    }
}
