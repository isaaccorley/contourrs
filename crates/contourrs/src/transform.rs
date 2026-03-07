/// Affine transform matching rasterio convention.
///
/// Transforms pixel coordinates (col, row) to georeferenced coordinates:
///   x = a * col + b * row + c
///   y = d * col + e * row + f
#[derive(Debug, Clone, Copy)]
pub struct AffineTransform {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
}

impl Default for AffineTransform {
    fn default() -> Self {
        Self::identity()
    }
}

impl AffineTransform {
    pub fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: 1.0,
            f: 0.0,
        }
    }

    pub fn new(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Self {
        Self { a, b, c, d, e, f }
    }

    #[inline]
    pub fn is_identity(&self) -> bool {
        self.a == 1.0
            && self.b == 0.0
            && self.c == 0.0
            && self.d == 0.0
            && self.e == 1.0
            && self.f == 0.0
    }

    /// Transform pixel coordinate to georeferenced coordinate.
    #[inline]
    pub fn apply(&self, col: f64, row: f64) -> (f64, f64) {
        let x = self.a * col + self.b * row + self.c;
        let y = self.d * col + self.e * row + self.f;
        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let t = AffineTransform::identity();
        assert_eq!(t.apply(0.0, 0.0), (0.0, 0.0));
        assert_eq!(t.apply(1.0, 0.0), (1.0, 0.0));
        assert_eq!(t.apply(0.0, 1.0), (0.0, 1.0));
    }

    #[test]
    fn test_scale_offset() {
        let t = AffineTransform::new(10.0, 0.0, 100.0, 0.0, -10.0, 200.0);
        assert_eq!(t.apply(0.0, 0.0), (100.0, 200.0));
        assert_eq!(t.apply(1.0, 0.0), (110.0, 200.0));
        assert_eq!(t.apply(0.0, 1.0), (100.0, 190.0));
    }

    #[test]
    fn test_default_is_identity() {
        let d = AffineTransform::default();
        let i = AffineTransform::identity();
        assert_eq!(d.apply(3.0, 7.0), i.apply(3.0, 7.0));
    }
}
