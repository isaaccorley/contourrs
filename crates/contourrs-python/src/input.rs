//! Validate NumPy inputs and dispatch each supported dtype to the Rust core.

use contourrs::{AffineTransform, Connectivity, RasterGrid, RasterValue};
use geo_types::Polygon;
use numpy::{Element, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::{Bound, PyAny, PyAnyMethods, PyResult, Python};

pub(crate) type Transform = Option<(f64, f64, f64, f64, f64, f64)>;

#[derive(Clone, Copy)]
pub(crate) enum Operation<'thresholds> {
    Shapes(Connectivity),
    Contours(&'thresholds [f64]),
}

pub(crate) fn parse_connectivity(value: u8) -> PyResult<Connectivity> {
    match value {
        4 => Ok(Connectivity::Four),
        8 => Ok(Connectivity::Eight),
        _ => Err(PyValueError::new_err("connectivity must be 4 or 8")),
    }
}

fn parse_transform(transform: Transform) -> AffineTransform {
    transform.map_or_else(AffineTransform::identity, |(a, b, c, d, e, f)| {
        AffineTransform::new(a, b, c, d, e, f)
    })
}

fn extract_mask<'py>(
    mask: Option<&Bound<'py, PyAny>>,
    source: &Bound<'py, PyAny>,
) -> PyResult<Option<PyReadonlyArray2<'py, bool>>> {
    let Some(mask) = mask else { return Ok(None) };
    let array = mask
        .cast::<PyArray2<bool>>()
        .map_err(|_| PyTypeError::new_err("mask must be a 2D bool array"))?;
    let guard = array.try_readonly()?;
    if !guard.is_c_contiguous() {
        return Err(PyValueError::new_err("mask must be C-contiguous"));
    }
    let source_shape: Vec<usize> = source
        .getattr("shape")
        .and_then(|shape| shape.extract())
        .map_err(|_| PyValueError::new_err("source must have a 2D 'shape' attribute"))?;
    if source_shape.len() < 2 {
        return Err(PyValueError::new_err("source must be at least 2D"));
    }
    let mask_shape = guard.shape();
    if mask_shape != &source_shape[..2] {
        return Err(PyValueError::new_err(format!(
            "mask shape {mask_shape:?} does not match source shape {:?}",
            &source_shape[..2]
        )));
    }
    Ok(Some(guard))
}

pub(crate) fn polygons<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    mask: Option<&Bound<'py, PyAny>>,
    operation: Operation<'_>,
    transform: Transform,
) -> PyResult<Vec<(Polygon<f64>, f64)>> {
    let dtype: String = source.getattr("dtype")?.getattr("name")?.extract()?;
    let mask_guard = extract_mask(mask, source)?;
    let mask = mask_guard
        .as_ref()
        .map(|mask| mask.as_slice())
        .transpose()?;
    let transform = parse_transform(transform);

    // Runtime NumPy dtypes select statically typed Rust slices. All validation
    // and computation live in the generic function, independent of output format.
    macro_rules! dispatch {
        ($($name:literal => $ty:ty),+ $(,)?) => {
            match dtype.as_str() {
                $($name => process_array::<$ty>(py, source, $name, mask, operation, transform),)+
                other => Err(PyTypeError::new_err(format!("Unsupported dtype: {other}"))),
            }
        };
    }
    dispatch!(
        "uint8" => u8, "uint16" => u16, "uint32" => u32,
        "int16" => i16, "int32" => i32, "float32" => f32, "float64" => f64,
    )
}

fn process_array<T: Element + RasterValue>(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    dtype: &str,
    mask: Option<&[bool]>,
    operation: Operation<'_>,
    transform: AffineTransform,
) -> PyResult<Vec<(Polygon<f64>, f64)>> {
    let array = source
        .cast::<PyArray2<T>>()
        .map_err(|_| PyTypeError::new_err(format!("Cannot interpret source as {dtype}")))?;
    let guard = array.try_readonly()?;
    if !guard.is_c_contiguous() {
        return Err(PyValueError::new_err("source must be C-contiguous"));
    }
    let data = guard
        .as_slice()
        .map_err(|_| PyValueError::new_err("source must be C-contiguous"))?;
    let shape = guard.shape();
    let grid = RasterGrid::try_new(data, shape[1], shape[0])
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    // Both NumPy guards remain alive while Rust works without the GIL.
    Ok(py.detach(|| match operation {
        Operation::Shapes(connectivity) => {
            contourrs::polygonize(&grid, mask, connectivity, transform)
        }
        Operation::Contours(thresholds) => contourrs::contours(&grid, thresholds, mask, transform),
    }))
}
