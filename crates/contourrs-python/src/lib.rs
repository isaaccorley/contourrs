#![allow(clippy::useless_conversion)] // false positive from PyO3 proc macro expansion

use arrow::array::{Array, StructArray};
use arrow::ffi::to_ffi;
use numpy::{PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::conversion::IntoPyObject;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use contourrs::{AffineTransform, Connectivity, RasterGrid};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Emit let-bindings for mask guard + slice. The guard (`_mask_ro`) keeps
/// the numpy array alive; `mask_slice` is `Option<&[bool]>`.
macro_rules! extract_mask {
    ($mask:expr => $guard:ident, $slice:ident) => {
        let $guard: Option<PyReadonlyArray2<bool>> = if let Some(mask_arr) = $mask {
            let mask_np = mask_arr.cast::<numpy::PyArray2<bool>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("mask must be a 2D bool array")
            })?;
            Some(mask_np.readonly())
        } else {
            None
        };
        let $slice = $guard.as_ref().map(|m| m.as_slice()).transpose()?;
    };
}

fn parse_transform(
    connectivity: Option<u8>,
    transform: Option<(f64, f64, f64, f64, f64, f64)>,
) -> PyResult<(Option<Connectivity>, AffineTransform)> {
    let conn = match connectivity {
        Some(4) => Some(Connectivity::Four),
        Some(8) => Some(Connectivity::Eight),
        Some(_) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "connectivity must be 4 or 8",
            ))
        }
        None => None,
    };
    let affine = match transform {
        Some((a, b, c, d, e, f)) => AffineTransform::new(a, b, c, d, e, f),
        None => AffineTransform::identity(),
    };
    Ok((conn, affine))
}

// ---------------------------------------------------------------------------
// Dtype dispatch macros (polygonize)
// ---------------------------------------------------------------------------

macro_rules! dispatch_geojson {
    ($py:expr, $source:expr, $mask_slice:expr, $conn:expr, $affine:expr, $dtype:expr,
     $($ty:ty => $name:expr),+ $(,)?) => {
        match $dtype {
            $( $name => {
                let arr = $source.cast::<numpy::PyArray2<$ty>>()
                    .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
                        format!("Cannot interpret source as {}", $name)))?;
                let arr = arr.readonly();
                let data = arr.as_slice().expect("contiguous array required");
                let shape = arr.shape();
                let (height, width) = (shape[0], shape[1]);
                let polys = $py.detach(|| {
                    let grid = RasterGrid::new(data, width, height);
                    contourrs::polygonize(&grid, $mask_slice, $conn, $affine)
                });
                polygons_to_geojson_list($py, &polys)
            })+
            other => Err(pyo3::exceptions::PyTypeError::new_err(
                format!("Unsupported dtype: {}", other))),
        }
    };
}

macro_rules! dispatch_arrow {
    ($py:expr, $source:expr, $mask_slice:expr, $conn:expr, $affine:expr, $dtype:expr,
     $($ty:ty => $name:expr),+ $(,)?) => {
        match $dtype {
            $( $name => {
                let arr = $source.cast::<numpy::PyArray2<$ty>>()
                    .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
                        format!("Cannot interpret source as {}", $name)))?;
                let arr = arr.readonly();
                let data = arr.as_slice().expect("contiguous array required");
                let shape = arr.shape();
                let (height, width) = (shape[0], shape[1]);
                let polys = $py.detach(|| {
                    let grid = RasterGrid::new(data, width, height);
                    contourrs::polygonize(&grid, $mask_slice, $conn, $affine)
                });
                polygons_to_arrow_table($py, &polys)
            })+
            other => Err(pyo3::exceptions::PyTypeError::new_err(
                format!("Unsupported dtype: {}", other))),
        }
    };
}

macro_rules! dtype_list {
    ($mac:ident, $py:expr, $source:expr, $mask_slice:expr, $conn:expr, $affine:expr, $dtype:expr) => {
        $mac!(
            $py, $source, $mask_slice, $conn, $affine, $dtype,
            u8 => "uint8", u16 => "uint16", u32 => "uint32",
            i16 => "int16", i32 => "int32",
            f32 => "float32", f64 => "float64",
        )
    };
}

// ---------------------------------------------------------------------------
// Dtype dispatch macros (contours)
// ---------------------------------------------------------------------------

macro_rules! dispatch_contour_geojson {
    ($py:expr, $source:expr, $thresholds:expr, $mask_slice:expr, $affine:expr, $dtype:expr,
     $($ty:ty => $name:expr),+ $(,)?) => {
        match $dtype {
            $( $name => {
                let arr = $source.cast::<numpy::PyArray2<$ty>>()
                    .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
                        format!("Cannot interpret source as {}", $name)))?;
                let arr = arr.readonly();
                let data = arr.as_slice().expect("contiguous array required");
                let shape = arr.shape();
                let (height, width) = (shape[0], shape[1]);
                let thresholds = $thresholds;
                let polys = $py.detach(|| {
                    let grid = RasterGrid::new(data, width, height);
                    contourrs::contours(&grid, thresholds, $mask_slice, $affine)
                });
                polygons_to_geojson_list($py, &polys)
            })+
            other => Err(pyo3::exceptions::PyTypeError::new_err(
                format!("Unsupported dtype: {}", other))),
        }
    };
}

macro_rules! dispatch_contour_arrow {
    ($py:expr, $source:expr, $thresholds:expr, $mask_slice:expr, $affine:expr, $dtype:expr,
     $($ty:ty => $name:expr),+ $(,)?) => {
        match $dtype {
            $( $name => {
                let arr = $source.cast::<numpy::PyArray2<$ty>>()
                    .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
                        format!("Cannot interpret source as {}", $name)))?;
                let arr = arr.readonly();
                let data = arr.as_slice().expect("contiguous array required");
                let shape = arr.shape();
                let (height, width) = (shape[0], shape[1]);
                let thresholds = $thresholds;
                let polys = $py.detach(|| {
                    let grid = RasterGrid::new(data, width, height);
                    contourrs::contours(&grid, thresholds, $mask_slice, $affine)
                });
                polygons_to_arrow_table($py, &polys)
            })+
            other => Err(pyo3::exceptions::PyTypeError::new_err(
                format!("Unsupported dtype: {}", other))),
        }
    };
}

macro_rules! contour_dtype_list {
    ($mac:ident, $py:expr, $source:expr, $thresholds:expr, $mask_slice:expr, $affine:expr, $dtype:expr) => {
        $mac!(
            $py, $source, $thresholds, $mask_slice, $affine, $dtype,
            u8 => "uint8", u16 => "uint16", u32 => "uint32",
            i16 => "int16", i32 => "int32",
            f32 => "float32", f64 => "float64",
        )
    };
}

// ---------------------------------------------------------------------------
// shapes() — GeoJSON dicts (rasterio compat)
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (source, mask=None, connectivity=4, transform=None))]
fn shapes<'py>(
    py: Python<'py>,
    source: &Bound<'py, pyo3::PyAny>,
    mask: Option<&Bound<'py, pyo3::PyAny>>,
    connectivity: u8,
    transform: Option<(f64, f64, f64, f64, f64, f64)>,
) -> PyResult<Py<PyAny>> {
    let (conn, affine) = parse_transform(Some(connectivity), transform)?;
    let conn = conn.unwrap();
    let dtype_str: String = source.getattr("dtype")?.getattr("name")?.extract()?;

    extract_mask!(mask => _mask_ro, mask_slice);

    dtype_list!(
        dispatch_geojson,
        py,
        source,
        mask_slice,
        conn,
        affine,
        dtype_str.as_str()
    )
}

fn polygons_to_geojson_list(
    py: Python<'_>,
    polygons: &[(geo_types::Polygon<f64>, f64)],
) -> PyResult<Py<PyAny>> {
    let type_key = pyo3::intern!(py, "type");
    let polygon_str = pyo3::intern!(py, "Polygon");
    let coords_key = pyo3::intern!(py, "coordinates");

    let items: Vec<Py<PyAny>> = polygons
        .iter()
        .map(|(polygon, value)| {
            let dict = PyDict::new(py);
            dict.set_item(type_key, polygon_str)?;

            let mut ring_objects: Vec<Py<PyAny>> =
                Vec::with_capacity(1 + polygon.interiors().len());
            ring_objects.push(ring_to_py(py, polygon.exterior())?);
            for hole in polygon.interiors() {
                ring_objects.push(ring_to_py(py, hole)?);
            }
            dict.set_item(coords_key, PyList::new(py, &ring_objects)?)?;

            let tuple = PyTuple::new(
                py,
                &[
                    dict.unbind().into_any(),
                    value.into_pyobject(py)?.unbind().into_any(),
                ],
            )?;
            Ok(tuple.unbind().into_any())
        })
        .collect::<PyResult<Vec<_>>>()?;

    Ok(PyList::new(py, items)?.unbind().into_any())
}

fn ring_to_py(py: Python<'_>, ring: &geo_types::LineString<f64>) -> PyResult<Py<PyAny>> {
    let coord_objects: Vec<Py<PyAny>> = ring
        .0
        .iter()
        .map(|c| Ok(PyTuple::new(py, [c.x, c.y])?.unbind().into_any()))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, coord_objects)?.unbind().into_any())
}

// ---------------------------------------------------------------------------
// shapes_arrow() — zero-copy Arrow Table with WKB geometry
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (source, mask=None, connectivity=4, transform=None))]
fn shapes_arrow<'py>(
    py: Python<'py>,
    source: &Bound<'py, pyo3::PyAny>,
    mask: Option<&Bound<'py, pyo3::PyAny>>,
    connectivity: u8,
    transform: Option<(f64, f64, f64, f64, f64, f64)>,
) -> PyResult<Py<PyAny>> {
    let (conn, affine) = parse_transform(Some(connectivity), transform)?;
    let conn = conn.unwrap();
    let dtype_str: String = source.getattr("dtype")?.getattr("name")?.extract()?;

    extract_mask!(mask => _mask_ro, mask_slice);

    dtype_list!(
        dispatch_arrow,
        py,
        source,
        mask_slice,
        conn,
        affine,
        dtype_str.as_str()
    )
}

fn polygons_to_arrow_table(
    py: Python<'_>,
    polygons: &[(geo_types::Polygon<f64>, f64)],
) -> PyResult<Py<PyAny>> {
    // Build RecordBatch in Rust
    let batch = contourrs::arrow::polygons_to_record_batch(polygons)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Export via Arrow C Data Interface
    let struct_array = StructArray::from(batch);
    let (ffi_array, ffi_schema) = to_ffi(&struct_array.to_data())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Heap-allocate for stable pointers; pyarrow takes ownership via release callback
    let array_ptr = Box::into_raw(Box::new(ffi_array)) as usize;
    let schema_ptr = Box::into_raw(Box::new(ffi_schema)) as usize;

    // Import into PyArrow
    let pa = py.import("pyarrow")?;
    let rb_cls = pa.getattr("RecordBatch")?;
    let record_batch = rb_cls.call_method1("_import_from_c", (array_ptr, schema_ptr))?;

    // Wrap as Table with GeoParquet metadata (schema-level metadata lost in C Data Interface)
    let geo_meta = r#"{"version":"1.1.0","primary_column":"geometry","columns":{"geometry":{"encoding":"WKB","geometry_types":["Polygon"]}}}"#;
    let meta = PyDict::new(py);
    meta.set_item(pyo3::intern!(py, "geo"), geo_meta)?;

    let table_cls = pa.getattr("Table")?;
    let table = table_cls.call_method1(
        "from_batches",
        (PyList::new(py, [record_batch.unbind().into_any()])?,),
    )?;
    let table = table.call_method1("replace_schema_metadata", (meta,))?;

    Ok(table.unbind())
}

// ---------------------------------------------------------------------------
// contours() — isoband GeoJSON dicts
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (source, thresholds, mask=None, transform=None))]
fn contours<'py>(
    py: Python<'py>,
    source: &Bound<'py, pyo3::PyAny>,
    thresholds: Vec<f64>,
    mask: Option<&Bound<'py, pyo3::PyAny>>,
    transform: Option<(f64, f64, f64, f64, f64, f64)>,
) -> PyResult<Py<PyAny>> {
    let (_, affine) = parse_transform(None, transform)?;
    let dtype_str: String = source.getattr("dtype")?.getattr("name")?.extract()?;

    extract_mask!(mask => _mask_ro, mask_slice);

    contour_dtype_list!(
        dispatch_contour_geojson,
        py,
        source,
        &thresholds,
        mask_slice,
        affine,
        dtype_str.as_str()
    )
}

// ---------------------------------------------------------------------------
// contours_arrow() — isoband Arrow Table with WKB geometry
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (source, thresholds, mask=None, transform=None))]
fn contours_arrow<'py>(
    py: Python<'py>,
    source: &Bound<'py, pyo3::PyAny>,
    thresholds: Vec<f64>,
    mask: Option<&Bound<'py, pyo3::PyAny>>,
    transform: Option<(f64, f64, f64, f64, f64, f64)>,
) -> PyResult<Py<PyAny>> {
    let (_, affine) = parse_transform(None, transform)?;
    let dtype_str: String = source.getattr("dtype")?.getattr("name")?.extract()?;

    extract_mask!(mask => _mask_ro, mask_slice);

    contour_dtype_list!(
        dispatch_contour_arrow,
        py,
        source,
        &thresholds,
        mask_slice,
        affine,
        dtype_str.as_str()
    )
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _contourrs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(shapes, m)?)?;
    m.add_function(wrap_pyfunction!(shapes_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(contours, m)?)?;
    m.add_function(wrap_pyfunction!(contours_arrow, m)?)?;
    Ok(())
}
