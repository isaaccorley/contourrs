#![allow(clippy::useless_conversion)] // false positive from PyO3 proc macro expansion

use arrow::array::{Array, StructArray};
use arrow::ffi::to_ffi;
use numpy::{Element, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use contourrs_core::{AffineTransform, Connectivity, RasterGrid, RasterValue};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn parse_args<'py>(
    source: &Bound<'py, pyo3::PyAny>,
    mask: Option<&Bound<'py, pyo3::PyAny>>,
    connectivity: u8,
    transform: Option<(f64, f64, f64, f64, f64, f64)>,
) -> PyResult<(Connectivity, AffineTransform, Option<Vec<bool>>, String)> {
    let conn = match connectivity {
        4 => Connectivity::Four,
        8 => Connectivity::Eight,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "connectivity must be 4 or 8",
            ))
        }
    };
    let affine = match transform {
        Some((a, b, c, d, e, f)) => AffineTransform::new(a, b, c, d, e, f),
        None => AffineTransform::identity(),
    };
    let mask_opt: Option<Vec<bool>> = if let Some(mask_arr) = mask {
        let mask_np = mask_arr
            .downcast::<numpy::PyArray2<bool>>()
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("mask must be a 2D bool array"))?;
        let mask_ro = mask_np.readonly();
        Some(mask_ro.as_slice()?.to_vec())
    } else {
        None
    };
    let dtype_str: String = source.getattr("dtype")?.getattr("name")?.extract()?;
    Ok((conn, affine, mask_opt, dtype_str))
}

fn run_polygonize<T: RasterValue + Element>(
    arr: &PyReadonlyArray2<T>,
    mask: Option<&[bool]>,
    connectivity: Connectivity,
    transform: AffineTransform,
) -> Vec<(geo_types::Polygon<f64>, f64)> {
    let shape = arr.shape();
    let (height, width) = (shape[0], shape[1]);
    let data = arr.as_slice().expect("contiguous array required");
    let grid = RasterGrid::new(data, width, height);
    contourrs_core::polygonize(&grid, mask, connectivity, transform)
}

// ---------------------------------------------------------------------------
// Dtype dispatch macros
// ---------------------------------------------------------------------------

macro_rules! dispatch_geojson {
    ($py:expr, $source:expr, $mask:expr, $conn:expr, $affine:expr, $dtype:expr,
     $($ty:ty => $name:expr),+ $(,)?) => {
        match $dtype {
            $( $name => {
                let arr = $source.downcast::<numpy::PyArray2<$ty>>()
                    .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
                        format!("Cannot interpret source as {}", $name)))?;
                let arr = arr.readonly();
                let polys = run_polygonize(&arr, $mask, $conn, $affine);
                polygons_to_geojson_list($py, &polys)
            })+
            other => Err(pyo3::exceptions::PyTypeError::new_err(
                format!("Unsupported dtype: {}", other))),
        }
    };
}

macro_rules! dispatch_arrow {
    ($py:expr, $source:expr, $mask:expr, $conn:expr, $affine:expr, $dtype:expr,
     $($ty:ty => $name:expr),+ $(,)?) => {
        match $dtype {
            $( $name => {
                let arr = $source.downcast::<numpy::PyArray2<$ty>>()
                    .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
                        format!("Cannot interpret source as {}", $name)))?;
                let arr = arr.readonly();
                let polys = run_polygonize(&arr, $mask, $conn, $affine);
                polygons_to_arrow_table($py, &polys)
            })+
            other => Err(pyo3::exceptions::PyTypeError::new_err(
                format!("Unsupported dtype: {}", other))),
        }
    };
}

macro_rules! dtype_list {
    ($mac:ident, $py:expr, $source:expr, $mask:expr, $conn:expr, $affine:expr, $dtype:expr) => {
        $mac!(
            $py, $source, $mask, $conn, $affine, $dtype,
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
) -> PyResult<PyObject> {
    let (conn, affine, mask_opt, dtype) = parse_args(source, mask, connectivity, transform)?;
    dtype_list!(
        dispatch_geojson,
        py,
        source,
        mask_opt.as_deref(),
        conn,
        affine,
        dtype.as_str()
    )
}

fn polygons_to_geojson_list(
    py: Python<'_>,
    polygons: &[(geo_types::Polygon<f64>, f64)],
) -> PyResult<PyObject> {
    let type_key = pyo3::intern!(py, "type");
    let polygon_str = pyo3::intern!(py, "Polygon");
    let coords_key = pyo3::intern!(py, "coordinates");

    let items: Vec<PyObject> = polygons
        .iter()
        .map(|(polygon, value)| {
            let dict = PyDict::new_bound(py);
            dict.set_item(type_key, polygon_str)?;

            let mut ring_objects: Vec<PyObject> = Vec::with_capacity(1 + polygon.interiors().len());
            ring_objects.push(ring_to_py(py, polygon.exterior())?);
            for hole in polygon.interiors() {
                ring_objects.push(ring_to_py(py, hole)?);
            }
            dict.set_item(coords_key, PyList::new_bound(py, &ring_objects))?;

            let tuple = PyTuple::new_bound(py, &[dict.to_object(py), value.to_object(py)]);
            Ok(tuple.to_object(py))
        })
        .collect::<PyResult<Vec<_>>>()?;

    Ok(PyList::new_bound(py, items).into())
}

fn ring_to_py(py: Python<'_>, ring: &geo_types::LineString<f64>) -> PyResult<PyObject> {
    let coord_objects: Vec<PyObject> = ring
        .0
        .iter()
        .map(|c| PyTuple::new_bound(py, [c.x, c.y]).to_object(py))
        .collect();
    Ok(PyList::new_bound(py, coord_objects).into())
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
) -> PyResult<PyObject> {
    let (conn, affine, mask_opt, dtype) = parse_args(source, mask, connectivity, transform)?;
    dtype_list!(
        dispatch_arrow,
        py,
        source,
        mask_opt.as_deref(),
        conn,
        affine,
        dtype.as_str()
    )
}

fn polygons_to_arrow_table(
    py: Python<'_>,
    polygons: &[(geo_types::Polygon<f64>, f64)],
) -> PyResult<PyObject> {
    // Build RecordBatch in Rust
    let batch = contourrs_core::arrow::polygons_to_record_batch(polygons)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Export via Arrow C Data Interface
    let struct_array = StructArray::from(batch);
    let (ffi_array, ffi_schema) = to_ffi(&struct_array.to_data())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Heap-allocate for stable pointers; pyarrow takes ownership via release callback
    let array_ptr = Box::into_raw(Box::new(ffi_array)) as usize;
    let schema_ptr = Box::into_raw(Box::new(ffi_schema)) as usize;

    // Import into PyArrow
    let pa = py.import_bound("pyarrow")?;
    let rb_cls = pa.getattr("RecordBatch")?;
    let record_batch = rb_cls.call_method1("_import_from_c", (array_ptr, schema_ptr))?;

    // Wrap as Table with GeoParquet metadata
    let geo_meta = r#"{"version":"1.1.0","primary_column":"geometry","columns":{"geometry":{"encoding":"WKB","geometry_types":["Polygon"]}}}"#;
    let meta = PyDict::new_bound(py);
    meta.set_item(pyo3::intern!(py, "geo"), geo_meta)?;

    let table_cls = pa.getattr("Table")?;
    let table = table_cls.call_method1(
        "from_batches",
        (PyList::new_bound(py, [record_batch.to_object(py)]),),
    )?;
    let table = table.call_method1("replace_schema_metadata", (meta,))?;

    Ok(table.to_object(py))
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _contourrs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(shapes, m)?)?;
    m.add_function(wrap_pyfunction!(shapes_arrow, m)?)?;
    Ok(())
}
