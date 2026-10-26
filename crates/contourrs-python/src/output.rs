//! Convert native polygons into Python geometry dictionaries and Arrow tables.

use arrow::array::{Array, StructArray};
use arrow::ffi;
use geo_types::{LineString, Polygon};
use pyo3::conversion::IntoPyObject;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::{Py, PyAny, PyAnyMethods, PyResult, Python};
use pyo3::types::{PyDict, PyDictMethods, PyList, PyTuple};

pub(crate) fn polygons_to_geojson_list(
    py: Python<'_>,
    polygons: &[(Polygon<f64>, f64)],
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

fn ring_to_py(py: Python<'_>, ring: &LineString<f64>) -> PyResult<Py<PyAny>> {
    let coord_objects: Vec<Py<PyAny>> = ring
        .0
        .iter()
        .map(|c| Ok(PyTuple::new(py, [c.x, c.y])?.unbind().into_any()))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, coord_objects)?.unbind().into_any())
}

pub(crate) fn polygons_to_arrow_table(
    py: Python<'_>,
    polygons: &[(Polygon<f64>, f64)],
) -> PyResult<Py<PyAny>> {
    // Build RecordBatch in Rust
    let batch = contourrs::arrow::polygons_to_record_batch(polygons)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Export via Arrow C Data Interface
    let struct_array = StructArray::from(batch);
    let (mut ffi_array, mut ffi_schema) = ffi::to_ffi(&struct_array.to_data())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

    let pa = py.import("pyarrow")?;
    let rb_cls = pa.getattr("RecordBatch")?;
    // PyArrow synchronously moves the array/schema and clears their release
    // callbacks. Rust retains the C structs and drops them on every exit path;
    // an unconsumed export still releases its buffers through Arrow's Drop impl.
    let array_ptr = std::ptr::from_mut(&mut ffi_array) as usize;
    let schema_ptr = std::ptr::from_mut(&mut ffi_schema) as usize;
    let record_batch = rb_cls.call_method1("_import_from_c", (array_ptr, schema_ptr))?;

    // Layer GeoArrow field metadata + GeoParquet schema metadata onto the
    // imported schema. Field metadata may be lost through the C Data Interface.
    let imported_schema = record_batch.getattr("schema")?;

    // GeoArrow extension type metadata on geometry field
    let geo_field_meta = PyDict::new(py);
    geo_field_meta.set_item("ARROW:extension:name", "geoarrow.wkb")?;
    geo_field_meta.set_item("ARROW:extension:metadata", "{}")?;

    let geo_field = imported_schema.call_method1("field", ("geometry",))?;
    let geo_field = geo_field.call_method1("with_metadata", (geo_field_meta,))?;
    let val_field = imported_schema.call_method1("field", ("value",))?;

    // GeoParquet schema-level metadata for compatibility
    let geo_meta = r#"{"version":"1.1.0","primary_column":"geometry","columns":{"geometry":{"encoding":"WKB","crs":null,"geometry_types":["Polygon"]}}}"#;
    let schema_meta = PyDict::new(py);
    schema_meta.set_item("geo", geo_meta)?;

    let schema_fn = pa.getattr("schema")?;
    let schema = schema_fn.call1((PyList::new(
        py,
        [geo_field.unbind().into_any(), val_field.unbind().into_any()],
    )?,))?;
    let schema = schema.call_method1("with_metadata", (schema_meta,))?;

    let table_cls = pa.getattr("Table")?;
    let batches = PyList::new(py, [record_batch.unbind().into_any()])?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("schema", schema)?;
    let table = table_cls.call_method("from_batches", (batches,), Some(&kwargs))?;

    Ok(table.unbind())
}
