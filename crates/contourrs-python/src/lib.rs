//! NumPy bindings for raster polygonization and filled contour bands.

mod input;
mod output;

use input::{Operation, Transform};
use pyo3::prelude::{Bound, Py, PyAny, PyModule, PyResult, Python};
use pyo3::types::PyModuleMethods;
use pyo3::{pyfunction, pymodule, wrap_pyfunction};

#[pyfunction]
#[pyo3(signature = (source, mask=None, connectivity=4, transform=None))]
fn shapes<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    mask: Option<&Bound<'py, PyAny>>,
    connectivity: u8,
    transform: Transform,
) -> PyResult<Py<PyAny>> {
    let operation = Operation::Shapes(input::parse_connectivity(connectivity)?);
    let polygons = input::polygons(py, source, mask, operation, transform)?;
    output::polygons_to_geojson_list(py, &polygons)
}

#[pyfunction]
#[pyo3(signature = (source, mask=None, connectivity=4, transform=None))]
fn shapes_arrow<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    mask: Option<&Bound<'py, PyAny>>,
    connectivity: u8,
    transform: Transform,
) -> PyResult<Py<PyAny>> {
    let operation = Operation::Shapes(input::parse_connectivity(connectivity)?);
    let polygons = input::polygons(py, source, mask, operation, transform)?;
    output::polygons_to_arrow_table(py, &polygons)
}

#[pyfunction]
#[pyo3(signature = (source, thresholds, mask=None, transform=None))]
fn contours<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    thresholds: Vec<f64>,
    mask: Option<&Bound<'py, PyAny>>,
    transform: Transform,
) -> PyResult<Py<PyAny>> {
    let polygons = input::polygons(
        py,
        source,
        mask,
        Operation::Contours(&thresholds),
        transform,
    )?;
    output::polygons_to_geojson_list(py, &polygons)
}

#[pyfunction]
#[pyo3(signature = (source, thresholds, mask=None, transform=None))]
fn contours_arrow<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    thresholds: Vec<f64>,
    mask: Option<&Bound<'py, PyAny>>,
    transform: Transform,
) -> PyResult<Py<PyAny>> {
    let polygons = input::polygons(
        py,
        source,
        mask,
        Operation::Contours(&thresholds),
        transform,
    )?;
    output::polygons_to_arrow_table(py, &polygons)
}

#[pymodule]
fn _contourrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(shapes, m)?)?;
    m.add_function(wrap_pyfunction!(shapes_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(contours, m)?)?;
    m.add_function(wrap_pyfunction!(contours_arrow, m)?)?;
    Ok(())
}
