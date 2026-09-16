//! PyO3 bindings for the stable native continuity surface.
//!
//! Python remains the compatibility oracle for this closure. These bindings
//! expose native operations without replacing or bypassing the reference code.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use serde_json::Value;
use std::path::PathBuf;

fn py_error(error: almi_core::AlmiError) -> PyErr {
    match error {
        almi_core::AlmiError::InvalidInput(_)
        | almi_core::AlmiError::Integrity(_)
        | almi_core::AlmiError::Security(_)
        | almi_core::AlmiError::UnsupportedVersion(_) => PyValueError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

fn json_value_to_python(py: Python<'_>, value: Value) -> PyResult<Py<PyAny>> {
    let text = serde_json::to_string(&value).map_err(|error| {
        PyRuntimeError::new_err(format!("failed to serialize native result: {error}"))
    })?;
    let json = PyModule::import(py, "json")?;
    Ok(json.call_method1("loads", (text,))?.unbind())
}

#[pyfunction]
fn abi_version() -> u32 {
    almi_core::ABI_VERSION
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn init_workspace(py: Python<'_>, path: String, name: String, seed: i64) -> PyResult<Py<PyAny>> {
    let result = almi_continuity::initialize_workspace(PathBuf::from(path), &name, seed)
        .map_err(py_error)?;
    json_value_to_python(
        py,
        serde_json::to_value(result).map_err(|error| {
            PyRuntimeError::new_err(format!("failed to serialize workspace identity: {error}"))
        })?,
    )
}

#[pyfunction]
fn inspect_workspace(py: Python<'_>, path: String) -> PyResult<Py<PyAny>> {
    let result = almi_continuity::inspect_workspace(PathBuf::from(path)).map_err(py_error)?;
    json_value_to_python(
        py,
        serde_json::to_value(result).map_err(|error| {
            PyRuntimeError::new_err(format!("failed to serialize workspace inspection: {error}"))
        })?,
    )
}

#[pyfunction]
fn export_cosmos(py: Python<'_>, workspace: String, bundle: String) -> PyResult<Py<PyAny>> {
    let result = almi_cosmos::export_bundle(PathBuf::from(workspace), PathBuf::from(bundle))
        .map_err(py_error)?;
    json_value_to_python(
        py,
        serde_json::to_value(result).map_err(|error| {
            PyRuntimeError::new_err(format!("failed to serialize bundle metadata: {error}"))
        })?,
    )
}

#[pyfunction]
fn verify_cosmos(py: Python<'_>, bundle: String) -> PyResult<Py<PyAny>> {
    let result = almi_cosmos::verify_bundle(PathBuf::from(bundle)).map_err(py_error)?;
    json_value_to_python(
        py,
        serde_json::to_value(result).map_err(|error| {
            PyRuntimeError::new_err(format!("failed to serialize bundle verification: {error}"))
        })?,
    )
}

#[pyfunction]
fn inspect_cosmos(py: Python<'_>, bundle: String) -> PyResult<Py<PyAny>> {
    let result = almi_cosmos::inspect_bundle(PathBuf::from(bundle)).map_err(py_error)?;
    json_value_to_python(
        py,
        serde_json::to_value(result).map_err(|error| {
            PyRuntimeError::new_err(format!("failed to serialize bundle inspection: {error}"))
        })?,
    )
}

#[pyfunction]
fn import_cosmos(py: Python<'_>, bundle: String, workspace: String) -> PyResult<Py<PyAny>> {
    let result = almi_cosmos::import_bundle(PathBuf::from(bundle), PathBuf::from(workspace))
        .map_err(py_error)?;
    json_value_to_python(
        py,
        serde_json::to_value(result).map_err(|error| {
            PyRuntimeError::new_err(format!("failed to serialize bundle import result: {error}"))
        })?,
    )
}

#[pymodule]
fn almi_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(abi_version, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(init_workspace, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_workspace, module)?)?;
    module.add_function(wrap_pyfunction!(export_cosmos, module)?)?;
    module.add_function(wrap_pyfunction!(verify_cosmos, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_cosmos, module)?)?;
    module.add_function(wrap_pyfunction!(import_cosmos, module)?)?;
    Ok(())
}
