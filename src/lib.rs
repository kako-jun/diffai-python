//! Python bindings for diffai - AI/ML model diff tool.
//!
//! This module provides Python bindings for the diffai-core library using PyO3.

#![allow(clippy::useless_conversion)]
#![allow(clippy::uninlined_format_args)]

use diffai_core::{
    diff as core_diff, diff_paths as core_diff_paths, format_output as core_format_output,
    DiffOptions, DiffResult, OutputFormat, TensorStats,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use regex::Regex;
use serde_json::Value;

// ============================================================================
// Main diff function
// ============================================================================

/// Unified diff function for Python
///
/// Compare two Python objects (dicts, lists, or primitives) and return differences.
///
/// Args:
///     old: The old value (dict, list, or primitive)
///     new: The new value (dict, list, or primitive)
///     **kwargs: Optional parameters:
///         epsilon (float): Numerical comparison tolerance
///         array_id_key (str): Key to use for array element identification
///         ignore_keys_regex (str): Regex pattern for keys to ignore
///         path_filter (str): Only show differences in paths containing this string
///         output_format (str): Output format ("diffai", "json", "yaml")
///
/// Returns:
///     List[Dict]: List of differences found
#[pyfunction]
#[pyo3(signature = (old, new, **kwargs))]
fn diff(
    py: Python,
    old: &Bound<'_, PyAny>,
    new: &Bound<'_, PyAny>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyObject> {
    let old_json = python_to_json_value(old)?;
    let new_json = python_to_json_value(new)?;
    let options = build_options_from_kwargs(kwargs)?;

    let results = core_diff(&old_json, &new_json, Some(&options)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Diff error: {e}"))
    })?;

    let py_results = PyList::empty(py);
    for result in results {
        let py_result = diff_result_to_python(py, &result)?;
        py_results.append(py_result)?;
    }

    Ok(py_results.into())
}

/// Compare two files or directories
///
/// Args:
///     old_path: Path to the old file or directory
///     new_path: Path to the new file or directory
///     **kwargs: Optional parameters (same as diff())
///
/// Returns:
///     List[Dict]: List of differences found
#[pyfunction]
#[pyo3(signature = (old_path, new_path, **kwargs))]
fn diff_paths(
    py: Python,
    old_path: &str,
    new_path: &str,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyObject> {
    let options = build_options_from_kwargs(kwargs)?;

    let results = core_diff_paths(old_path, new_path, Some(&options)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Diff error: {e}"))
    })?;

    let py_results = PyList::empty(py);
    for result in results {
        let py_result = diff_result_to_python(py, &result)?;
        py_results.append(py_result)?;
    }

    Ok(py_results.into())
}

// ============================================================================
// Format output function
// ============================================================================

/// Format diff results as string
///
/// Args:
///     results: List of diff results from diff() function
///     format: Output format ("diffai", "json", "yaml")
///
/// Returns:
///     Formatted string output
#[pyfunction]
fn format_output(results: &Bound<'_, PyList>, format: &str) -> PyResult<String> {
    let rust_results = python_results_to_rust(results)?;

    let output_format = OutputFormat::parse_format(format).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid format: {e}"))
    })?;

    core_format_output(&rust_results, output_format).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Format error: {e}"))
    })
}

// ============================================================================
// Helper functions
// ============================================================================

fn python_to_json_value(py_obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if py_obj.is_none() {
        Ok(Value::Null)
    } else if let Ok(b) = py_obj.extract::<bool>() {
        Ok(Value::Bool(b))
    } else if let Ok(i) = py_obj.extract::<i64>() {
        Ok(Value::Number(i.into()))
    } else if let Ok(f) = py_obj.extract::<f64>() {
        Ok(Value::Number(
            serde_json::Number::from_f64(f).unwrap_or(0.into()),
        ))
    } else if let Ok(s) = py_obj.extract::<String>() {
        Ok(Value::String(s))
    } else if let Ok(list) = py_obj.downcast::<PyList>() {
        let mut vec = Vec::new();
        for item in list.iter() {
            vec.push(python_to_json_value(&item)?);
        }
        Ok(Value::Array(vec))
    } else if let Ok(dict) = py_obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (key, value) in dict.iter() {
            let key_str = key.extract::<String>()?;
            let json_value = python_to_json_value(&value)?;
            map.insert(key_str, json_value);
        }
        Ok(Value::Object(map))
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported Python type",
        ))
    }
}

fn json_value_to_python(py: Python, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.to_object(py)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py))
            } else {
                Ok(py.None())
            }
        }
        Value::String(s) => Ok(s.to_object(py)),
        Value::Array(arr) => {
            let py_list = PyList::empty(py);
            for item in arr {
                let py_item = json_value_to_python(py, item)?;
                py_list.append(py_item)?;
            }
            Ok(py_list.into())
        }
        Value::Object(obj) => {
            let py_dict = PyDict::new(py);
            for (key, val) in obj {
                let py_value = json_value_to_python(py, val)?;
                py_dict.set_item(key, py_value)?;
            }
            Ok(py_dict.into())
        }
    }
}

fn tensor_stats_to_python(py: Python, stats: &TensorStats) -> PyResult<PyObject> {
    let py_dict = PyDict::new(py);
    py_dict.set_item("mean", stats.mean)?;
    py_dict.set_item("std", stats.std)?;
    py_dict.set_item("min", stats.min)?;
    py_dict.set_item("max", stats.max)?;
    py_dict.set_item("shape", &stats.shape)?;
    py_dict.set_item("dtype", &stats.dtype)?;
    py_dict.set_item("element_count", stats.element_count)?;
    Ok(py_dict.into())
}

fn diff_result_to_python(py: Python, result: &DiffResult) -> PyResult<PyObject> {
    let py_dict = PyDict::new(py);

    match result {
        DiffResult::Added(path, value) => {
            py_dict.set_item("type", "Added")?;
            py_dict.set_item("path", path)?;
            py_dict.set_item("value", json_value_to_python(py, value)?)?;
        }
        DiffResult::Removed(path, value) => {
            py_dict.set_item("type", "Removed")?;
            py_dict.set_item("path", path)?;
            py_dict.set_item("value", json_value_to_python(py, value)?)?;
        }
        DiffResult::Modified(path, old_val, new_val) => {
            py_dict.set_item("type", "Modified")?;
            py_dict.set_item("path", path)?;
            py_dict.set_item("old_value", json_value_to_python(py, old_val)?)?;
            py_dict.set_item("new_value", json_value_to_python(py, new_val)?)?;
        }
        DiffResult::TypeChanged(path, old_val, new_val) => {
            py_dict.set_item("type", "TypeChanged")?;
            py_dict.set_item("path", path)?;
            py_dict.set_item("old_value", json_value_to_python(py, old_val)?)?;
            py_dict.set_item("new_value", json_value_to_python(py, new_val)?)?;
        }
        DiffResult::TensorShapeChanged(path, old_shape, new_shape) => {
            py_dict.set_item("type", "TensorShapeChanged")?;
            py_dict.set_item("path", path)?;
            py_dict.set_item("old_shape", old_shape)?;
            py_dict.set_item("new_shape", new_shape)?;
        }
        DiffResult::TensorStatsChanged(path, old_stats, new_stats) => {
            py_dict.set_item("type", "TensorStatsChanged")?;
            py_dict.set_item("path", path)?;
            py_dict.set_item("old_stats", tensor_stats_to_python(py, old_stats)?)?;
            py_dict.set_item("new_stats", tensor_stats_to_python(py, new_stats)?)?;
        }
        DiffResult::TensorDataChanged(path, old_mean, new_mean) => {
            py_dict.set_item("type", "TensorDataChanged")?;
            py_dict.set_item("path", path)?;
            py_dict.set_item("old_mean", old_mean)?;
            py_dict.set_item("new_mean", new_mean)?;
        }
        DiffResult::ModelArchitectureChanged(path, old_arch, new_arch) => {
            py_dict.set_item("type", "ModelArchitectureChanged")?;
            py_dict.set_item("path", path)?;
            py_dict.set_item("old_architecture", old_arch)?;
            py_dict.set_item("new_architecture", new_arch)?;
        }
        DiffResult::WeightSignificantChange(path, magnitude) => {
            py_dict.set_item("type", "WeightSignificantChange")?;
            py_dict.set_item("path", path)?;
            py_dict.set_item("change_magnitude", magnitude)?;
        }
        DiffResult::ActivationFunctionChanged(path, old_fn, new_fn) => {
            py_dict.set_item("type", "ActivationFunctionChanged")?;
            py_dict.set_item("path", path)?;
            py_dict.set_item("old_activation", old_fn)?;
            py_dict.set_item("new_activation", new_fn)?;
        }
        DiffResult::LearningRateChanged(path, old_lr, new_lr) => {
            py_dict.set_item("type", "LearningRateChanged")?;
            py_dict.set_item("path", path)?;
            py_dict.set_item("old_learning_rate", old_lr)?;
            py_dict.set_item("new_learning_rate", new_lr)?;
        }
        DiffResult::OptimizerChanged(path, old_opt, new_opt) => {
            py_dict.set_item("type", "OptimizerChanged")?;
            py_dict.set_item("path", path)?;
            py_dict.set_item("old_optimizer", old_opt)?;
            py_dict.set_item("new_optimizer", new_opt)?;
        }
        DiffResult::LossChange(path, old_loss, new_loss) => {
            py_dict.set_item("type", "LossChange")?;
            py_dict.set_item("path", path)?;
            py_dict.set_item("old_loss", old_loss)?;
            py_dict.set_item("new_loss", new_loss)?;
        }
        DiffResult::AccuracyChange(path, old_acc, new_acc) => {
            py_dict.set_item("type", "AccuracyChange")?;
            py_dict.set_item("path", path)?;
            py_dict.set_item("old_accuracy", old_acc)?;
            py_dict.set_item("new_accuracy", new_acc)?;
        }
        DiffResult::ModelVersionChanged(path, old_ver, new_ver) => {
            py_dict.set_item("type", "ModelVersionChanged")?;
            py_dict.set_item("path", path)?;
            py_dict.set_item("old_version", old_ver)?;
            py_dict.set_item("new_version", new_ver)?;
        }
    }

    Ok(py_dict.into())
}

fn python_results_to_rust(results: &Bound<'_, PyList>) -> PyResult<Vec<DiffResult>> {
    let mut rust_results = Vec::new();

    for item in results.iter() {
        let dict = item.downcast::<PyDict>()?;

        let diff_type: String = dict
            .get_item("type")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'type' field"))?
            .extract()?;

        let path: String = dict
            .get_item("path")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'path' field"))?
            .extract()?;

        let result = match diff_type.as_str() {
            "Added" => {
                let value = dict.get_item("value")?.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'value' field")
                })?;
                DiffResult::Added(path, python_to_json_value(&value)?)
            }
            "Removed" => {
                let value = dict.get_item("value")?.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'value' field")
                })?;
                DiffResult::Removed(path, python_to_json_value(&value)?)
            }
            "Modified" => {
                let old_value = dict.get_item("old_value")?.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'old_value' field")
                })?;
                let new_value = dict.get_item("new_value")?.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'new_value' field")
                })?;
                DiffResult::Modified(
                    path,
                    python_to_json_value(&old_value)?,
                    python_to_json_value(&new_value)?,
                )
            }
            "TypeChanged" => {
                let old_value = dict.get_item("old_value")?.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'old_value' field")
                })?;
                let new_value = dict.get_item("new_value")?.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'new_value' field")
                })?;
                DiffResult::TypeChanged(
                    path,
                    python_to_json_value(&old_value)?,
                    python_to_json_value(&new_value)?,
                )
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid diff type: {}",
                    diff_type
                )))
            }
        };

        rust_results.push(result);
    }

    Ok(rust_results)
}

fn build_options_from_kwargs(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<DiffOptions> {
    let mut options = DiffOptions::default();

    if let Some(kwargs) = kwargs {
        if let Some(epsilon) = kwargs.get_item("epsilon")? {
            options.epsilon = Some(epsilon.extract::<f64>()?);
        }

        if let Some(array_id_key) = kwargs.get_item("array_id_key")? {
            options.array_id_key = Some(array_id_key.extract::<String>()?);
        }

        if let Some(ignore_keys_regex) = kwargs.get_item("ignore_keys_regex")? {
            let pattern: String = ignore_keys_regex.extract()?;
            let regex = Regex::new(&pattern).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid regex: {e}"))
            })?;
            options.ignore_keys_regex = Some(regex);
        }

        if let Some(path_filter) = kwargs.get_item("path_filter")? {
            options.path_filter = Some(path_filter.extract::<String>()?);
        }

        if let Some(output_format) = kwargs.get_item("output_format")? {
            let format_str: String = output_format.extract()?;
            let format = OutputFormat::parse_format(&format_str).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid output format: {e}"
                ))
            })?;
            options.output_format = Some(format);
        }
    }

    Ok(options)
}

// ============================================================================
// Python module
// ============================================================================

/// diffai-python: AI/ML model diff tool
///
/// Provides high-performance comparison of PyTorch, Safetensors, NumPy, and MATLAB files.
/// Powered by Rust for blazing fast performance.
#[pymodule]
fn diffai_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Main diff functions
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    m.add_function(wrap_pyfunction!(diff_paths, m)?)?;

    // Format output function
    m.add_function(wrap_pyfunction!(format_output, m)?)?;

    // Version
    m.add("__version__", "0.4.0")?;

    Ok(())
}
