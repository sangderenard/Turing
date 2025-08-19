use pyo3::prelude::*;

/// Create a buffer with the given size.
#[pyfunction]
fn create_buffer(_size: usize) -> PyResult<()> {
    // Placeholder implementation
    Err(pyo3::exceptions::PyNotImplementedError::new_err("rust backend stub"))
}

#[pymodule]
fn rust_backend(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_buffer, m)?)?;
    Ok(())
}
