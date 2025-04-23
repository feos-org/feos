use feos_core::FeosError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
// use pythonize::PythonizeError;
use thiserror::Error;

pub type PyFeosResult<T> = Result<T, PyFeosError>;

// pub struct PyFeosError(pub FeosError);
// #[derive(Error, Debug)]
// #[error(transparent)]
// pub struct PyFeosError(#[from] pub FeosError);

#[derive(Debug, Error)]
pub enum PyFeosError {
    #[error("{0}")]
    Error(String),
    #[error(transparent)]
    FeosError(#[from] FeosError),
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
    // #[error(transparent)]
    // PythonizeError(#[from] PythonizeError),
}

// impl From<FeosError> for PyFeosError {
//     fn from(value: FeosError) -> Self {
//         Self(value)
//     }
// }

impl From<PyFeosError> for PyErr {
    fn from(e: PyFeosError) -> PyErr {
        PyRuntimeError::new_err(e.to_string())
    }
}
