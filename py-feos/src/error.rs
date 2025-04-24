use feos_core::FeosError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PyFeosError {
    #[error("{0}")]
    Error(String),
    #[error(transparent)]
    FeosError(#[from] FeosError),
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
    #[cfg(feature = "rayon")]
    #[error(transparent)]
    Rayon(#[from] rayon::ThreadPoolBuildError),
}

impl From<PyFeosError> for PyErr {
    fn from(e: PyFeosError) -> PyErr {
        PyRuntimeError::new_err(e.to_string())
    }
}
