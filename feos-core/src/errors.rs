use crate::parameter::ParameterError;
use num_dual::linalg::LinAlgError;
use thiserror::Error;

/// Error type for improperly defined states and convergence problems.
#[derive(Error, Debug)]
pub enum EosError {
    #[error("{0}")]
    Error(String),
    #[error("`{0}` did not converge within the maximum number of iterations.")]
    NotConverged(String),
    #[error("`{0}` encountered illegal values during the iteration.")]
    IterationFailed(String),
    #[error("Iteration resulted in trivial solution.")]
    TrivialSolution,
    #[error("Equation of state is initialized for {0} components while the input specifies {1} components.")]
    IncompatibleComponents(usize, usize),
    #[error("Invalid state in {0}: {1} = {2}.")]
    InvalidState(String, String, f64),
    #[error("Undetermined state: {0}.")]
    UndeterminedState(String),
    #[error("System is supercritical.")]
    SuperCritical,
    #[error("No phase split according to stability analysis.")]
    NoPhaseSplit,
    #[error("Wrong input units. Expected {0}, got {1}")]
    WrongUnits(String, String),
    #[error(transparent)]
    ParameterError(#[from] ParameterError),
    #[error(transparent)]
    LinAlgError(#[from] LinAlgError),
    #[cfg(feature = "rayon")]
    #[error(transparent)]
    RayonError(#[from] rayon::ThreadPoolBuildError),
}

/// Convenience type for `Result<T, EosError>`.
pub type EosResult<T> = Result<T, EosError>;
