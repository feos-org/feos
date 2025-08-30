use num_dual::linalg::LinAlgError;
use std::io;
use thiserror::Error;

/// Error type for improperly defined states and convergence problems.
#[derive(Error, Debug)]
pub enum FeosError {
    // generic error with custom message
    #[error("{0}")]
    Error(String),

    // errors related to algorithms
    #[error("`{0}` did not converge within the maximum number of iterations.")]
    NotConverged(String),
    #[error("`{0}` encountered illegal values during the iteration.")]
    IterationFailed(String),
    #[error("Iteration resulted in trivial solution.")]
    TrivialSolution,
    #[error(
        "Equation of state is initialized for {0} components while the input specifies {1} components."
    )]
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

    // errors related to file handling
    #[error(transparent)]
    FileIO(#[from] io::Error),

    // json errors
    #[error(transparent)]
    Serde(#[from] serde_json::Error),

    // errors related to parameter handling
    #[error("The following component(s) were not found: {0}")]
    ComponentsNotFound(String),
    #[error(
        "The identifier '{0}' is not known. ['cas', 'name', 'iupacname', 'smiles', inchi', 'formula']"
    )]
    IdentifierNotFound(String),
    #[error("Information missing.")]
    InsufficientInformation,
    #[error("Incompatible parameters: {0}")]
    IncompatibleParameters(String),
    #[error("Missing parameters: {0}")]
    MissingParameters(String),

    // other errors
    #[error(transparent)]
    LinAlgError(#[from] LinAlgError),
    #[cfg(feature = "rayon")]
    #[error(transparent)]
    RayonError(#[from] rayon::ThreadPoolBuildError),
    #[cfg(feature = "ndarray")]
    #[error(transparent)]
    ShapeError(#[from] ndarray::ShapeError),
}

/// Convenience type for `Result<T, FeosError>`.
pub type FeosResult<T> = Result<T, FeosError>;
