use std::sync::Arc;

use enum_dispatch::enum_dispatch;
use feos_core::{FeosError, FeosResult};
use ndarray::{Array1, ScalarOperand};
use num_dual::DualNum;

use super::parameters::CubicParameters;

mod soave;
pub use soave::{
    PengRobinson1976, PengRobinson1978, PengRobinson2019, RedlichKwong1972, RedlichKwong2019, Soave,
};
mod mathias_copeman;
pub use mathias_copeman::MathiasCopeman;
mod twu;
pub use twu::{GeneralizedTwu, Twu};

#[enum_dispatch]
pub trait AlphaFunction {
    fn alpha<D: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        acentric_factor: &Array1<f64>,
        reduced_temperature: &Array1<D>,
    ) -> Array1<D>;

    /// Check for validity of alpha function against parameters, e.g.
    /// to assert that the number of components match.
    fn validate(&self, parameters: &Arc<CubicParameters>) -> FeosResult<()>;

    /// Generate the alpha function for a subset of components.
    fn subset(&self, component_list: &[usize]) -> Self;
}

#[enum_dispatch(AlphaFunction)]
#[derive(Debug, Clone)]
pub enum Alpha {
    Soave,
    PengRobinson1976,
    PengRobinson1978,
    PengRobinson2019,
    RedlichKwong1972,
    RedlichKwong2019,
    MathiasCopeman,
    GeneralizedTwu,
    Twu,
}
