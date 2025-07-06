use feos_core::{EquationOfState, IdealGas, Residual};

pub struct Strategy<R: Residual, I: IdealGas> {
    eos: EquationOfState<R, I>
}