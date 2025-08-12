use num_dual::DualNum;
use std::ops::Deref;

/// Ideal gas Helmholtz energy contribution.
pub trait IdealGas<D = f64>: Clone {
    type Real: IdealGas;
    type Lifted<D2: DualNum<f64, Inner = D> + Copy>: IdealGas<D2>;
    fn re(&self) -> Self::Real;
    fn lift<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2>;

    /// Implementation of an ideal gas model in terms of the
    /// logarithm of the cubic thermal de Broglie wavelength
    /// in units ln(A³) for each component in the system.
    fn ln_lambda3(&self, temperature: D) -> D;

    /// The name of the ideal gas model.
    fn ideal_gas_model(&self) -> &'static str;
}

pub trait IdealGasDyn {
    /// Implementation of an ideal gas model in terms of the
    /// logarithm of the cubic thermal de Broglie wavelength
    /// in units ln(A³) for each component in the system.
    fn ln_lambda3<D: DualNum<f64> + Copy>(&self, temperature: D) -> D;

    /// The name of the ideal gas model.
    fn ideal_gas_model(&self) -> &'static str;
}

impl<C: Deref<Target = T> + Clone, T: IdealGasDyn, D: DualNum<f64> + Copy> IdealGas<D> for C {
    type Real = Self;
    type Lifted<D2: DualNum<f64, Inner = D> + Copy> = Self;
    fn re(&self) -> Self::Real {
        self.clone()
    }
    fn lift<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
        self.clone()
    }

    fn ln_lambda3(&self, temperature: D) -> D {
        IdealGasDyn::ln_lambda3(self.deref(), temperature)
    }

    fn ideal_gas_model(&self) -> &'static str {
        T::ideal_gas_model(self)
    }
}
