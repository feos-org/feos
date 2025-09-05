use num_dual::DualNum;

/// Ideal gas Helmholtz energy contribution that allows calculating derivatives
/// with respect to model parameters.
pub trait IdealGasAD<D>: Clone {
    /// Implementation of an ideal gas model in terms of the
    /// logarithm of the cubic thermal de Broglie wavelength
    /// in units ln(A³) for each component in the system.
    fn ln_lambda3_ad<D2: DualNum<f64, Inner = D> + Copy>(&self, temperature: D2) -> D2;

    /// Implementation of an ideal gas model in terms of the
    /// logarithm of the cubic thermal de Broglie wavelength
    /// in units ln(A³) for each component in the system.
    fn ln_lambda3(&self, temperature: D) -> D;

    /// The name of the ideal gas model.
    fn ideal_gas_model(&self) -> &'static str;
}

/// Ideal gas Helmholtz energy contribution without automatic differentiation.
pub trait IdealGas {
    /// Implementation of an ideal gas model in terms of the
    /// logarithm of the cubic thermal de Broglie wavelength
    /// in units ln(A³) for each component in the system.
    fn ln_lambda3<D: DualNum<f64> + Copy>(&self, temperature: D) -> D;

    /// The name of the ideal gas model.
    fn ideal_gas_model(&self) -> &'static str;
}

impl<T: IdealGas, D: DualNum<f64> + Copy> IdealGasAD<D> for &T {
    fn ln_lambda3_ad<D2: DualNum<f64> + Copy>(&self, temperature: D2) -> D2 {
        IdealGas::ln_lambda3(*self, temperature)
    }

    fn ln_lambda3(&self, temperature: D) -> D {
        IdealGas::ln_lambda3(*self, temperature)
    }

    fn ideal_gas_model(&self) -> &'static str {
        T::ideal_gas_model(self)
    }
}
