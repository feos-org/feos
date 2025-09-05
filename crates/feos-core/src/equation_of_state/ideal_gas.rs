use num_dual::DualNum;

/// Ideal gas Helmholtz energy contribution that allows calculating derivatives
/// with respect to model parameters.
pub trait IdealGas<D = f64> {
    /// Implementation of an ideal gas model in terms of the
    /// logarithm of the cubic thermal de Broglie wavelength
    /// in units ln(A³) for each component in the system.
    fn ln_lambda3<D2: DualNum<f64, Inner = D> + Copy>(&self, temperature: D2) -> D2;

    /// The name of the ideal gas model.
    fn ideal_gas_model(&self) -> &'static str;
}
