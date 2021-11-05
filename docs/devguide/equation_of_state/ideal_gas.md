# The Ideal Gas Contribution

The `EquationOfState` trait has an `ideal_gas` method that returns a `IdealGasContribution` trait object.
If we don't overwrite this method, it returns a default contribution in which the de Broglie wavelength is unity for each component.

This contribution is important if you are interested in non-residual properties, e.g. total heat capacities and total enthalpies.
Note that the default implementation yields the correct results for properties that do not include derivatives with respect to temperature because the de Broglie wavelength then cancels out.

```rust
/// A general equation of state.
pub trait EquationOfState {
    // other methods omitted.

    /// Return the ideal gas contribution.
    ///
    /// Per default this function returns an ideal gas contribution
    /// in which the de Broglie wavelength is 1 for every component.
    /// Therefore, the correct ideal gas pressure is obtained even
    /// with no explicit ideal gas term. If a more detailed model is
    /// required (e.g. for the calculation of enthalpies) this function
    /// has to be overwritten.
    fn ideal_gas(&self) -> &dyn IdealGasContribution {
        &DefaultIdealGasContribution()
    }
}
```

The `IdealGasContribution` supertrait is assembled from `IdealGasContributionDual` (for an explanation why, see `HelmholtzEnergy` trait), where we have to provide an implementation for the `de_broglie_wavelength` (actually \\(\ln \Lambda^3 \\) with \\([\Lambda] = A\\)):

```rust
/// Ideal gas Helmholtz energy contribution that can
/// be evaluated using generalized (hyper) dual numbers.
///
/// This trait needs to be implemented generically or for
/// the specific types in the supertraits of [IdealGasContribution]
/// so that the implementor can be used as an ideal gas
/// contribution in the equation of state.
pub trait IdealGasContributionDual<D: DualNum<f64>> {
    /// The thermal de Broglie wavelength of each component in the form $\ln\left(\frac{\Lambda^3}{\AA^3}\right)$
    fn de_broglie_wavelength(&self, temperature: D, components: usize) -> Array1<D>;

    /// Evaluate the ideal gas contribution for a given state.
    ///
    /// In some cases it could be advantageous to overwrite this
    /// implementation instead of implementing the de Broglie
    /// wavelength.
    fn evaluate(&self, state: &StateHD<D>) -> D {
        let lambda = self.de_broglie_wavelength(state.temperature, state.moles.len());
        ((lambda
            + state.partial_density.mapv(|x| {
                if x.re() == 0.0 {
                    D::from(0.0)
                } else {
                    x.ln() - 1.0
                }
            }))
            * &state.moles)
            .sum()
    }
}
```

Accordingly, the Helmholtz energy is given by

\\[ \frac{A^\text{ideal gas}}{RT} = \sum_i^{N_s} n_i (\ln [\rho_i \Lambda_i^3] - 1) \\]

where \\(i\\) is the substance index and \\(N_s\\) denotes the number of substances in the mixture.
