# Additional Traits

`EquationOfState` is the only trait that is needed to compute properties, create states, and so on.
On top of that, further functionality can be added by introducing additional traits.

## `MolarWeight`

If an equation of state implements this trait, a `State` created with the equation of state additionally provides mass specific variants of all properties.

```rust
/// Molar weight of all components.
///
/// The trait is required to be able to calculate (mass)
/// specific properties.
pub trait MolarWeight<U: EOSUnit> {
    fn molar_weight(&self) -> QuantityArray1<U>;
}
```

## `EntropyScaling`

This trait provides methods to compute dynamic properties via entropy scaling.
We have to implement a *reference* to produce the reduced property and a *correlation function* that models the behavior of the logarithmic reduced property as a function of the reduced residual entropy.

```rust
/// Reference values and residual entropy correlations for entropy scaling.
pub trait EntropyScaling<U: EOSUnit, E: EquationOfState> {
    fn viscosity_reference(&self, state: &State<U, E>) -> Result<QuantityScalar<U>, EoSError>;
    fn viscosity_correlation(&self, s_res: f64, x: &Array1<f64>) -> Result<f64, EoSError>;
    fn diffusion_reference(&self, state: &State<U, E>) -> Result<QuantityScalar<U>, EoSError>;
    fn diffusion_correlation(&self, s_res: f64, x: &Array1<f64>) -> Result<f64, EoSError>;
    fn thermal_conductivity_reference(
        &self,
        state: &State<U, E>,
    ) -> Result<QuantityScalar<U>, EoSError>;
    fn thermal_conductivity_correlation(
        &self,
        s_res: f64,
        x: &Array1<f64>,
    ) -> Result<f64, EoSError>;
}

```