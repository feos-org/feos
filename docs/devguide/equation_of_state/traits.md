# Core Traits

In FeOs, an instance of an equation of state stores information of the chemical system (i.e. the model parameters) we are interested in.
But we formally make no assumptions about these model parameters - this is entirely up to the respective implementation which gives a lot of flexibility with respect to parameter handling and storage.
For example, we want to compute properties for a mixture containing water and acetone, so we create or read in the appropriate parameters and initialize an equation of state.
If we decide to work with different substances, we start over from scratch: somehow get parameters for the new mixture and create a *new* instance of the equation of state.
You can have multiple instances at the same time, and each can have their own parameters.

This concept is important, since the equation of state object, or more precisely a pointer to it, is shared between all thermodynamic states that we create with it.

## The `EquationOfState` trait

The entry point for learning to understand how equations of state are implemented in FeOs is the `EquationOfState` trait.
Shown below are the functions you have to implement when creating a new equation of state.

```rust
/// A general equation of state.
pub trait EquationOfState {
    /// Return the number of components of the equation of state.
    fn components(&self) -> usize;

    /// Return an equation of state consisting of the components
    /// contained in component_list.
    fn get_subset(&self, component_list: &[usize]) -> Self;

    /// Return the maximum density in Angstrom^-3.
    ///
    /// This value is used as an estimate for a liquid phase for phase
    /// equilibria and other iterations. It is not explicitly meant to
    /// be a mathematical limit for the density (if those exist in the
    /// equation of state anyways).
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64;

    /// Return a slice of the individual contributions (excluding the ideal gas)
    /// of the equation of state.
    fn residual(&self) -> &[Box<dyn HelmholtzEnergy>];

    // Other methods omitted.
}
```

Implementing the `components` method, you have to return the number of substances in the system you want to explore.
This method is important for algorithms that are specifically written for pure component systems where the return value of `components` is used to decide what method to use.

The `get_subset` method is used to create a new equation of state for a subset of components, where the `component_list` contains the indices of the components to pick.
This method allows calculating pure substance properties in a mixture in a convenient fashion.

The return value of `compute_max_density` is used as starting value for algorithms that perform iterations in the density (or the volume).
Note that the return value is a number density in units of inverse cubic angstrom.

The most important method is `residual`.
It returns a slice containing objects that implement the `HelmholtzEnergy` trait (in rust called "trait objects", see below).
In other words: it returns the *different contributions* to the residual Helmholtz energy.
This structure is motivated by the fact that equations of state can often be split into different, additive Helmholtz energy contributions.

Implementing `EquationOfState` comes down to defining one or more structs that all implement the `HelmholtzEnergy` trait.

## The `HelmholtzEnergyDual` and `HelmholtzEnergy` Traits

The `residual` method returns a slice of trait objects, denoted by the `dyn` keyword.

```rust
fn residual(&self) -> &[Box<dyn HelmholtzEnergy>];
```

A [trait object](https://doc.rust-lang.org/book/ch17-02-trait-objects.html) allows us to return a slice of *different structs* that share *common behavior*, i.e. all implement the same trait.
Discussion of trait objects is out of the scope of this documentation - for now it is sufficient to note that [not all traits can be made into trait objects](https://doc.rust-lang.org/book/ch17-02-trait-objects.html#object-safety-is-required-for-trait-objects).
In fact, a trait is not *object-safe* (i.e. cannot be made into a trait object) if it has generic type parameters.
This is a problem, since we want to be able to use generic dual numbers as type parameters for our equation of state.
The non-object-safe trait to do that is the `HelmholtzEnergyDual` trait:

```rust
// This trait cannot be made into a trait object
pub trait HelmholtzEnergyDual<D: DualNum<f64>> {
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D;
}
```

Our solution is to create an object-safe [supertrait](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#using-supertraits-to-require-one-traits-functionality-within-another-trait) that wraps `HelmholtzEnergyDual` with possible permutations of type parameters:

```rust
// automatically implemented for struct that implements `HelmholtzEnergyDual`
pub trait HelmholtzEnergy:
    HelmholtzEnergyDual<f64>
    + HelmholtzEnergyDual<Dual64>
    + HelmholtzEnergyDual<HyperDual64>
    + HelmholtzEnergyDual<Dual3_64>
    + HelmholtzEnergyDual<HyperDual<Dual64, f64>>
    + HelmholtzEnergyDual<Dual3<Dual64, f64>>
    + fmt::Display
{
}
```

Once we implement `HelmholtzEnergyDual` for our structs, the `HelmholtzEnergy` trait is automatically implemented and we can create and return trait objects.
This is a bit of an inconvenience but as long as we don't need `HelmholtzEnergyDual` with a new dual number it's not an issue in practice.

Note that `helmholtz_energy` function must return the **reduced** energy, i.e. \\( \frac{A^\text{res}}{k_B T}\\).

## Residual Helmholtz Energy

The residual Helmholtz energy is then computed as sum of all contributions:

```rust
/// Evaluate the residual reduced Helmholtz energy $\beta A^\mathrm{res}$.
///
/// For simple equations of state (see e.g. `PengRobinson`) it might be
/// easier to overwrite this function instead of implementing `residual`.
fn evaluate_residual<D: DualNum<f64>>(&self, state: &StateHD<D>) -> D
where
    dyn HelmholtzEnergy: HelmholtzEnergyDual<D>,
{
    self.residual()
        .iter()
        .map(|c| c.helmholtz_energy(state))
        .sum()
}
```

As noted before, this structure is informed by equations of state with multiple contributions to the residual Helmholtz energy.
If we implement an equation of state with a single contribution, it might be more convenient to overwrite the `evaluate_residual` function.

## Summary

To implement an equation of state that consists of multiple contributions to the residual Helmholtz energy, we:

1. Create a struct for our model parameters.
2. Create a struct for each contribution to the residual Helmholtz energy. Store the parameters in a reference counted pointer (e.g. `Rc`).
3. For each of these structs we implement the `HelmholtzEnergyDual` trait.
4. Create the struct that will be our equation of state. Store our model parameters (also in a `Rc`), and our contributions.
5. Then, implement `EquationOfState` (`components`, `get_subset`, `compute_max_density`, and `residual`) for this struct.
