# FeOs-AD - Implicit automatic differentiation of equations of state and phase equilibria

[![crate](https://img.shields.io/crates/v/feos-ad.svg)](https://crates.io/crates/feos-ad)
[![documentation](https://docs.rs/feos-ad/badge.svg)](https://docs.rs/feos-ad)

The `FeOs-AD` crate builds on the implementation of phase equilibrium calculations in `FeOs` to provide implicit automatic differentiation of properties and phase equilibria based on Helmholtz energy equations of state. Derivatives can be determined for any inputs, like temperature or pressure, but also model parameters.

Derivatives are calculated using forward automatic differentiation with generalized (hyper-) dual numbers from the [`num-dual`](https://github.com/itt-ustutt/num-dual) crate.

## Contents
For now, the most important properties and phase equilibria are implemented:
- **State construction**
    - from temperature and pressure
    - from pressure and entropy
    - from pressure and enthalpy
    - critical points
- **Phase equilibria**
    - bubble points
    - dew points
    - tp flash
- **Properties**
    - pressure, molar entropy, molar enthalpy

The following **models** are implemented:
- **PC-SAFT** for pure components and binary mixtures
- heterosegmented **gc-PC-SAFT** for pure components and mixtures
- The **Joback & Reid** GC method for ideal gas heat capacities

## Installation
Just add the dependency to your `Cargo.toml`
```toml
feos-ad = "0.2"
```