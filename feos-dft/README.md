# FeOs-DFT

[![crate](https://img.shields.io/crates/v/feos-dft.svg)](https://crates.io/crates/feos-dft)
[![documentation](https://docs.rs/feos-dft/badge.svg)](https://docs.rs/feos-dft)

Generic classical DFT implementations for the `feos` project.

The crate makes use of efficient numerical methods to calculate density profiles in inhomogeneous systems. Highlights include:
- Fast calculationof convolution integrals in cartesian (1D, 2D and 3D), polar, cylindrical, and spherical coordinate systems using FFT and related algorithms.
- Automatic calculation of partial derivatives of Helmholtz energy densities (including temperature derivatives) using automatic differentiation with [generalized (hyper-) dual numbers](https://github.com/itt-ustutt/num-dual).
- Modeling of heterosegmented molecules, including branched molecules.
- Functionalities for calculating surface tensions, adsorption isotherms, pair correlation functions, and solvation free energies.

As an example and for testing purposes different versions of fundamental measure theory for hard-sphere systems are included. Implementations for more sophisticated models can be found in the [feos](https://github.com/feos-org/feos) repository.

## Installation

Add this to your `Cargo.toml`

```toml
[dependencies]
feos-dft = "0.2"
```