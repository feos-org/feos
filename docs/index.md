# Welcome to `feos`!

`FeOs` is a framework for thermodynamic equations of state and classical density functional theory.
It is written in Rust with a Python interface.

> `FeOs` is in active development and so is this documentation. If you have questions or want to report bugs, please [file an issue](https://github.com/feos-org/feos-core/issues) or [discuss with us](https://github.com/feos-org/feos-core/discussions).

On these pages you will find information about both the Rust library, in which all data structures and algorithms are implemented,
and the Python interfaces.

## Project Structure

`FeOs` is consists of multiple, separate Rust crates.

### Python Wheel

`feos` collects all of the below implementations of equations of state and Helmholtz functionals and exposes them to Python. The distributed Python wheel is built from `feos`.

### Core Library Crates

- `feos-core` defines the interfaces for equations of state and provides objects that can be used to compute thermodynamic properties and phase equilibria. It also defines utility functionalities e.g. for parameter fitting.
- `feos-dft` defines the interfaces for classical density functional theory and provides objects that can be used to compute density profiles, adsorption isotherms and pores in different coordinate systems and dimensions.

### Equation of State and DFT Implementations

- `feos-pcsaft` implements the PC-SAFT equation of state and the corresponding Helmholtz energy functional.
- `feos-gc-pcsaft` implements the heterosegmented group contribution PC-SAFT equation of state and the corresponding Helmholtz energy functional.

## Concepts

- Interfaces use dimensioned properties (in SI units) which makes working with the code less error-prone.
- Equations of state can be implemented in Rust (more robust, faster at runtime) or as Python `class` (faster prototyping, slower at runtime, no Rust experience needed).
- Python interfaces are written in Rust. To expose an equation of state implemented in Rust, only very little code is needed.
- Implementing an equation of state is done by implementing one or more contributions to the Helmholtz energy.
- Analytical derivatives of the Helmholtz energy are not needed - we use generalized dual numbers to evaluate the Helmholtz energy and its partial derivatives.

## Features & Algorithms

- Thermodynamic properties as (partial) derivatives of the Helmholtz energy are computed using generalized dual numbers.
- Critical point calculations for pure substances and mixtures
- Phase equilibrium calculations for pure substances and mixtures
- Utilities to construct phase diagrams for pure substances and binary mixtures
- Stability analysis
- Dynamic properties via entropy scaling
- Utilities for parameter I/O (json format)
- Example implementation: Peng-Robinson equation of state

## Getting started: Python

- [Browse the examples](examples/index.rst) or download the Jupyter notebooks provided in the github repository,
- or take a look at [the Python API](api.rst).

## Getting started: Rust

- If you want to learn how the Rust library is structured, [take a look at the Rust guide](devguide/equation_of_state/index.rst)
- or check out the Rust API.

