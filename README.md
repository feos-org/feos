# FeOs - A Framework for Equations of State and Classical Density Functional Theory

[![crate](https://img.shields.io/crates/v/feos.svg)](https://crates.io/crates/feos)
[![documentation](https://docs.rs/feos/badge.svg)](https://docs.rs/feos)
[![documentation](https://img.shields.io/badge/docs-github--pages-blue)](https://feos-org.github.io/feos/)
[![repository](https://img.shields.io/pypi/v/feos)](https://pypi.org/project/feos/)

The `FeOs` package provides Rust implementations of different equation of state and Helmholtz energy functional models and corresponding Python bindings.

```python
from feos.eos import EquationOfState, State
from feos.pcsaft import PcSaftParameters

# Build an equation of state
parameters = PcSaftParameters.from_json(['methanol'], 'parameters.json')
eos = EquationOfState.pcsaft(parameters)

# Define thermodynamic conditions
critical_point = State.critical_point(eos)

# Compute properties
p = critical_point.pressure()
t = critical_point.temperature
print(f'Critical point for methanol: T={t}, p={p}.')
```
```terminal
Critical point for methanol: T=531.5 K, p=10.7 MPa.
```

## Models
The following models are currently published as part of the `FeOs` framework

|name|description|eos|dft|
|-|-|:-:|:-:|
|`pcsaft`|perturbed-chain (polar) statistical associating fluid theory|✓|✓|
|`gc-pcsaft`|(heterosegmented) group contribution PC-SAFT|✓|✓|
|`pets`|perturbed truncated and shifted Lennard-Jones mixtures|✓|✓|
|`uvtheory`|equation of state for Mie fluids and mixtures|✓||

The list is being expanded continuously. Currently under development are implementations of ePC-SAFT, a Helmholtz energy functional for the UV theory, and SAFT-VRQ-Mie.

Other public repositories that implement models within the `FeOs` framework, but are currently not part of the `feos` Python package, are

|name|description|eos|dft|
|-|-|:-:|:-:|
|[`feos-fused-chains`](https://github.com/feos-org/feos-fused-chains)|heterosegmented fused-sphere chain functional||✓|

## Parameters
In addition to the source code for the Rust and Python packages, this repository contains JSON files with previously published [parameters](https://github.com/feos-org/feos/tree/main/parameters) for the different models including group contribution methods. The parameter files can be read directly from Rust or Python.

## Properties and phase equilibria

The crate makes use of [generalized (hyper-) dual numbers](https://github.com/itt-ustutt/num-dual) to generically calculate exact partial derivatives from Helmholtz energy equations of state. The derivatives are used to calculate
- **equilibrium properties** (pressure, heat capacity, fugacity, and *many* more),
- **transport properties** (viscosity, thermal conductivity, diffusion coefficients) using the entropy scaling approach
- **critical points** and **phase equilibria** for pure components and mixtures.

In addition to that, utilities are provided to assist in the handling of **parameters** for both molecular equations of state and (homosegmented) group contribution methods and for the generation of phase diagrams for pure components and binary mixtures.

## Classical density functional theory

`FeOs` uses efficient numerical methods to calculate density profiles in inhomogeneous systems. Highlights include:
- Fast calculation of convolution integrals in cartesian (1D, 2D and 3D), polar, cylindrical, and spherical coordinate systems using FFT and related algorithms.
- Automatic calculation of partial derivatives of Helmholtz energy densities (including temperature derivatives) using automatic differentiation with [generalized (hyper-) dual numbers](https://github.com/itt-ustutt/num-dual).
- Modeling of heterosegmented molecules, including branched molecules.
- Functionalities for calculating surface tensions, adsorption isotherms, pair correlation functions, and solvation free energies.

## Features

Without additional features activated, the command
```
cargo test --release
```
will only build and test the core functionalities of the crate. To run unit and integration tests for specific models, run
```
cargo test --release --features pcsaft
```
to test, e.g., the implementation of PC-SAFT or
```
cargo test --release --features all_models
```
to run tests on all implemented models.

## Python package

`FeOs` uses the [`PyO3`](https://github.com/PyO3/pyo3) framework to provide Python bindings. The Python package can be installed via `pip` and runs on Windows, Linux and macOS:

```
pip install feos
```

If there is no compiled package for your system available from PyPI and you have a Rust compiler installed, you can instead build the python package from source using

```
pip install git+https://github.com/feos-org/feos
```

### Building from source

To compile the code you need the Rust compiler and `maturin` (>=0.13,<0.14) installed.
To install the package directly into the active environment, use

```
maturin develop --release --features python
```

and specify the models that you want to include in the python package as additional features, e.g.

```
maturin develop --release --features "python pcsaft dft"
```

for the PC-SAFT equation of state and Helmholtz energy functional. If you want to include all available models, use

```
maturin develop --release --features "python all_models"
```

To build wheels, use

```
maturin build --release --out dist --features "python ..."
```

## Documentation

For a documentation of the Python API, Python examples, and a guide to the underlying Rust framework check out the [documentation](https://feos-org.github.io/feos/).

## Developers

This software is currently maintained by members of the groups of
- Prof. Joachim Gross, [Institute of Thermodynamics and Thermal Process Engineering (ITT), University of Stuttgart](https://www.itt.uni-stuttgart.de/)
- Prof. André Bardow, [Energy and Process Systems Engineering (EPSE), ETH Zurich](https://epse.ethz.ch/)

## Contributing

`FeOs` grew from the need to maintain a common codebase used within the scientific work done in our groups. We share the code publicly as a platform to publish our own research but also encourage other researchers and developers to contribute their own models or implementations of existing equations of state.

If you want to contribute to ``FeOs``, there are several ways to go: improving the documentation and helping with language issues, testing the code on your systems to find bugs, adding new models or algorithms, or providing feature requests. Feel free to message us if you have questions or open an issue to discuss improvements.
