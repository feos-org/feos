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
In addition to the source code for the Rust and Python packages, this repository contains JSON files with previously published parameters for the different models including group contribution methods. The parameter files can be read directly from Rust or Python.

## Installation

`FeOs` can be installed via `pip` and runs on Windows, Linux and macOS:

```
pip install feos
```

If there is no compiled package for your system available from PyPI and you have a Rust compiler installed, you can instead build the python package from source using

```
pip install git+https://github.com/feos-org/feos
```

## Building from source

To compile the code you need the Rust compiler (`rustc >= 1.53`) and `maturin` installed.
To install the package directly into the active environment, use

```
maturin develop --release --cargo-extra-args="--features python"
```

and specify the models that you want to include in the python package as additional features, e.g.

```
maturin develop --release --cargo-extra-args="--features python --features pcsaft --features dft"
```

for the PC-SAFT equation of state and Helmholtz energy functional. If you want to include all available models, us

```
maturin develop --release --cargo-extra-args="--features python --features all_models"
```

To build wheels, use

```
maturin build --release --out dist --no-sdist --cargo-extra-args="--features python ..."
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
