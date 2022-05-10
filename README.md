# FeOs - A Framework for Equations of State and Classical Density Functional Theory

[![documentation](https://img.shields.io/badge/docs-github--pages-blue)](https://feos-org.github.io/feos/)
[![repository](https://img.shields.io/pypi/v/feos)](https://pypi.org/project/feos/)

The `FeOs` package conveniently provides bindings to the Rust implementations of different equation of state and Helmholtz energy functional models in a single Python package.

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
|[`feos-pcsaft`](https://github.com/feos-org/feos-pcsaft)|perturbed-chain (polar) statistical associating fluid theory|✓|✓|
|[`feos-gc-pcsaft`](https://github.com/feos-org/feos-gc-pcsaft)|(heterosegmented) group contribution PC-SAFT|✓|✓|
|[`feos-pets`](https://github.com/feos-org/feos-pets)|perturbed truncated and shifted Lennard-Jones fluid|✓|✓|
|[`feos-uvtheory`](https://github.com/feos-org/feos-uvtheory)|accurate description of Mie fluids and mixtures|✓||

The list is being expanded continuously. Currently under development are implementations of ePC-SAFT and equations of state/Helmholtz energy functionals for model fluids like LJ and Mie fluids.

Other public repositories that implement models within the `FeOs` framework, but are currently not part of the `feos` Python package, are

|name|description|eos|dft|
|-|-|:-:|:-:|
|[`feos-fused-chains`](https://github.com/feos-org/feos-fused-chains)|heterosegmented fused-sphere chain functional||✓|

## Installation

`FeOs` can be installed via `pip` and runs on Windows, Linux and macOS:

```
pip install feos
```

## Building from source

To compile the code you need the Rust compiler (`rustc >= 1.53`) and `maturin` installed.
To install the package directly into the active environment, use

```
maturin develop --release
```

To build wheels, use

```
maturin build --release --out dist --no-sdist
```

## Documentation

For a documentation of the Python API, Python examples, and a guide to the underlying Rust framework check out the [documentation](https://feos-org.github.io/feos/).

## Developers

This software is currently maintained by members of the groups of
- Prof. Joachim Gross, [Institute of Thermodynamics and Thermal Process Engineering (ITT), University of Stuttgart](https://www.itt.uni-stuttgart.de/)
- Prof. André Bardow, [Energy and Process Systems Engineering (EPSE), ETH Zurich](https://epse.ethz.ch/)

## Contributing

`FeOs` grew from the need to maintain a common codebase used within the scientific work done in our groups. We share the code publicly as a platform to publish our own research but also encourage other researchers and developers to contribute their own models or implementations of existing equations of state.

If you want to contribute to ``FeOs``, there are several ways to go: improving the documentation and helping with language issues, testing the code on your systems to find bugs, adding new models or algorithms, or providing feature requests. Feel free to message us if you have questions or open an issue in this or the model-specific repositories to discuss improvements.
