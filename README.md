# FeOs - A Framework for Equations of State and Classical Density Functional Theory

[![documentation](https://img.shields.io/badge/docs-github--pages-blue)](https://feos-org.github.io/feos/)
[![repository](https://img.shields.io/pypi/v/feos)](https://pypi.org/project/feos/)

The `FeOs` package conveniently provides bindings to the Rust implementations of different equation of state and Helmholtz energy functional models in a single Python package.

## Models
The following models are currently published as part of the `FeOs` framework

|name|description|eos|dft|
|-|-|:-:|:-:|
|[`feos-pcsaft`](https://github.com/feos-org/feos-pcsaft)|perturbed-chain (polar) statistical associating fluid theory|&#128504;|&#128504;|

The list is being expanded continuously. Currently under development are implementations of ePC-SAFT, (heterosegmented) group contribution PC-SAFT and equations of state/Helmholtz energy functionals for model fluids like LJ and Mie fluids.

Other public repositories that implement models within the `FeOs` framework, but are currently not part of the `feos` Python package, are

|name|description|eos|dft|
|-|-|:-:|:-:|
|[`feos-fused-chains`](https://github.com/feos-org/feos-fused-chains)|heterosegmented fused-sphere chain functional||&#128504;|

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