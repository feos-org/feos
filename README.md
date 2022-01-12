# FeOs - A Framework for Equations of State and Classical Density Functional Theory

[![documentation](https://img.shields.io/badge/docs-github--pages-blue)](https://feos-org.github.io/feos/)

This is the repository of the `FeOs` Python package.

# Models
The following models are currently published as part of the FeOs framework

|name|description|eos|DFT||
|-|-|:-:|:-:|-|
|[`feos-pcsaft`](../feos_pcsaft)|perturbed-chain (polar) statistical associating fluid theory|&#128504;|&#128504;|[![repository](https://img.shields.io/github/v/release/feos-org/feos-pcsaft?style=flat-square)](../feos-pcsaft) [![crate](https://img.shields.io/crates/v/feos-pcsaft.svg?style=flat-square)](https://crates.io/crates/feos-pcsaft) [![documentation](https://img.shields.io/docsrs/feos-pcsaft?style=flat-square)](https://docs.rs/feos-pcsaft)

The list is being expanded continuously. Currently under development are implementations of ePC-SAFT, (heterosegmented) group contribution PC-SAFT and model fluids like LJ and Mie fluids.

## Installation

`FeOs` con be installed via `pip` and runs on Windows, Linux and macOS:

```
pip install feos
```

## Building from source

To compile the code you need the Rust compiler (`rustc >= 1.53`) and `maturin` installed.
For development, use

```
maturin develop --release
```

To build wheels, use

```
maturin build --release --out dist --no-sdist
```
