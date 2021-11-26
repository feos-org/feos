# FeOs - A Framework for Equations of State and Classical Density Functional Theory

[![documentation](https://img.shields.io/badge/docs-github--pages-blue)](https://feos-org.github.io/feos/)

This is the repository of the `FeOs` Python package.

## Installation

Currently, `FeOs` is not hosted on pypi because we still work on the interfaces.
Once it is on pypi, you will be able to install it via `pip`:

```
pip install feos
```

## Building from source

To compile the code you need the Rust compiler (`rustc >= 1.51`) and `maturin` installed.
For development, use

```
maturin develop --release
```

For `develop` to work you need openBLAS installed and in your PATH.

To build wheels, use

```
maturin build --release --out dist --no-sdist --cargo-extra-args="--no-default-features --features openblas-static"
```

which statically links to openBLAS so that the wheel is manylinux compatible.

