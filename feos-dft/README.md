# FeOs-DFT

[![crate](https://img.shields.io/crates/v/feos-dft.svg)](https://crates.io/crates/feos-dft)
[![documentation](https://docs.rs/feos-dft/badge.svg)](https://docs.rs/feos-dft)
[![minimum rustc 1.51](https://img.shields.io/badge/rustc-1.51+-red.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)

Generic classical DFT implementations for the `feos` project.

## Installation

Add this to your `Cargo.toml`

```toml
[dependencies]
feos-dft = "0.2"
```

## Test building python wheel

From within a Python virtual environment with `maturin` installed, type

```
maturin build --release --out dist --no-sdist -m build_wheel/Cargo.toml
```
