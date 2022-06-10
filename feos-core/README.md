# FeOs - A Framework for Equations of State

[![crate](https://img.shields.io/crates/v/feos-core.svg)](https://crates.io/crates/feos-core)
[![documentation](https://docs.rs/feos-core/badge.svg)](https://docs.rs/feos-core)

Core traits and functionalities for the `feos` project.

The crate makes use of [generalized (hyper-) dual numbers](https://github.com/itt-ustutt/num-dual) to generically calculate exact partial derivatives from Helmholtz energy equations of state. The derivatives are used to calculate
- **properties**,
- **critical points**,
- and **phase equilibria**.

In addition to that, utilities are provided to assist in the handling of **parameters** for both molecular equations of state and (homosegmented) group contribution methods. Mainly as a simple test case, a **cubic** equation of state is published as part of this crate. Implementations of more sophisticated models are meant to be contained in individual crates. A list of currently available implementations can be found in the [feos](https://github.com/feos-org/feos) repository.

For information on how to implement your own equation of state, check out the [documentation](https://feos-org.github.io/feos/rustguide/index.html).

## Installation

Add this to your `Cargo.toml`

```toml
[dependencies]
feos-core = "0.2"
```

## Test building python wheel

From within a Python virtual environment with `maturin` installed, type

```
maturin build --release --out dist --no-sdist -m build_wheel/Cargo.toml
```
