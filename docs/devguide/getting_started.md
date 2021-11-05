## Overview

FeOs offers a suite of structs and traits that enable modelling of fluid properties via equations of state.
These building blocks can be used to compute properties of pure substances or mixtures, implement and test new algorithms, or to model thermodynamic processes.

The core features this crate provides are:

- the `EquationOfState` trait to formulate and use equations of state with minimal implementation effort,
- the `State` object that provides methods to create thermodynamic states for different combinations of state variables or at critical conditions, and that can be used to compute thermodynamic properties.
- The `VLEState` object with which phase equilibrium calculations can be performed.

## Installation

## Development Prerequisites

- This crate was developed and tested on Linux and Mac.
- You need the [rust compiler](https://www.rust-lang.org/tools/install) with version 1.51+ installed.
- If you don't want to statically build and link BLAS/LAPACK, you need to have those libraries (we use OpenBLAS) installed and available in your PATH.
- Use the editor you are comfortable with! For us, [Visual Studio Code](https://code.visualstudio.com/) with the [rust](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust) and [rust-analyzer](https://rust-analyzer.github.io/) extensions work pretty well.

## Rust Language Prerequisites

You should familiarize yourself with the core concepts of rust before working with the code.
We recommend working through the [rust book](https://doc.rust-lang.org/book/) and (at a minimum) learn

- about data types and structs,
- the rust ownership model,
- enums and pattern matching,
- smart pointers (`Rc` and `Box`),
- generic types and traits, and
- how a rust crate is organized (module structure, etc.).

## Where to Get Help

- File an issue in the [github repository](https://www.github.com).
