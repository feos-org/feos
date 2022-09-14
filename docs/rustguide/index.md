# Guide

Welcome to the {math}`\text{FeO}_\text{s}` Rust guide.
On the following pages we discuss the structure of the {math}`\text{FeO}_\text{s}` project and how to use, compile and extend it.

## Introduction

{math}`\text{FeO}_\text{s}` is primarily developed on Linux but it is tested and runs on Linux, macOS and Windows.
You need a [Rust compiler](https://www.rust-lang.org/tools/install) to compile the code.
For development, [Visual Studio Code](https://code.visualstudio.com/) with the [rust-analyzer](https://rust-analyzer.github.io/) plugin works pretty well, but you should use what you are comfortable in.

## Prerequisites

If you are unfamiliar with Rust a good place to start is the [Rust Programming Language Book](https://doc.rust-lang.org/book/).
To start following our guide, you should understand the following topics:

* how Rust projects, called *crates*, are structured (the *module* system),
* data types and `structs`,
* the Rust *ownership model*,
* `enums` and *pattern matching*,
* *traits*,

With these foundations you should be able to follow the discussion.
Eventually you'll need to learn and understand

* limitations of *traits* and *trait objects*,
* smart pointers (`Rc` and `Box`),
* and the `pyO3` crate if you are interested in the Python interface.

## Project Structure

Some common functionalities of {math}`\text{FeO}_\text{s}` are contained in separate workspace crates so that they can be used as standalone dependencies outside of {math}`\text{FeO}_\text{s}`.
The most important ones are

* `feos-core` (`core` for short): defines traits and structs for equations of state and implements thermodynamic states, phase equilibria and critical point routines.
* `feos-dft` (`dft` for short): builds on `core` and defines traits and structs for classical density functional theory and implements utilities to work with convolutions, external potentials, etc.

These crates offer abstractions for tasks that are common for all equations of state and Helmholtz energy functionals.
Using `core` and `dft`, the following *implementations* of equations of state and functionals are currently available:

* `pcsaft`: the [PC-SAFT](https://pubs.acs.org/doi/abs/10.1021/ie0003887) equation of state and Helmholtz energy functional.
* `gc-pcsaft`: the [hetero-segmented group contribution](https://aip.scitation.org/doi/full/10.1063/1.4945000) method of the PC-SAFT equation of state.
* `uv-theory`: the equation of state based on [uv-Theory](https://aip.scitation.org/doi/full/10.1063/5.0073572).
* `pets`: the [PeTS](https://www.tandfonline.com/doi/full/10.1080/00268976.2018.1447153) equation of state and Helmholtz energy functional.

In addition to that, the `hard_sphere` and `assocation` modules contain implementations of the corresponding Helmholtz energy contributions that are used across multiple models.

To reduce compile times during development, every model is gated by its own `feature`. Specific parts of the library can be built and tested by passing the corresponding feature flags to the Rust compiler.

Due to the particular treatment of procedural macros in Rust, an additional workspace crate `feos-derive` provides procedural macros for the implementation of the `EquationOfState` and `HelmholtzEnergyFunctional` traits for `enum`s used in FFIs like `PyO3`.


## Where to Get Help

{math}`\text{FeO}_\text{s}` is openly developed on [github](https://github.com/feos-org). Each crate has it's own github repository where you can use the *discussion* feature or file an *issue*.


```{eval-rst}
.. toctree::
   :maxdepth: 1
   :hidden:

   core/index
```
