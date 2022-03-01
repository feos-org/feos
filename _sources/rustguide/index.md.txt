# Guide

Welcome to the {math}`\text{FeO}_\text{s}` Rust guide.
On the following pages we discuss the structure of the {math}`\text{FeO}_\text{s}` project and how to use, compile and extend it.

## Introduction

{math}`\text{FeO}_\text{s}` is primarily developed on Linux but it is tested and runs on Linux, macOS and Windows.
You need a [Rust compiler](https://www.rust-lang.org/tools/install) (version 1.51+) to compile the code.
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

{math}`\text{FeO}_\text{s}` is split into multiple crates that build on each other.
The most important ones are

* `feos-core` (`core` for short): defines traits and structs for equations of state and implements thermodynamic states, phase equilibria and critical point routines.
* `feos-dft` (`dft` for short): builds on `core` and defines traits and structs for classical density functional theory and implements utilities to work with convolutions, external potentials, etc.

These crates offer abstractions for tasks that are common for all equations of state and Helmholtz energy functionals.
Using `core` and `dft`, the following *implementations* of equations of state and functionals are currently available:

* `feos-pcsaft`: the [PC-SAFT equation of state](https://pubs.acs.org/doi/abs/10.1021/ie0003887).

The following crates are actively worked on and will be released in the near future:

* `feos-gc-pcsaft`: the [hetero-segmented group contribution](https://aip.scitation.org/doi/full/10.1063/1.4945000) method of the PC-SAFT equation of state.
* `feos-uv-theory`: the equation of state based on [uv-Theory](https://aip.scitation.org/doi/full/10.1063/5.0073572).
* `feos-thol`: the [Thol equation of state](https://aip.scitation.org/doi/full/10.1063/1.4945000) for pure Lennard-Jones fluids.
* `feos-pets`: the [PeTS equation of state](https://www.tandfonline.com/doi/full/10.1080/00268976.2018.1447153).


## Where to Get Help

{math}`\text{FeO}_\text{s}` is openly developed on [github](https://github.com/feos-org). Each crate has it's own github repository where you can use the *discussion* feature or file an *issue*.


```{eval-rst}
.. toctree::
   :maxdepth: 1
   :hidden:

   core/index
```
