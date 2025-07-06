# `feos-core`

In this section, we discuss the `feos-core` crate.
We will learn how equations of state are abstracted using traits, how generalized (hyper-) dual numbers are utilized and how thermodynamic states and phase equilibria are defined.

## Setup

To setup your workspace, [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the [`feos` github repository](https://github.com/feos-org/feos) and from that clone the fork.

To build the Rust library, switch to the `feos-core` crate
```
cd feos-core
```

and type:

```
cargo build
```

## Important crates

`feos-core` depends on a number of other important crates. Most notably, we use

- `quantity`: for scalar and vector valued dimensioned quantities. Those are used in almost all user-facing interfaces.
- `num-dual`: for generalized (hyper-) dual numbers. These data types are very important because we use them to be able to compute partial, higher-order derivatives of the Helmholtz energy without needing to implement them analytically.
- `ndarray`: for multidimensional arrays. We use these when mathematical operations are performed on arrays. They are a central data structure for `feos-dft`.
- `pyo3`: for the Python interface. All interfaces to Python are written in pure Rust using `PyO3`. It's awesome.

```{eval-rst}
.. toctree::
   :maxdepth: 1
   :hidden:

   equation_of_state
   state
```