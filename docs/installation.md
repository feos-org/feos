# Installation

`````{tab-set}
````{tab-item} Python
{math}`\text{FeO}_\text{s}` is available on `PyPI`. You can install it using `pip`:

```
pip install feos
```

If you have a Rust compiler installed and want to have the latest development version (github `main` branch), you can build and install the python wheel via

```
pip install git+https://github.com/feos-org/feos
```
````

````{tab-item} Rust
In Rust, DFT and each equation of state or Helmholtz energy functional is a separate, optional module behind a feature flag.
To use {math}`\text{FeO}_\text{s}` with all models, use the `all_models` feature:

```
cargo add feos --features all_models
```

In the following example we use only the PC-SAFT equation of state and Helmholtz energy functionals:

```
cargo add feos --features "dft pcsaft"
```

The generic implementations for properties and phase equilibria (`feos-core`) and for DFT (`feos-dft`) are re-exported as `feos::core` and `feos::dft`, so no additional dependencies are required.

{math}`\text{FeO}_\text{s}` makes extensive use of SI units provided by the `quantity` crate, which has to be added separately:

```
cargo add quantity
```

Make sure that the version of `quantity` matches the one used by `feos`. Otherwise, the units are not compatible.
````
`````
