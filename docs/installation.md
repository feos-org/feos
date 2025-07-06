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
In Rust, the `feos-dft` crate and each equation of state or DFT functional is a separate, optional module behind a feature flag.
To use {math}`\text{FeO}_\text{s}` with all models (including Python bindings), use the `all_models` feature:

```toml
[dependencies]
feos-core = "0.4"
feos = { version="0.4", features = ["all_models"] }
quantity = "0.6"
```

To access the generic implementations for properties and phase equilibria, the `feos-core` crate is also included as dependency. Finally, {math}`\text{FeO}_\text{s}` makes extensive use of SI units provided by the `quantity` crate.

In the following example we use only the PC-SAFT equation of state and Helmholtz energy functionals (without Python bindings):

```toml
[dependencies]
feos-core = "0.4"
feos-dft = "0.4"
feos = { version="0.4", features = ["dft", "pcsaft"] }
quantity = "0.6"
```

To access generic DFT functionalities like interfaces and adsorption isotherms, the `feos-dft` crate is added to the dependencies.
````
`````
