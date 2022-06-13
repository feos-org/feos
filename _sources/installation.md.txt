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
To use {math}`\text{FeO}_\text{s}` with all models (including Python bindings), use the `all` feature:

```toml
[dependencies]
feos = { version="0.2", features = ["all"] }
quantity = "0.5"
```

In the following example we use only the PC-SAFT equation of state and Helmholtz energy functionals (without Python bindings):

```toml
[dependencies]
feos = { version="0.2", features = ["dft", "pcsaft"] }
quantity = "0.5"
```
````
`````
