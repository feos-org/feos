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
In Rust, each equation of state or DFT functional is a separate crate and is hosted on [crates.io](https://crates.io).
For example, if you want to use the PC-SAFT equation of state, add this to your `Cargo.toml`:

```toml
[dependencies]
feos-core = "0.1"
feos-dft = "0.1"
feos-pcsaft = "0.1"
quantity = "0.4"
```
````
`````
