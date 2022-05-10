---
hide-toc: true
---

# Welcome to {math}`\text{FeO}_\text{s}`

{math}`\text{FeO}_\text{s}` is a **framework** for **thermodynamic equations of state** (EoS) and **classical density functional theory** (DFT).
It is written in **Rust** with a **Python** interface.

## Usage

`````{tab-set}
````{tab-item} Python
```python
from feos.eos import EquationOfState, State
from feos.pcsaft import PcSaftParameters

# Build an equation of state
parameters = PcSaftParameters.from_json(['methanol'], 'parameters.json')
eos = EquationOfState.pcsaft(parameters)

# Define thermodynamic conditions
critical_point = State.critical_point(eos)

# Compute properties
p = critical_point.pressure()
t = critical_point.temperature
print(f'Critical point for methanol: T={t}, p={p}.')
```
```terminal
Critical point for methanol: T=531.5 K, p=10.7 MPa.
```
````

````{tab-item} Rust
```rust
// some imports omitted
use feos_core::{State, Contributions};
use feos_pcsaft::{PcSaft, PcSaftParameters};

// Build an equation of state
let parameters = PcSaftParameters.from_json(vec!["methanol"], "parameters.json")?;
let eos = Rc::new(PcSaft::new(Rc::new(parameters)));

// Define thermodynamic conditions
let critical_point = State::critical_point(&eos, None, None, Default::default())?;

// Compute properties
let p = critical_point.pressure(Contributions::Total);
let t = critical_point.temperature;
println!("Critical point for methanol: T={}, p={}.", t, p);
```
```terminal
Critical point for methanol: T=531.5 K, p=10.7 MPa.
```
````
`````

## Features

```{dropdown} Equations of State
:open:

- thermodynamic **properties**
- **phase equilibria** for pure substances and mixtures
- **critical point** calculations for pure substances and mixtures
- **dynamic properties** (entropy scaling)
- **stability analysis**
---
**Implemented equations of state**
- PC-SAFT (incl. group contribution method)
- uv-Theory
- PeTS
```

```{dropdown} Density Functional Theory
:open:

- **interfacial** properties,
- properties in **pores** and at **walls**,
- **adsorption isotherms**,
- **solvation free energies**,
- different **dimensions** and **coordinate systems**
```

```{dropdown} Extensibility / Usability
:open:

- Helmholtz energy uses **generalized (hyper-) dual numbers** - **no analytical derivatives are needed**.
- Interfaces use **dimensioned quantities**.
- Python bindings are written in Rust - **robust type checking** and **error handling**.
```


## Getting started

- Learn how to [install the code](installation).
- Browse the [python examples](examples/index).
- Delve into the [python API](api/index).
- Interested in extending the library? Check out our [Rust guide](rustguide/index)!

```{toctree}
:hidden:

installation
help_and_feedback
```


```{toctree}
:caption: Python
:hidden:

api/index
examples/index
recipes/index
```

```{toctree}
:caption: Rust
:hidden:

rustguide/index
rust_api
```