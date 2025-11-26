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
import feos

# Build an equation of state
parameters = feos.Parameters.from_json(['methanol'], 'parameters.json')
eos = feos.EquationOfState.pcsaft(parameters)

# Define thermodynamic conditions
critical_point = feos.State.critical_point(eos)

# Compute properties
p = critical_point.pressure()
t = critical_point.temperature
print(f'Critical point for methanol: T={t}, p={p}.')
```
```output
Critical point for methanol: T=531.5 K, p=10.7 MPa.
```
````

````{tab-item} Rust
```rust
// some imports omitted
use feos::core::parameter::{IdentifierOption, Parameters};
use feos::core::{Contributions, State};
use feos::pcsaft::PcSaft;

// Build an equation of state
let parameters = Parameters::from_json(
    vec!["methanol"],
    "parameters.json",
    None,
    IdentifierOption::Name,
)?;
let eos = &PcSaft::new(parameters);

// Define thermodynamic conditions
let critical_point = State::critical_point(&eos, None, None, None, Default::default())?;

// Compute properties
let p = critical_point.pressure(Contributions::Total);
let t = critical_point.temperature;
println!("Critical point for methanol: T={}, p={}.", t, p);
```
```output
Critical point for methanol: T=531.5 K, p=10.7 MPa.
```
````
`````

## Getting started

- Learn how to [install the code](installation).
- Browse the [python tutorials](tutorials/index).
- Need something specific? May we have a [recipe](recipes/index) for that.
- Questions or comments? [We are happy to hear from you](help_and_feedback)!

## Want to learn more?
- Delve into the [python](api/index) or [Rust API](rust_api).
- Learn about the underlying theory in our [Theory guide](theory/eos/properties)!

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
- ePC-SAFT
- uv-Theory
- SAFT-VR-Mie and the extension to quantum fluids SAFT-VRQ-Mie
- PeTS
- Multiparameter Helmholtz energy equations of state for common pure components
```

```{dropdown} Density Functional Theory
:open:

- **interfacial** properties,
- properties in **nanopores** and at **walls**,
- **adsorption isotherms**,
- **solvation free energies**,
- different **dimensions** and **coordinate systems**
```

```{dropdown} Extensibility / Usability
:open:

- Helmholtz energy uses **generalized (hyper-) dual numbers** - no analytical derivatives are needed.
- Interfaces use **dimensioned quantities** - never accidentally mix molar and mass-specific properties.
- Python bindings are written in Rust - **robust type checking** and **error handling**.
```

```{toctree}
:hidden:

installation
help_and_feedback
```


```{toctree}
:caption: Python
:hidden:

tutorials/index
recipes/index
api/index
```

```{toctree}
:caption: Rust
:hidden:

rust_api
```

```{toctree}
:caption: Theory
:hidden:

theory/eos/index
theory/dft/index
theory/models/index
```