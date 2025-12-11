# FeOs Python Documentation

Welcome to the documentation for the **FeOs Python package** - Python bindings for the FeOs framework for equations of state and classical density functional theory.

## What is FeOs?

FeOs is a comprehensive framework that provides Rust implementations of different equation of state and Helmholtz energy functional models with corresponding Python bindings. The Python package offers a user-friendly interface to perform:

- **Thermodynamic property calculations** for pure components and mixtures
- **Phase equilibrium calculations** including critical points and phase diagrams  
- **Classical density functional theory** calculations for inhomogeneous systems
- **Parameter estimation** for equation of state models

## Key Features

- **Multiple EoS models**: PC-SAFT, EPC-SAFT, SAFT-VR Mie, PETS, UV Theory, and more
- **Group contribution methods**: GC-PC-SAFT for parameter prediction
- **DFT calculations**: Adsorption isotherms, surface tension, pair correlation functions
- **Parameter management**: JSON-based parameter databases and estimation tools
- **High performance**: Rust backend with automatic differentiation
- **Type safety**: Comprehensive type hints for excellent IDE support

## Quick Example

Get started with a simple thermodynamic calculation:

```python
import feos

# Define PC-SAFT parameters for methanol
record = feos.PureRecord(
    feos.Identifier(name="methanol"),
    molarweight=32.04,
    m=1.5255,
    sigma=3.23,
    epsilon_k=188.9,
    kappa_ab=0.035176,
    epsilon_k_ab=2899.5,
    na=1,
    nb=1,
)

# Create equation of state
parameters = feos.Parameters.new_pure(record)
eos = feos.EquationOfState.pcsaft(parameters)

# Calculate critical point
critical_point = feos.State.critical_point_pure(eos)
print(f"Critical temperature: {critical_point.temperature}")
print(f"Critical pressure: {critical_point.pressure()}")
```

## Available Models

The following models are currently implemented:

| Model | Description | EoS | DFT |
|-------|-------------|-----|-----|
| `pcsaft` | Perturbed-chain statistical associating fluid theory | ✓ | ✓ |
| `epcsaft` | Electrolyte PC-SAFT | ✓ | |
| `gc-pcsaft` | Group contribution PC-SAFT | ✓ | ✓ |
| `pets` | Perturbed truncated and shifted Lennard-Jones | ✓ | ✓ |
| `uvtheory` | Equation of state for Mie fluids | ✓ | |
| `saftvrqmie` | SAFT-VR for quantum fluids | ✓ | ✓ |
| `saftvrmie` | SAFT-VR for variable range Mie interactions | ✓ | |

## Getting Started

Ready to dive in? Here's how to get started:

1. **[Installation](getting-started/installation.md)** - Install the package via pip or build from source
2. **[Quick Start](getting-started/quickstart.md)** - Your first calculations with FeOs
3. **[Core Concepts](getting-started/concepts.md)** - Understand the main components

## Documentation Structure

- **[Getting Started](getting-started/installation.md)** - Installation and basic usage
- **[User Guide](user-guide/working-with-states.md)** - Detailed guides for different functionalities
- **[Examples](examples/basic-properties.md)** - Practical code examples for common tasks
- **[API Reference](api/feos.md)** - Complete API documentation

## Need Help?

- Browse the **[Examples](examples/basic-properties.md)** for code you can copy and adapt
- Check the **[API Reference](api/feos.md)** for detailed method documentation
- Visit the [main FeOs repository](https://github.com/feos-org/feos) for issues and discussions

## Citation

If you use FeOs in your research, please cite our publication:

```bibtex
@article{rehner2023feos,
  author = {Rehner, Philipp and Bauer, Gernot and Gross, Joachim},
  title = {FeOs: An Open-Source Framework for Equations of State and Classical Density Functional Theory},
  journal = {Industrial \& Engineering Chemistry Research},
  volume = {62},
  number = {12},
  pages = {5347-5357},
  year = {2023},
}
```
