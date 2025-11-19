# Core Concepts

Understanding the core concepts of FeOs will help you use the library effectively. This page explains the main components and how they work together.

## Overview

FeOs is built around a few key concepts:

- **Parameters** define the molecular properties of chemical components
- **Equations of State (EoS)** describe how those components behave thermodynamically
- **States** represent specific thermodynamic conditions
- **Phase Equilibria** calculations find coexisting phases
- **DFT profiles** describe inhomogeneous fluid systems

## Parameters

Parameters contain the molecular information needed for thermodynamic calculations.

### Pure Component Parameters

Parameters for a single component are stored in a `PureRecord`:

```python
from feos.parameters import PureRecord, Identifier

# PC-SAFT parameters for water
water = PureRecord(
    identifier=Identifier(name="water", cas="7732-18-5"),
    molarweight=18.015,
    m=1.2047,      # segment number
    sigma=2.7927,  # segment diameter [Å]  
    epsilon_k=353.944, # dispersion energy [K]
    kappa_ab=0.0451,   # association volume
    epsilon_k_ab=2500.7, # association energy [K]
    na=1,          # association sites of type A
    nb=1,          # association sites of type B
)
```

### Parameter Meanings

Different equation of state models use different parameters:

**PC-SAFT parameters:**
- `m`: Number of segments in the chain
- `sigma`: Segment diameter [Å]
- `epsilon_k`: Dispersion energy [K]
- `kappa_ab`: Association volume parameter
- `epsilon_k_ab`: Association energy [K]
- `na`, `nb`: Number of association sites

**PETS parameters:**
- `sigma`: Hard sphere diameter [Å]
- `epsilon_k`: Dispersion energy [K]

### Binary Interaction Parameters

For mixtures, you may need binary interaction parameters:

```python
from feos.parameters import BinaryRecord

# Binary interaction for water-ethanol
binary = BinaryRecord(
    id1=Identifier(name="water"),
    id2=Identifier(name="ethanol"),
    k_ij=0.02,  # Binary interaction parameter
)
```

## Equations of State

An equation of state describes the relationship between pressure, volume, temperature, and composition.

### Creating an EoS

```python
from feos import EquationOfState
from feos.parameters import Parameters

# Create parameters object
parameters = Parameters.new_pure(water)

# Create PC-SAFT equation of state
eos = EquationOfState.pcsaft(parameters)
```

### Available Models

FeOs supports several equation of state models:

- **PC-SAFT**: Perturbed-chain statistical associating fluid theory
- **EPC-SAFT**: Electrolyte PC-SAFT (for ionic systems)
- **GC-PC-SAFT**: Group contribution PC-SAFT
- **PETS**: Perturbed truncated and shifted Lennard-Jones
- **SAFT-VR Mie**: Variable range SAFT with Mie potential
- **UV Theory**: For Mie fluids

Each model has its own parameter requirements and applicability range.

## States

A `State` represents a thermodynamic condition with known temperature, pressure, density, or other properties.

### Creating States

States can be created from different combinations of properties:

```python
from feos import State

# From temperature and pressure
state1 = State(eos, temperature=298.15, pressure=101325)

# From temperature and density  
state2 = State(eos, temperature=298.15, density=1000)

# From temperature and molar volume
state3 = State(eos, temperature=298.15, volume=0.018)
```

### State Properties

Once you have a state, you can calculate many properties:

```python
# Basic properties
print(f"Pressure: {state.pressure()}")
print(f"Density: {state.density}")
print(f"Molar volume: {state.molar_volume()}")

# Derived properties
print(f"Compressibility: {state.compressibility()}")
print(f"Speed of sound: {state.speed_of_sound()}")
print(f"Heat capacity (cp): {state.molar_isobaric_heat_capacity()}")
print(f"Heat capacity (cv): {state.molar_isochoric_heat_capacity()}")

# Chemical potential and fugacity
print(f"Chemical potential: {state.chemical_potential()}")
print(f"Fugacity coefficient: {state.ln_phi()}")
```

## Phase Equilibria

Phase equilibria calculations find conditions where multiple phases coexist.

### Vapor-Liquid Equilibrium

```python
from feos import PhaseEquilibrium

# Calculate vapor pressure at given temperature
vle = PhaseEquilibrium.pure(eos, temperature=298.15, pressure=None)
print(f"Vapor pressure: {vle.vapor().pressure()}")
print(f"Liquid density: {vle.liquid().density}")
print(f"Vapor density: {vle.vapor().density}")
```

### Critical Points

```python
# Critical point of pure component
critical = State.critical_point_pure(eos)
print(f"Critical temperature: {critical.temperature}")
print(f"Critical pressure: {critical.pressure()}")
print(f"Critical density: {critical.density}")
```

## Mixtures

For multicomponent systems, you need parameters for all components plus any binary interactions:

```python
# Multiple pure components
water_record = PureRecord(...)  # water parameters
ethanol_record = PureRecord(...)  # ethanol parameters

# Binary interaction
binary_record = BinaryRecord(...)

# Create mixture parameters
mixture_params = Parameters.from_records(
    pure_records=[water_record, ethanol_record],
    binary_records=[binary_record]
)

# Create mixture EoS
mixture_eos = EquationOfState.pcsaft(mixture_params)

# Create mixture state (need mole fractions)
mixture_state = State(
    mixture_eos,
    temperature=298.15,
    pressure=101325,
    molefracs=[0.5, 0.5]  # 50% water, 50% ethanol
)
```

## Density Functional Theory (DFT)

DFT calculations describe inhomogeneous fluid systems like interfaces and confined fluids.

### Helmholtz Energy Functionals

```python
from feos.dft import HelmholtzEnergyFunctional

# Create functional (requires DFT-capable EoS)
functional = HelmholtzEnergyFunctional.pcsaft(parameters)
```

### Density Profiles

DFT calculates density profiles in various geometries:

```python
from feos.dft import PlanarInterface

# Vapor-liquid interface
vle = PhaseEquilibrium.pure(eos, temperature=400, pressure=None)
interface = PlanarInterface.from_tanh(vle, n_grid=200, l_grid=10e-9)
interface.solve()

print(f"Surface tension: {interface.surface_tension()}")
```

## Units and Quantities

FeOs uses the `si-units` package for physical quantities:

```python
# All inputs are in SI base units
T = 298.15  # Kelvin
p = 101325  # Pascal

# Outputs are SIObject instances with units
density = state.density
print(f"Value: {density.value}")  # Numerical value
print(f"Units: {density.units}")  # Units string
print(f"Full: {density}")         # Value with units
```

## Error Handling

FeOs calculations can fail for various reasons:

```python
try:
    state = State(eos, temperature=298.15, pressure=1e10)  # Very high pressure
except RuntimeError as e:
    print(f"Calculation failed: {e}")
```

Common failure modes:
- **Convergence failure**: Iteration didn't converge
- **Invalid conditions**: Unphysical temperature/pressure
- **Phase detection**: Multiple solutions exist

## Next Steps

Now that you understand the core concepts:

- **[Working with States](../user-guide/working-with-states.md)** - Advanced state calculations
- **[Parameters](../user-guide/parameters.md)** - Parameter management and databases
- **[Examples](../examples/basic-properties.md)** - Practical code examples