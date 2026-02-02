# Quick Start

This guide will get you up and running with FeOs in just a few minutes. We'll walk through the basic workflow of setting up parameters, creating an equation of state, and calculating thermodynamic properties.

## Basic Workflow

The typical FeOs workflow follows these steps:

1. **Define parameters** for your chemical components
2. **Create an equation of state** using those parameters  
3. **Define thermodynamic states** at specific conditions
4. **Calculate properties** for those states

Let's walk through each step with a concrete example.

## Step 1: Import FeOs

Start by importing the main components:

```python
from feos import EquationOfState, State
from feos.parameters import PureRecord, Identifier, Parameters
```

## Step 2: Define Component Parameters

Create parameters for a pure component. Let's use methanol as an example:

```python
# PC-SAFT parameters for methanol (Gross & Sadowski, 2002)
methanol_record = PureRecord(
    identifier=Identifier(name="methanol", cas="67-56-1"),
    molarweight=32.04,  # g/mol
    m=1.5255,           # segment number
    sigma=3.23,         # segment diameter [Å]
    epsilon_k=188.9,    # dispersion energy [K]
    kappa_ab=0.035176,  # association volume
    epsilon_k_ab=2899.5, # association energy [K]
    na=1,               # number of association sites A
    nb=1,               # number of association sites B
)

# Create parameters object
parameters = Parameters.new_pure(methanol_record)
print(f"Parameters created for: {methanol_record.identifier.name}")
```

## Step 3: Create Equation of State

Use the parameters to create a PC-SAFT equation of state:

```python
# Create PC-SAFT equation of state
eos = EquationOfState.pcsaft(parameters)
print("PC-SAFT equation of state created")
```

## Step 4: Calculate Properties

Now you can calculate thermodynamic properties at different conditions.

### Critical Point

```python
# Calculate critical point
critical_point = State.critical_point_pure(eos)

print(f"Critical temperature: {critical_point.temperature}")
print(f"Critical pressure: {critical_point.pressure()}")
print(f"Critical density: {critical_point.density}")
```

### Properties at Specific Conditions

```python
# Create state at room temperature and atmospheric pressure
T = 298.15  # K
p = 101325  # Pa
room_temp_state = State(eos, temperature=T, pressure=p)

print(f"Density at {T} K, {p} Pa: {room_temp_state.density}")
print(f"Molar volume: {room_temp_state.molar_volume()}")
print(f"Compressibility factor: {room_temp_state.compressibility()}")
```

### Vapor Pressure

```python
# Calculate vapor pressure at room temperature
vapor_pressure = State.vapor_pressure(eos, T)
print(f"Vapor pressure at {T} K: {vapor_pressure}")

# Or calculate boiling temperature at atmospheric pressure
boiling_temp = State.boiling_temperature(eos, p)
print(f"Boiling temperature at {p} Pa: {boiling_temp}")
```

## Complete Example

Here's the complete code from this quick start:

```python
from feos import EquationOfState, State
from feos.parameters import PureRecord, Identifier, Parameters

# Define methanol parameters
methanol_record = PureRecord(
    identifier=Identifier(name="methanol", cas="67-56-1"),
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
parameters = Parameters.new_pure(methanol_record)
eos = EquationOfState.pcsaft(parameters)

# Calculate critical point
critical_point = State.critical_point_pure(eos)
print(f"Critical point: T = {critical_point.temperature}, p = {critical_point.pressure()}")

# Properties at room conditions
T = 298.15  # K
p = 101325  # Pa
state = State(eos, temperature=T, pressure=p)
print(f"Density at room conditions: {state.density}")

# Vapor pressure
pv = State.vapor_pressure(eos, T)
print(f"Vapor pressure at {T} K: {pv}")
```

## Working with Units

FeOs uses the `si-units` package for unit handling. All quantities are returned as `SIObject` instances:

```python
# Temperature and pressure are unitless in SI base units (K and Pa)
T = 298.15  # K
p = 101325  # Pa

state = State(eos, temperature=T, pressure=p)

# Properties are returned with units
density = state.density
print(f"Density: {density}")           # Will show value with units
print(f"Density value: {density.value}")  # Just the numerical value
print(f"Density units: {density.units}")  # Just the units
```

## Different Equation of State Models

FeOs supports multiple equation of state models. Here's how to use different ones:

```python
# PC-SAFT (as shown above)
eos_pcsaft = EquationOfState.pcsaft(parameters)

# If you have PETS parameters
# eos_pets = EquationOfState.pets(pets_parameters)

# If you have SAFT-VR Mie parameters  
# eos_saftvrmie = EquationOfState.saftvrmie(saftvrmie_parameters)
```

## Next Steps

Now that you understand the basics, explore:

- **[Core Concepts](concepts.md)** - Deeper understanding of FeOs components
- **[Working with States](../user-guide/working-with-states.md)** - More advanced state calculations
- **[Parameters](../user-guide/parameters.md)** - Loading parameters from JSON files
- **[Examples](../examples/basic-properties.md)** - More practical examples

## Common Issues

- **Unit confusion**: Remember that temperature is in Kelvin and pressure in Pascal
- **Parameter units**: Molecular parameters have specific units (see parameter documentation)
- **Association**: Not all components have association sites (set `na=0, nb=0` for non-associating)
- **Convergence**: Some calculations may not converge for extreme conditions