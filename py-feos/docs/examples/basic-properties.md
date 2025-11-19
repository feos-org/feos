# Basic Properties

This example shows how to calculate basic thermodynamic properties for pure components.

## Pure Component Properties

Let's calculate various properties for methane using PC-SAFT:

```python
from feos import EquationOfState, State
from feos.parameters import PureRecord, Identifier, Parameters

# PC-SAFT parameters for methane
methane = PureRecord(
    identifier=Identifier(name="methane", cas="74-82-8"),
    molarweight=16.04,
    m=1.0,
    sigma=3.7039,
    epsilon_k=150.03,
)

# Create equation of state
parameters = Parameters.new_pure(methane)
eos = EquationOfState.pcsaft(parameters)

# Define conditions
T = 298.15  # K (room temperature)
p = 101325  # Pa (1 atm)

# Create state
state = State(eos, temperature=T, pressure=p)

print("=== Basic Properties ===")
print(f"Temperature: {state.temperature} K")
print(f"Pressure: {state.pressure()} Pa")
print(f"Density: {state.density}")
print(f"Molar volume: {state.molar_volume()}")
print(f"Compressibility factor: {state.compressibility()}")
```

## Derived Properties

Calculate heat capacities, speed of sound, and other derived properties:

```python
print("\n=== Derived Properties ===")
print(f"Isobaric heat capacity (Cp): {state.molar_isobaric_heat_capacity()}")
print(f"Isochoric heat capacity (Cv): {state.molar_isochoric_heat_capacity()}")
print(f"Heat capacity ratio (γ): {state.isentropic_compressibility()}")
print(f"Speed of sound: {state.speed_of_sound()}")
print(f"Joule-Thomson coefficient: {state.joule_thomson()}")
```

## Chemical Potential and Fugacity

For phase equilibrium calculations, chemical potential and fugacity are important:

```python
print("\n=== Chemical Properties ===")
print(f"Chemical potential: {state.chemical_potential()}")
print(f"Fugacity coefficient (ln φ): {state.ln_phi()}")
print(f"Fugacity: {state.fugacity()}")
```

## Properties at Different Conditions

Calculate properties at various temperatures and pressures:

```python
print("\n=== Properties at Different Conditions ===")

# Temperature range at constant pressure
temperatures = [250, 300, 350, 400]  # K
pressure = 101325  # Pa

print("T [K]\tDensity [kg/m³]\tCp [J/mol/K]")
print("-" * 40)

for T in temperatures:
    state = State(eos, temperature=T, pressure=pressure)
    density = state.density.value  # Get numerical value
    cp = state.molar_isobaric_heat_capacity().value
    print(f"{T}\t{density:.2f}\t\t{cp:.2f}")
```

## Critical Properties

Calculate critical point properties:

```python
print("\n=== Critical Properties ===")

# Calculate critical point
critical = State.critical_point_pure(eos)

print(f"Critical temperature: {critical.temperature} K")
print(f"Critical pressure: {critical.pressure()} Pa")
print(f"Critical density: {critical.density}")
print(f"Critical compressibility: {critical.compressibility()}")
```

## Vapor Pressure

Calculate vapor pressure at different temperatures:

```python
print("\n=== Vapor Pressure ===")

temperatures = [200, 250, 300, 350]  # K

print("T [K]\tP_sat [Pa]")
print("-" * 20)

for T in temperatures:
    try:
        p_sat = State.vapor_pressure(eos, T)
        print(f"{T}\t{p_sat.value:.0f}")
    except RuntimeError:
        print(f"{T}\tN/A (above critical)")
```

## Different Models

Compare results from different equation of state models:

```python
print("\n=== Model Comparison ===")

# Create different EoS for the same component
eos_pcsaft = EquationOfState.pcsaft(parameters)

# For comparison, we would need PETS parameters:
# pets_params = PureRecord(
#     identifier=Identifier(name="methane"),
#     molarweight=16.04,
#     sigma=3.7,  # Different parameter meaning in PETS
#     epsilon_k=150,
# )
# eos_pets = EquationOfState.pets(Parameters.new_pure(pets_params))

# Calculate density at same conditions
T, p = 298.15, 101325
state_pcsaft = State(eos_pcsaft, temperature=T, pressure=p)

print(f"PC-SAFT density: {state_pcsaft.density}")
# print(f"PETS density: {state_pets.density}")
```

## Working with Units

Understanding how to work with units in FeOs:

```python
print("\n=== Working with Units ===")

state = State(eos, temperature=298.15, pressure=101325)
density = state.density

# Access different aspects of the quantity
print(f"Full quantity: {density}")
print(f"Numerical value: {density.value}")
print(f"Units: {density.units}")

# Convert to different units (if needed for calculations)
density_value = density.value  # kg/m³ in SI units
print(f"Density in g/cm³: {density_value / 1000:.6f}")
```

## Error Handling

Handle common calculation errors:

```python
print("\n=== Error Handling ===")

try:
    # Try an extreme condition that might fail
    extreme_state = State(eos, temperature=1000, pressure=1e8)  # Very high T and p
    print(f"Extreme condition density: {extreme_state.density}")
except RuntimeError as e:
    print(f"Calculation failed: {e}")

try:
    # Try vapor pressure above critical temperature
    high_temp_vp = State.vapor_pressure(eos, 500)  # Above critical
    print(f"Vapor pressure: {high_temp_vp}")
except RuntimeError as e:
    print(f"Vapor pressure calculation failed: {e}")
```

## Complete Example Script

Here's a complete script you can run:

```python
#!/usr/bin/env python3
"""
Basic thermodynamic property calculations with FeOs
"""

from feos import EquationOfState, State
from feos.parameters import PureRecord, Identifier, Parameters

def main():
    # Define component
    methane = PureRecord(
        identifier=Identifier(name="methane", cas="74-82-8"),
        molarweight=16.04,
        m=1.0,
        sigma=3.7039,
        epsilon_k=150.03,
    )
    
    # Create EoS
    parameters = Parameters.new_pure(methane)
    eos = EquationOfState.pcsaft(parameters)
    
    # Calculate properties at standard conditions
    state = State(eos, temperature=298.15, pressure=101325)
    
    print("Methane properties at 298.15 K, 1 atm:")
    print(f"  Density: {state.density}")
    print(f"  Compressibility: {state.compressibility():.4f}")
    print(f"  Heat capacity (Cp): {state.molar_isobaric_heat_capacity()}")
    
    # Critical point
    critical = State.critical_point_pure(eos)
    print(f"\nCritical point:")
    print(f"  Temperature: {critical.temperature}")
    print(f"  Pressure: {critical.pressure()}")

if __name__ == "__main__":
    main()
```

This example demonstrates the fundamental property calculations available in FeOs. Next, explore [pure component VLE](pure-component-vle.md) for vapor-liquid equilibrium calculations.