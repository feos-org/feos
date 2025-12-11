# Working with States

States are central to thermodynamic calculations in FeOs. This guide explains how to create states, calculate properties, and handle different scenarios.

## State Creation

States can be created from different combinations of intensive and extensive properties.

### From Temperature and Pressure

The most common way to create a state:

```python
from feos import EquationOfState, State
from feos.parameters import PureRecord, Identifier, Parameters

# Set up equation of state (see previous examples)
# ... eos creation code ...

# Create state from T and p
state = State(eos, temperature=298.15, pressure=101325)
```

### From Temperature and Density

When you know the density:

```python
# Density in kg/m³
state = State(eos, temperature=298.15, density=1000)
```

### From Temperature and Volume

Specify molar volume:

```python
# Molar volume in m³/mol
state = State(eos, temperature=298.15, volume=0.024)
```

### Mixed Specifications

You can combine different property specifications:

```python
# Temperature and enthalpy
state = State(eos, temperature=298.15, molar_enthalpy=1000)

# Temperature and entropy  
state = State(eos, temperature=298.15, molar_entropy=100)

# Pressure and enthalpy (will iterate to find temperature)
state = State(eos, pressure=101325, molar_enthalpy=1000)
```

## State Properties

Once you have a state, calculate various thermodynamic properties.

### Basic Properties

```python
# Primary state variables
print(f"Temperature: {state.temperature}")
print(f"Pressure: {state.pressure()}")
print(f"Density: {state.density}")
print(f"Molar volume: {state.molar_volume()}")

# Derived properties
print(f"Compressibility factor: {state.compressibility()}")
print(f"Mass density: {state.mass_density()}")
```

### Energy Properties

```python
# Internal energy and enthalpy
print(f"Molar internal energy: {state.molar_internal_energy()}")
print(f"Specific internal energy: {state.specific_internal_energy()}")
print(f"Molar enthalpy: {state.molar_enthalpy()}")
print(f"Specific enthalpy: {state.specific_enthalpy()}")

# Entropy
print(f"Molar entropy: {state.molar_entropy()}")
print(f"Specific entropy: {state.specific_entropy()}")

# Free energies
print(f"Molar Helmholtz energy: {state.molar_helmholtz_energy()}")
print(f"Molar Gibbs energy: {state.molar_gibbs_energy()}")
```

### Heat Capacities

```python
# Heat capacities
print(f"Cp (isobaric): {state.molar_isobaric_heat_capacity()}")
print(f"Cv (isochoric): {state.molar_isochoric_heat_capacity()}")
print(f"Specific Cp: {state.specific_isobaric_heat_capacity()}")
print(f"Specific Cv: {state.specific_isochoric_heat_capacity()}")

# Heat capacity ratio
gamma = state.molar_isobaric_heat_capacity() / state.molar_isochoric_heat_capacity()
print(f"Heat capacity ratio (γ): {gamma}")
```

### Transport Properties

```python
# Speed of sound
print(f"Speed of sound: {state.speed_of_sound()}")

# Compressibilities
print(f"Isothermal compressibility: {state.isothermal_compressibility()}")
print(f"Isentropic compressibility: {state.isentropic_compressibility()}")

# Expansion coefficient
print(f"Thermal expansion: {state.thermal_expansion_coefficient()}")

# Joule-Thomson coefficient
print(f"Joule-Thomson coefficient: {state.joule_thomson()}")
```

## Mixture States

For multicomponent systems, you need to specify composition.

### Binary Mixture

```python
# Create binary mixture parameters (see parameter examples)
# ... mixture parameter setup ...

# Specify composition with mole fractions
mixture_state = State(
    mixture_eos,
    temperature=298.15,
    pressure=101325,
    molefracs=[0.3, 0.7]  # 30% component 1, 70% component 2
)
```

### Partial Properties

For mixtures, you can calculate partial molar properties:

```python
# Partial molar volumes
partial_volumes = mixture_state.partial_molar_volume()
print(f"Partial molar volumes: {partial_volumes}")

# Chemical potentials
chemical_potentials = mixture_state.chemical_potential()
print(f"Chemical potentials: {chemical_potentials}")

# Fugacity coefficients
ln_phi = mixture_state.ln_phi()
print(f"ln(φ): {ln_phi}")
```

## Special States

### Critical Points

```python
# Pure component critical point
critical = State.critical_point_pure(eos)
print(f"Critical T: {critical.temperature}")
print(f"Critical p: {critical.pressure()}")
print(f"Critical ρ: {critical.density}")

# For mixtures, critical points depend on composition
# mixture_critical = State.critical_point(mixture_eos, molefracs=[0.5, 0.5])
```

### Saturation States

```python
# Vapor pressure calculation
T = 298.15
p_sat = State.vapor_pressure(eos, T)
print(f"Vapor pressure at {T} K: {p_sat}")

# Boiling temperature
p = 101325  # Pa
T_boil = State.boiling_temperature(eos, p)
print(f"Boiling temperature at {p} Pa: {T_boil}")

# Saturation density
liquid_density = State.saturated_liquid_density(eos, T)
vapor_density = State.saturated_vapor_density(eos, T)
print(f"Liquid density: {liquid_density}")
print(f"Vapor density: {vapor_density}")
```

## State Iterations

Some state specifications require iterative solution.

### Density Initialization

When creating states, you can guide the density iteration:

```python
# Specify initial guess for density iteration
state = State(
    eos,
    temperature=298.15,
    pressure=101325,
    density_initialization="liquid"  # or "vapor"
)

# Or provide specific initial density
state = State(
    eos,
    temperature=298.15,
    pressure=101325,
    density_initialization=1000  # kg/m³
)
```

### Temperature Iterations

For some property combinations, temperature is iterated:

```python
# Provide initial temperature guess
state = State(
    eos,
    pressure=101325,
    molar_enthalpy=1000,
    initial_temperature=300  # K
)
```

## Error Handling and Troubleshooting

### Common Failure Modes

```python
try:
    # This might fail if conditions are too extreme
    extreme_state = State(eos, temperature=10000, pressure=1e10)
except RuntimeError as e:
    print(f"State creation failed: {e}")

try:
    # Vapor pressure above critical temperature
    high_temp_vp = State.vapor_pressure(eos, 1000)
except RuntimeError as e:
    print(f"No vapor pressure above critical: {e}")
```

### Convergence Issues

```python
# Adjust iteration parameters for difficult cases
state = State(
    eos,
    temperature=298.15,
    pressure=101325,
    max_iter=100,  # Increase iterations
    tol=1e-12,     # Tighter tolerance
    verbosity=feos.Verbosity.Iter  # Show iteration progress
)
```

### Phase Detection

```python
# When multiple solutions exist, specify which phase
liquid_state = State(
    eos,
    temperature=300,
    pressure=p_sat,
    density_initialization="liquid"
)

vapor_state = State(
    eos,
    temperature=300,
    pressure=p_sat,
    density_initialization="vapor"
)
```

## Performance Considerations

### Batch Calculations

For many calculations at different conditions:

```python
# More efficient than creating many individual states
temperatures = [250, 275, 300, 325, 350]
pressures = [1e5, 2e5, 3e5, 4e5, 5e5]

results = []
for T in temperatures:
    for p in pressures:
        try:
            state = State(eos, temperature=T, pressure=p)
            results.append({
                'T': T,
                'p': p,
                'density': state.density.value,
                'cp': state.molar_isobaric_heat_capacity().value
            })
        except RuntimeError:
            # Skip failed calculations
            continue

print(f"Calculated {len(results)} state points")
```

### Property Caching

Properties are calculated on demand and cached:

```python
# First call calculates and caches
cp1 = state.molar_isobaric_heat_capacity()

# Second call returns cached value (faster)
cp2 = state.molar_isobaric_heat_capacity()
```

This guide covers the fundamentals of working with states. Next, explore [phase equilibria](phase-equilibria.md) for vapor-liquid equilibrium calculations.