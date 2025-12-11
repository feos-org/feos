# Introduction

Working with FeO$_\text{s}$ broadly requires these steps:

1. **Specifying parameters:** Create a `Parameters` or `GcParameters` object that contains parameters for the substance(s) of interest.
2. **Initializing a model:** That's either an `EquationOfState` or a `HelmholtzEnergyFunctional`.
3. **Specifying thermodynamic conditions:** That's either a `State` object, a `PhaseEquilibrium` object. There are several utility functions that build these objects in bulk, like a `PhaseDiagram`.
4. **Calculate properties:** Once the conditions are specified, you can calcuate properties. FeO$_\text{s}$ caches computations so repeated calls to the same method are very fast.

FeO$_\text{s}$ uses dimensioned units as in- and ouputs. Make sure you have the `si_units` package installed.

## Example

```python
import si_units as si
from feos.parameters import Identifier, PureRecord, Parameters
from feos import Contributions, EquationOfState, State

# Specify parameters for PC-SAFT
methane = PureRecord(
    identifier=Identifier(name="methane"),
    molarweight=16.043,
    m=1.0,
    sigma=3.7039,
    epsilon_k=150.03
)
parameters = Parameters.new_pure(methane)

# Initialize model: at this point parameters are checked for consistency with the model.
model = EquationOfState.pcsaft(parameters)

# Specify thermodynamic conditions and calculate property.
state = State(model, temperature=300 * si.KELVIN, pressure=5 * si.BAR)
entropy = state.molar_entropy(Contributions.Residual)
```
