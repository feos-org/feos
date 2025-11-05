# Equations of state

The key data structure in FeOs is the `EquationOfState` class that contains all implemented equations of state.
The `State` and `PhaseEquilibrium` objects are used to define thermodynamic conditions and -- once created -- can be used to compute properties.

## The `EquationOfState` class

### Residual Helmholtz energy models

```{eval-rst}
.. currentmodule:: feos

.. autosummary::
    EquationOfState.pcsaft
    EquationOfState.epcsaft
    EquationOfState.gc_pcsaft
    EquationOfState.peng_robinson
    EquationOfState.pets
    EquationOfState.uvtheory
    EquationOfState.saftvrmie
    EquationOfState.saftvrqmie
```

### Ideal gas models

```{eval-rst}

.. autosummary::
    EquationOfState.dippr
    EquationOfState.joback
```


### Example: Combine a DIPPR ideal gas model with PC-SAFT

```python
import feos

pc_saft_parameters = feos.Parameters.from_json(
    ['methane', 'ethane'], 
    'pc_saft_parameters.json'
)
dippr_parameters = feos.Parameters.from_json(
    ['methane', 'ethane'], 
    'dippr_parameters.json'
)
eos = feos.EquationOfState.pcsaft(pc_saft_parameters).dippr(dippr_parameters)
```

### Example: Combine the ideal gas model of Joback & Reid with PC-SAFT

```python
import feos

pc_saft_parameters = feos.Parameters.from_json_smiles(
    ['CCC', 'CCCC'],
    'smarts.json',
    'pcsaft_group_parameters.json'
)
joback_parameters = feos.Parameters.from_json_smiles(
    ['CCC', 'CCCC'],
    'smarts.json',
    'joback_parameters.json'
)
eos = feos.EquationOfState.pcsaft(pc_saft_parameters).joback(joback_parameters)
```

### Models defined in Python

```{eval-rst}
.. currentmodule:: feos

.. autosummary::
    EquationOfState.python_residual
    EquationOfState.python_ideal_gas
```

## Data types

```{eval-rst}
.. currentmodule:: feos

.. autosummary::
    :toctree: generated/
    
    EquationOfState
    Contributions
    Verbosity
    State
    StateVec
    PhaseEquilibrium
    PhaseDiagram
```
