# `feos.eos`

The `eos` module contains the `EquationOfState` object that contains all implemented equations of state.
The `State` and `PhaseEquilibrium` objects are used to define thermodynamic conditions and -- once created -- can be used to compute properties.

If you want to adjust parameters of a model to experimental data you can use classes and utilities from the `estimator` module.

## `EquationOfState`

```{eval-rst}
.. currentmodule:: feos.eos

.. autosummary::
    :toctree: generated/

    EquationOfState
    EquationOfState.pcsaft
    EquationOfState.gc_pcsaft
    EquationOfState.peng_robinson
    EquationOfState.pets
    EquationOfState.python_residual
    EquationOfState.python_ideal_gas
    EquationOfState.uvtheory
    EquationOfState.saftvrqmie
```

### Models defined in Python

```{eval-rst}
.. currentmodule:: feos.eos

.. autosummary::
    :toctree: generated/

    EquationOfState.python_residual
    EquationOfState.python_ideal_gas
```

## Other data types

```{eval-rst}
.. currentmodule:: feos.eos

.. autosummary::
    :toctree: generated/

    Contributions
    Verbosity
    State
    StateVec
    PhaseEquilibrium
    PhaseDiagram
```

## The `estimator` module

### Import 

```python
from feos.eos.estimator import Estimator, DataSet, Loss, Phase
```

```{eval-rst}
.. currentmodule:: feos.eos.estimator

.. autosummary::
    :toctree: generated/

    Estimator
    DataSet
    Loss
    Phase
```
