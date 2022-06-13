# `feos.eos`

The `eos` module contains the `EquationOfState` object that contains all implemented equations of state.
The `State` and `PhaseEquilibrium` objects are used to define thermodynamic conditions and -- once created -- can be used to compute properties.

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
    EquationOfState.python
    EquationOfState.uvtheory
```

## Other data types

```{eval-rst}
.. currentmodule:: feos.eos

.. autosummary::
    :toctree: generated/

    Contributions
    Verbosity
    State
    PhaseEquilibrium
    PhaseDiagram
```