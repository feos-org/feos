# `feos.uvtheory`

Utilities to build `UVParameters`.

## Example

```python
from feos.eos import EquationOfState
from feos.uvtheory import UVParameters, Perturbation

# binary system
parameters = UVParameters.from_lists(
   rep=[12.0, 15.0],
   att=[6.0]*2,
   sigma=[3.5, 3.7],
   epsilon_k=[120.0, 150.0]
)
uvtheory = EquationOfState.uvtheory(
   parameters,
   perturbation=Perturbation.WeeksChandlerAndersen
)
```

## Data types

```{eval-rst}
.. currentmodule:: feos.uvtheory

.. autosummary::
    :toctree: generated/

    Perturbation
    Identifier
    ChemicalRecord
    PureRecord
    BinaryRecord
    UVRecord
    UVParameters
```