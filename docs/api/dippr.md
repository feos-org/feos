# `feos.dippr`

Ideal gas model based on DIPPR correlations.

## Example: Combine a DIPPR ideal gas model with PC-SAFT

```python
from feos.eos import EquationOfState
from feos.pcsaft import PcSaftParameters
from feos.dippr import Dippr

pc_saft_parameters = PcSaftParameters.from_json(
    ['methane', 'ethane'], 
    'pc_saft_parameters.json'
)
dippr = Dippr.from_json(
    ['methane', 'ethane'], 
    'dippr_parameters.json'
)
eos = EquationOfState.pcsaft(pc_saft_parameters).dippr(dippr)
```

## Data types

```{eval-rst}
.. currentmodule:: feos.dippr

.. autosummary::
    :toctree: generated/

    Identifier
    IdentifierOption
    DipprRecord
    PureRecord
    Dippr
```