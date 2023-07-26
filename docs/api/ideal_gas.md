# `feos.ideal_gas`

Utilities to build parameters for ideal gas models.

## Example: Combine the ideal gas model of Joback & Reid with PC-SAFT

```python
from feos.eos import EquationOfState
from feos.pcsaft import PcSaftParameters
from feos.ideal_gas import JobackParameters

pc_saft_parameters = PcSaftParameters.from_json(
    ['methane', 'ethane'], 
    'pc_saft_parameters.json'
)
joback_parameters = JobackParameters.from_json(
    ['methane', 'ethane'], 
    'joback_parameters.json'
)
eos = EquationOfState.pcsaft(pc_saft_parameters).joback(joback_parameters)
```

## Data types

```{eval-rst}
.. currentmodule:: feos.ideal_gas

.. autosummary::
    :toctree: generated/

    JobackRecord
    JobackParameters
```