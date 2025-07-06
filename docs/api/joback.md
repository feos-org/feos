# `feos.joback`

Ideal gas model based on the Joback & Reid GC method.

## Example: Combine the ideal gas model of Joback & Reid with PC-SAFT

```python
from feos.eos import EquationOfState, State
from feos.pcsaft import PcSaftParameters
from feos.joback import Joback

pc_saft_parameters = PcSaftParameters.from_json_smiles(
    ['CCC', 'CCCC'],
    'smarts.json',
    'pcsaft_group_parameters.json'
)
joback = Joback.from_json_smiles(
    ['CCC', 'CCCC'],
    'smarts.json',
    'joback_parameters.json'
)
eos = EquationOfState.pcsaft(pc_saft_parameters).joback(joback)
```

## Data types

```{eval-rst}
.. currentmodule:: feos.joback

.. autosummary::
    :toctree: generated/

    Identifier
    IdentifierOption
    JobackRecord
    PureRecord
    SegmentRecord
    Joback
```