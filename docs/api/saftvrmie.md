# `feos.saftvrmie`

Utilities to build `SaftVRMieParameters`.

## Example

```python
from feos.saftvrmie import SaftVRMieParameters

path = 'parameters/saftvrmie/lafitte2013.json'
parameters = SaftVRMieParameters.from_json(['ethane', 'methane'], path)
```

## Data types

```{eval-rst}
.. currentmodule:: feos.saftvrmie

.. autosummary::
    :toctree: generated/

    Identifier
    IdentifierOption
    PureRecord
    BinaryRecord
    SaftVRMieRecord
    SaftVRMieBinaryRecord
    SaftVRMieParameters
```
