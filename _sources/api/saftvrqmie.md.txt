# `feos.saftvrqmie`

Utilities to build `SaftVRQMieParameters`.

## Example

```python
from feos.saftvrqmie import SaftVRQMieParameters

parameters = SaftVRQMieParameters.from_json(['hydrogen', 'neon'], 'parameters.json')
```

## Data types

```{eval-rst}
.. currentmodule:: feos.saftvrqmie

.. autosummary::
    :toctree: generated/

    FeynmanHibbsOrder
    Identifier
    PureRecord
    BinaryRecord
    SaftVRQMieRecord
    SaftVRQMieParameters
```