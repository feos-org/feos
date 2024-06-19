# `feos.uvtheory`

Utilities to build `UVTheoryParameters`.

## Example

```python
from feos.uvtheory import UVTheoryParameters

parameters = UVTheoryParameters.from_json(['methane', 'ethane'], 'parameters.json')
```

## Data types

```{eval-rst}
.. currentmodule:: feos.uvtheory

.. autosummary::
    :toctree: generated/

    Identifier
    IdentifierOption
    PureRecord
    BinaryRecord
    Perturbation
    UVTheoryRecord
    UVTheoryBinaryRecord
    UVTheoryParameters
```
