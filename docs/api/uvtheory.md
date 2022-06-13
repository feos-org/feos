# `feos.uvtheory`

Utilities to build `UVParameters`.

## Example

```python
from feos.uvtheory import UVParameters

parameters = UVParameters.from_json(['methane', 'ethane'], 'parameters.json')
```

## Data types

```{eval-rst}
.. currentmodule:: feos.uvtheory

.. autosummary::
    :toctree: generated/

    Identifier
    ChemicalRecord
    PureRecord
    SegmentRecord
    BinaryRecord
    BinarySegmentRecord
    Perturbation
    UVRecord
    UVParameters
```