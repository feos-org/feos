# `feos.epcsaft`

Utilities to build `ElectrolytePcSaftParameters`.

## Example

```python
from feos.epcsaft import ElectrolytePcSaftParameters

pure_path = 'parameters/epcsaft/held2014_w_permittivity_added.json'
binary_path = 'parameters/epcsaft/held2014_binary.json'
parameters = ElectrolytePcSaftParameters.from_json(['water', 'sodium ion', 'chloride ion'], pure_path, binary_path)
```

## Data types

```{eval-rst}
.. currentmodule:: feos.epcsaft

.. autosummary::
    :toctree: generated/

    Identifier
    IdentifierOption
    ChemicalRecord
    SmartsRecord
    ElectrolytePcSaftVariants
    ElectrolytePcSaftRecord
    ElectrolytePcSaftBinaryRecord
    PureRecord
    SegmentRecord
    BinaryRecord
    BinarySegmentRecord   
    ElectrolytePcSaftParameters
    PermittivityRecord
```