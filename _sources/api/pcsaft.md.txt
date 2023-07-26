# `feos.pcsaft`

Utilities to build `PcSaftParameters`. To learn more about ways to build parameters from files or within Python, see [this example](/examples/eos/pcsaft/pcsaft_working_with_parameters).

## Example

```python
from feos.pcsaft import PcSaftParameters

parameters = PcSaftParameters.from_json(['methane', 'ethane'], 'parameters.json')
```

## Data types

```{eval-rst}
.. currentmodule:: feos.pcsaft

.. autosummary::
    :toctree: generated/

    Identifier
    ChemicalRecord
    PureRecord
    SegmentRecord
    BinaryRecord
    BinarySegmentRecord
    DQVariants
    PcSaftRecord
    PcSaftParameters
```