# `feos.si`

This module contains data types for dimensioned quantities, i.e. floating point values multiplied with units.
If you want to learn more about this module, take a look at [this notebook](/tutorials/utility/core_working_with_units).

## Example usage

```python
from feos.si import KELVIN, RGAS, METER, MOL, BAR

p_ig = 25.0 * MOL / METER**3 * RGAS * 315.5 * KELVIN
print('Ideal gas pressure: {:.2f} bar'.format(p_ig / BAR))
```
```
Ideal gas pressure: 0.66 bar
```

## Data types

```{eval-rst}
.. currentmodule:: feos.si

.. autosummary::
    :toctree: generated/

    SINumber
    SIArray1
    SIArray2
    SIArray3
    SIArray4
```