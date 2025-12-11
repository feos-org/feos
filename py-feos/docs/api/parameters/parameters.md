# Parameters

```python
from feos.parameters import Parameters

parameters = Parameters(...)
```

!!! info

    Parameters are note evaluated until provided to an equation of state.

::: feos.Parameters
    options:
      members:
        - from_records
        - new_pure
        - new_binary
        - from_multiple_json
        - from_json
        - to_json_str
        - pure_records
        - binary_records
      summary: 
        attributes: true
        functions: true

---

::: feos.GcParameters
    options:
      members:
        - from_segments
        - from_json_segments
        - from_smiles
        - from_json_smiles
        - chemical_records
        - segment_records
        - binary_segment_records
      summary: 
        attributes: true
        functions: true
