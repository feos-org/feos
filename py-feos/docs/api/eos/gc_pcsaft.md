# GC-PC-SAFT

Group contribution PC-SAFT equation of state for molecules described using segments.

```python
from feos.parameters import GcParameters
from feos import EquationOfState

# Load group contribution parameters
parameters = GcParameters.from_json_segments(
    ["methanol", "ethanol"],
    "parameters/gc_pcsaft/chemical_records.json",
    "parameters/gc_pcsaft/segments.json"
)

# Create GC-PC-SAFT equation of state
gc_pcsaft = EquationOfState.gc_pcsaft(parameters)
```

::: feos.EquationOfState.gc_pcsaft
    options:
      show_root_full_path: true
      docstring_section_style: spacy
      summary: 
        attributes: false
        functions: true