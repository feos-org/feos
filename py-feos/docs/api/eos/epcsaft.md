# ePC-SAFT

Electrolyte PC-SAFT equation of state for ionic and non-ionic systems.

```python
from feos.parameters import Parameters
from feos import EquationOfState

# Load parameters for an ionic system
parameters = Parameters.from_json(
    ["water", "sodium chloride"],
    "parameters/epcsaft/example.json"
)

# Create ePC-SAFT equation of state
epcsaft = EquationOfState.epcsaft(
    parameters, 
    epcsaft_variant="advanced"
)
```

::: feos.EquationOfState.epcsaft
    options:
      show_root_full_path: true
      docstring_section_style: spacy
      summary: 
        attributes: false
        functions: true