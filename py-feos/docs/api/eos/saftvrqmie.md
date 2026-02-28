# SAFT-VRQ Mie

SAFT-VRQ Mie equation of state for quantum fluids with Mie interactions.

```python
from feos.parameters import Parameters
from feos import EquationOfState

# Load SAFT-VRQ Mie parameters
parameters = Parameters.from_json(
    ["hydrogen", "helium"],
    "parameters/saftvrqmie/example.json"
)

# Create SAFT-VRQ Mie equation of state
saftvrqmie = EquationOfState.saftvrqmie(
    parameters, 
    inc_nonadd_term=True
)
```

::: feos.EquationOfState.saftvrqmie
    options:
      show_root_full_path: true
      docstring_section_style: spacy
      summary: 
        attributes: false
        functions: true