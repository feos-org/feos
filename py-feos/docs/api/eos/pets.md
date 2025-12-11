# PeTS

Perturbed truncated and shifted Lennard-Jones equation of state.

```python
from feos.parameters import Parameters
from feos import EquationOfState

# Load PeTS parameters
parameters = Parameters.from_json(
    ["methane", "ethane"],
    "parameters/pets/example.json"
)

# Create PeTS equation of state
pets = EquationOfState.pets(parameters)
```

::: feos.EquationOfState.pets
    options:
      show_root_full_path: true
      docstring_section_style: spacy
      summary: 
        attributes: false
        functions: true