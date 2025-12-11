# UV Theory

UV theory equation of state for Mie fluids with different perturbation variants.

```python
from feos.parameters import Parameters
from feos import EquationOfState

# Load UV theory parameters
parameters = Parameters.from_json(
    ["argon", "methane"],
    "parameters/uvtheory/example.json"
)

# Create UV theory equation of state
uvtheory = EquationOfState.uvtheory(
    parameters, 
    perturbation="WCA"
)
```

::: feos.EquationOfState.uvtheory
    options:
      show_root_full_path: true
      docstring_section_style: spacy
      summary: 
        attributes: false
        functions: true