# Critical point of pure substance

```python
from feos.eos import EquationOfState, State
from feos.pcsaft import PcSaftParameters

parameters = PcSaftParameters.from_json(['methanol'], 'parameters.json')
eos = EquationOfState.pcsaft(parameters)
critical_point = State.critical_point(eos)
```