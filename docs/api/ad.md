# Automatic differentiation

Automatic differentiation (AD) is used all throughout FeOs to calculate Helmholtz energy derivatives and Jacobians/Hessians for numerical solvers.
This section refers specifically to automatic (implicit) differentiation of phase equilibria with respect to model parameters that are crucial for parameter prediction or estimation.

Within Rust most phase equilibrium calculations can be used in an AD context, e.g., in order to calculate derivatives of process models (see [here](https://github.com/feos-org/feos-campd)). The Python interface focuses on the important use case of massively parallel phase equilibrium calculations for parameter estimation or prediction.

## Available equations of state
Only a subset of the models in FeOs can be used to calculate derivatives with respect to model parameters. However, those dedicated implementations also result in unprecedented performance (see example below). Similar to the [`EquationOfState`](eos.md#the-equationofstate-class) class, models with AD capabilities are collected in the `EquationOfStateAD` class. 

The currently available models are:

|Model|Description|Parameters pure|Parameters binary|
|-|-|-|-|
|`PcSaftNonAssoc`|The PC-SAFT equation of state including a dipolar contribution but no association|`m`, `sigma`, `epsilon_k`, `mu`|`k_ij`|
|`PcSaftFull`|The PC-SAFT equation of state with a dipolar contribution and association|`m`, `sigma`, `epsilon_k`, `mu`, `kappa_ab`, `epsilon_k_ab`, `na`, `nb`|`k_ij`|

## Properties 
Currently the following phase equilibrium properties are available in the AD interface of FeOs. We plan to extend the list in the future.

```{eval-rst}
.. currentmodule:: feos

.. autosummary::
    :toctree: generated/
    
    vapor_pressure_derivatives
    liquid_density_derivatives
    equilibrium_liquid_density_derivatives
    bubble_point_pressure_derivatives
    dew_point_pressure_derivatives
```

## Examples
The following example calculates pure-component vapor pressures including their derivatives with respect to the core PC-SAFT parameters for 10 Million temperatures in little more than two seconds.
```python
import feos
import numpy as np
n = 10000000
fit_params = ["m", "sigma", "epsilon_k"]
parameters = np.array([[1.5,3.4,230.,2.3]]*n)
temperature = np.expand_dims(np.linspace(250., 400.0, n),1)
eos = feos.EquationOfStateAD.PcSaftNonAssoc
%time feos.vapor_pressure_derivatives(eos, fit_params, parameters, temperature)
```
```
CPU times: user 1min 36s, sys: 850 ms, total: 1min 37s
Wall time: 2.29 s
```

For the most complex case, a binary mixture of cross-associating mixtures, the following example calculates 100.000 bubble point pressures and their derivative with respect to the binary interaction parameter in 3 seconds.
```python
import feos
import numpy as np
n = 100000
fit_params = ["k_ij"]
parameters = np.array([[1.5,3.4,230.,2.3,0.01,1200.,1.0,2.0,2.3,3.5,245.,1.4,0.005,500.,1.0,1.0,0.01]]*n)
temperature = np.linspace(200., 388.0, n)
molefracs = np.array([.5]*n)
pressure = np.array([1e5]*n)
input = np.stack((temperature, molefracs, pressure),axis=1)
eos = feos.EquationOfStateAD.PcSaftFull
%time feos.bubble_point_pressure_derivatives(eos, fit_params, parameters, input)
```
```
CPU times: user 3min 10s, sys: 16.3 ms, total: 3min 10s
Wall time: 3.06 s
```