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
All properties that have parallel automatic differentiation with respect to model parameters enabled are available from the `Property` class

```{eval-rst}
.. currentmodule:: feos

.. autosummary::
    :toctree: generated/
    
    Property
```

Currently the following phase equilibrium properties are available in the AD interface of FeOs. We plan to extend the list in the future.

|Property|Pure/Binary|Inputs|
|-|-|-|
|`vapor_pressure`|Pure|Temperature|
|`boiling_temperature`|Pure|Pressure|
|`liquid_density`|Pure|Temperature, Pressure|
|`equilibrium_liquid_density`|Pure|Temperature|
|`enthalpy_of_vaporization`|Pure|Temperature|
|`residual_isobaric_heat_capacity`|Pure|Temperature, Pressure|
|`bubble_point_pressure`|Binary|Temperature, x1, Presssure estimate|
|`dew_point_pressure`|Binary|Temperature, y1, Presssure estimate|

For all properties `xyz`, `Property` contains a method `Property.xyz(eos, input)` which uses a model from the [`EquationOfState`](eos.md#the-equationofstate-class) class. This evaluation can be useful for comparisons of the same data to established models. For parameter estimation or learning, the `Property.xyz_derivatives(model, parameter_names, parameters, input)` methods can be used to determine values and derivatives with respect to the model parameters indicated in `parameter_names`.

## Examples
The following example calculates pure-component vapor pressures including their derivatives with respect to the core PC-SAFT parameters for 10 Million temperatures in merely two seconds.

```python
import feos
import numpy as np

n = 10_000_000
fit_params = ["m", "sigma", "epsilon_k"]

# order: m, sigma, epsilon_k, mu
parameters = np.array([[1.5, 3.4, 230.0, 2.3]] * n)
temperature = np.expand_dims(np.linspace(250.0, 400.0, n), 1)
eos = feos.EquationOfStateAD.PcSaftNonAssoc
%time feos.Property.vapor_pressure_derivatives(eos, fit_params, parameters, temperature)
```
```
CPU times: user 1min 38s, sys: 611 ms, total: 1min 39s
Wall time: 1.98 s
```

For the most complex case, a binary mixture of cross-associating mixtures, the following example calculates 100.000 bubble point pressures and their derivative with respect to the binary interaction parameter in less than 5 seconds.
```python
import feos
import numpy as np

n = 100_000
fit_params = ["k_ij"]
parameters = np.array([[
    # Substance 1: m, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab, na, nb
    1.5, 3.4, 230.0, 2.3, 0.01, 1200.0, 1.0, 2.0, 
    # Substance 2: m, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab, na, nb
    2.3, 3.5, 245.0, 1.4, 0.005, 500.0, 1.0, 1.0,
    # k_ij
    0.01,
]] * n)
temperature = np.linspace(200.0, 388.0, n)
molefracs = np.array([0.5] * n)
pressure = np.array([1e5] * n)
input = np.stack((temperature, molefracs, pressure), axis=1)
eos = feos.EquationOfStateAD.PcSaftFull
%time feos.Property.bubble_point_pressure_derivatives(eos, fit_params, parameters, input)
```
```
CPU times: user 4min 52s, sys: 87.1 ms, total: 4min 53s
Wall time: 4.74 s
```