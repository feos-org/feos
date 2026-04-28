# Adjusting PC-SAFT parameters to experimental data

## Notebooks

- Adjust **PC-SAFT parameters** $m$, $\sigma$ and $\epsilon_k$ for a pure substance. [🠒 Notebook](adjust_non_polar_non_asssociating.ipynb)
- Adjust **binary PC-SAFT parameter** $k_{ij}$ to VLE data. [🠒 Notebook](adjust_kij.ipynb)
- Adjust **entropy scaling** correlation parameters for the viscosity of a pure substance using PC-SAFT. [🠒 Notebook](adjust_viscosity_correlation.ipynb)

## Python scripts

Self-contained scripts fit the same hexane data (vapor pressure + liquid density) with
the same initial parameters and Huber loss ($\delta = 0.05$), so their results and
performance can be compared directly.

| Script | Solver | Jacobian |
|--------|--------|----------|
| [`fit_hexane_ad.py`](fit_hexane_ad.py) | Levenberg-Marquardt via `PureRegressor` | exact, via automatic differentiation |
| [`fit_hexane_scipy.py`](fit_hexane_scipy.py) | `scipy.optimize.least_squares` | finite differences |
| [`fit_hexane_ad_scipy.py`](fit_hexane_ad_scipy.py) | `scipy.optimize.least_squares` | exact, via `PureRegressor.predict` |

`fit_hexane_ad_scipy.py` uses `PureRegressor` as a compute engine only — data loading
and AD-based gradient evaluation — while delegating the optimisation loop to scipy.
