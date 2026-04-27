"""
PC-SAFT parameter fitting for hexane: AD gradients from PureRegressor, scipy as solver.

PureRegressor handles data loading and evaluates the model + exact AD Jacobian.
A thin cached wrapper bridges that to scipy.optimize.least_squares.

Run with:
    uv run python fit_hexane_ad_scipy.py
"""

import os
import time

import numpy as np
from feos import (
    EquationOfStateAD,
    LiquidDensityDataset,
    PureRegressor,
    VaporPressureDataset,
)
from scipy.optimize import least_squares

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PENALTY = 10.0


class RegressorPredictor:
    """Wraps a PureRegressor as a cached cost/jac computation for scipy.

    PureRegressor.predict() computes model values and exact AD gradients
    w.r.t. the fitted parameters in one parallel pass.
    Scipy requires separate cost and jac calls, so this class caches
    the result of a single evaluation of both so that each the first call
    to cost or jac triggers a model evaluation, and subsequent calls reuse
    the cached result.
    """

    def __init__(self, regressor: PureRegressor, all_params: dict):
        self.regressor = regressor
        all_names = regressor.all_param_names
        fit_names = regressor.fitted_param_names
        self._fit_indices = [all_names.index(n) for n in fit_names]
        self._base = np.array([all_params[n] for n in all_names])

        n = len(regressor.target())
        p = len(fit_names)
        self._last_x: np.ndarray | None = None
        self._prediction: np.ndarray = np.empty(n)
        self._grad: np.ndarray = np.empty((n, p))
        self._conv: np.ndarray = np.zeros(n, dtype=bool)

    def _full_params(self, x: np.ndarray) -> np.ndarray:
        """Insert the fitted values x into the full canonical parameter vector."""
        full = self._base.copy()
        for j, idx in enumerate(self._fit_indices):
            full[idx] = x[j]
        return full

    def _evaluate(self, x: np.ndarray) -> None:
        if self._last_x is None or not np.array_equal(x, self._last_x):
            self._last_x = x.copy()
            full = self._full_params(x)
            self._prediction, self._grad, self._conv = self.regressor.predict(
                list(full)
            )

    def cost(self, x: np.ndarray) -> np.ndarray:
        self._evaluate(x)
        target = self.regressor.target()
        return np.where(self._conv, (self._prediction - target) / target, PENALTY)

    def jac(self, x: np.ndarray) -> np.ndarray:
        self._evaluate(x)
        target = self.regressor.target()
        j = self._grad / target[:, np.newaxis]
        j[~self._conv] = 0.0
        return j


def main():
    vp = VaporPressureDataset.from_csv(
        os.path.join(DATA_DIR, "hexane_vapor_pressure_si.csv"),
        name="vapor pressure",
    )
    rho = LiquidDensityDataset.from_csv(
        os.path.join(DATA_DIR, "hexane_liquid_density_si.csv"),
        name="liquid density",
    )
    print(f"Loaded {len(vp)} vapor-pressure and {len(rho)} liquid-density points.")

    initial = {"m": 3.0, "sigma": 3.8, "epsilon_k": 240.0, "mu": 0.0}
    fit = ["m", "sigma", "epsilon_k"]
    reg = PureRegressor(
        model=EquationOfStateAD.PcSaftNonAssoc,
        datasets=[vp, rho],
        params=initial,
        fit=fit,
        weights=[1.0, 1.0],
    )

    predictor = RegressorPredictor(reg, initial)
    x0 = np.array([initial[n] for n in fit])
    t0 = time.perf_counter()
    result = least_squares(
        predictor.cost,
        x0,
        jac=predictor.jac,
        loss="huber",
        f_scale=0.05,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=200 * (len(x0) + 1),
        method="trf",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    converged = result.success or result.cost < 1e-8
    print(f"\nConverged : {converged}  (status {result.status}: {result.message})")
    print(f"Evaluations: {result.nfev}   elapsed: {elapsed_ms:.1f} ms")

    print("\nOptimal parameters:")
    for name, val in zip(fit, result.x):
        print(f"  {name:<12} = {val:.6g}")

    r = predictor.cost(result.x)
    conv = predictor._conv
    n_vp = len(vp)
    r_vp, r_rho = r[:n_vp], r[n_vp:]
    conv_vp, conv_rho = conv[:n_vp], conv[n_vp:]

    print("\nAAD per dataset:")
    for label, rv, cv in [
        ("vapor pressure", r_vp, conv_vp),
        ("liquid density", r_rho, conv_rho),
    ]:
        converged_r = rv[cv & (rv != PENALTY)]
        if len(converged_r):
            print(f"  {label:<20} {np.abs(converged_r).mean() * 100:.2f} %")
        else:
            print(f"  {label:<20} no convergence")

    print("\nPer-dataset convergence:")
    for label, cv in [("vapor pressure", conv_vp), ("liquid density", conv_rho)]:
        print(f"  {label:<20} {cv.sum()}/{len(cv)} converged")


if __name__ == "__main__":
    main()
