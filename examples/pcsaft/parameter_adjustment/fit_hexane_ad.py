"""
PC-SAFT parameter fitting for hexane using the AD-based regressor.

Fits m, sigma, and epsilon_k to vapor-pressure and liquid-density data
using the Levenberg-Marquardt algorithm with automatic differentiation.

Run with:
    uv run python fit_hexane_ad.py
"""

import os

from feos import (
    EquationOfStateAD,
    LiquidDensityDataset,
    LossFunction,
    PureRegressor,
    RegressorConfig,
    VaporPressureDataset,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


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

    # All four PC-SAFT parameters must be supplied; mu is fixed at 0.
    # The "fit" list controls which are optimised.
    reg = PureRegressor(
        model=EquationOfStateAD.PcSaftNonAssoc,
        datasets=[vp, rho],
        params={"m": 3.0, "sigma": 3.8, "epsilon_k": 240.0, "mu": 0.0},
        fit=["m", "sigma", "epsilon_k"],
        weights=[1.0, 1.0],
    )

    config = RegressorConfig(
        ftol=1e-8, xtol=1e-8, gtol=1e-8, stepbound=0.1, patience=200
    )
    result = reg.fit(config=config, loss=LossFunction.huber(0.05))

    print(f"\nConverged : {result.converged}  ({result.termination_reason})")
    print(f"Evaluations: {result.n_evaluations}   elapsed: {result.elapsed_ms:.1f} ms")

    print("\nOptimal parameters:")
    for name, val in result.optimal_params_dict().items():
        print(f"  {name:<12} = {val:.6g}")

    print("\nAAD per dataset:")
    for name, aad in zip(result.dataset_names, result.aad_per_dataset):
        if aad is not None:
            print(f"  {name:<20} {aad:.2f} %")
        else:
            print(f"  {name:<20} no convergence")

    datasets = reg.evaluate_datasets()  # uses fitted params automatically
    print("\nPer-dataset convergence:")
    for name, ds in zip(result.dataset_names, datasets):
        n_conv = sum(ds["converged"])
        n_total = len(ds["converged"])
        print(f"  {name:<20} {n_conv}/{n_total} converged")


if __name__ == "__main__":
    main()
