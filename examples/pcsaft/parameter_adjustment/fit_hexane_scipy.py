"""
PC-SAFT parameter fitting for hexane using scipy.optimize.least_squares.

Uses the same data, initial parameters, and Huber loss (delta=0.05) as
fit_hexane_ad.py so that both approaches can be compared directly.

Run with:
    uv run python fit_hexane_scipy.py
"""

import os
import time

import numpy as np
import si_units as si
from feos import (
    EquationOfState,
    Identifier,
    Parameters,
    PhaseEquilibrium,
    PureRecord,
    State,
)
from scipy.optimize import least_squares

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
M_HEXANE = 86.177  # g/mol


def build_eos(m, sigma, epsilon_k):
    ident = Identifier(name="hexane")
    rec = PureRecord(
        ident, molarweight=M_HEXANE, m=m, sigma=sigma, epsilon_k=epsilon_k, mu=0.0
    )
    return EquationOfState.pcsaft(Parameters.new_pure(rec))


def load_vp():
    t, p = [], []
    with open(os.path.join(DATA_DIR, "hexane_vapor_pressure_si.csv")) as f:
        next(f)  # header
        for line in f:
            ti, pi = line.split(",")
            t.append(float(ti))
            p.append(float(pi))
    return np.array(t), np.array(p)


def load_rho():
    t, p, rho = [], [], []
    with open(os.path.join(DATA_DIR, "hexane_liquid_density_si.csv")) as f:
        next(f)  # header
        for line in f:
            ti, pi, ri = line.split(",")
            t.append(float(ti))
            p.append(float(pi))
            rho.append(float(ri))
    return np.array(t), np.array(p), np.array(rho)


PENALTY = 10.0


def compute_vp_residuals(eos, t_arr, psat_exp):
    res = np.full(len(t_arr), PENALTY)
    for i, (t, psat) in enumerate(zip(t_arr, psat_exp)):
        try:
            vle = PhaseEquilibrium.pure(eos, t * si.KELVIN)
            psat_calc = vle.liquid.pressure() / si.PASCAL
            res[i] = (psat_calc - psat) / psat
        except Exception:
            pass
    return res


def compute_rho_residuals(eos, t_arr, p_arr, rho_exp):
    res = np.full(len(t_arr), PENALTY)
    for i, (t, p, rho) in enumerate(zip(t_arr, p_arr, rho_exp)):
        try:
            state = State(
                eos,
                temperature=t * si.KELVIN,
                pressure=p * si.PASCAL,
                density_initialization="liquid",
            )
            rho_calc = state.density / (si.KILO * si.MOL / si.METER**3)
            res[i] = (rho_calc - rho) / rho
        except Exception:
            pass
    return res


def residuals(params, t_vp, psat_exp, t_rho, p_rho, rho_exp):
    eos = build_eos(*params)
    r_vp = compute_vp_residuals(eos, t_vp, psat_exp)
    r_rho = compute_rho_residuals(eos, t_rho, p_rho, rho_exp)
    return np.concatenate([r_vp, r_rho])


def aad(r, mask):
    converged = mask & (r != PENALTY)
    if not converged.any():
        return None
    return np.abs(r[converged]).mean() * 100.0


def main():
    t_vp, psat_exp = load_vp()
    t_rho, p_rho, rho_exp = load_rho()
    print(f"Loaded {len(t_vp)} vapor-pressure and {len(t_rho)} liquid-density points.")

    x0 = [3.0, 3.8, 240.0]  # m, sigma, epsilon_k — same as fit_hexane_ad.py

    t0 = time.perf_counter()
    result = least_squares(
        residuals,
        x0,
        loss="huber",
        f_scale=0.05,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        args=(t_vp, psat_exp, t_rho, p_rho, rho_exp),
        max_nfev=200 * (len(x0) + 1),
        method="trf",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    m_opt, sigma_opt, eps_opt = result.x
    converged = result.success or result.cost < 1e-8

    print(f"\nConverged : {converged}  (status {result.status}: {result.message})")
    print(f"Evaluations: {result.nfev}   elapsed: {elapsed_ms:.1f} ms")

    print("\nOptimal parameters:")
    print(f"  {'m':<12} = {m_opt:.6g}")
    print(f"  {'sigma':<12} = {sigma_opt:.6g}")
    print(f"  {'epsilon_k':<12} = {eps_opt:.6g}")

    # AAD from unweighted relative residuals at optimum (no loss applied)
    eos_opt = build_eos(*result.x)
    r_vp = compute_vp_residuals(eos_opt, t_vp, psat_exp)
    r_rho = compute_rho_residuals(eos_opt, t_rho, p_rho, rho_exp)
    conv_vp = r_vp != PENALTY
    conv_rho = r_rho != PENALTY

    print("\nAAD per dataset:")
    aad_vp = aad(r_vp, conv_vp)
    aad_rho = aad(r_rho, conv_rho)
    print(
        f"  {'vapor pressure':<20} {aad_vp:.2f} %"
        if aad_vp
        else f"  {'vapor pressure':<20} no convergence"
    )
    print(
        f"  {'liquid density':<20} {aad_rho:.2f} %"
        if aad_rho
        else f"  {'liquid density':<20} no convergence"
    )

    print("\nPer-dataset convergence:")
    print(f"  {'vapor pressure':<20} {conv_vp.sum()}/{len(conv_vp)} converged")
    print(f"  {'liquid density':<20} {conv_rho.sum()}/{len(conv_rho)} converged")


if __name__ == "__main__":
    main()
