use super::PcSaftParameters;
use crate::hard_sphere::HardSphereProperties;
use crate::pcsaft::eos::polar::{
    MeanSegmentNumbers, Multipole, AD, ADQ, ALPHA, AQ, BD, BDQ, BQ, CD, CDQ, CQ, PI_SQ_43,
};
use feos_core::EosError;
use ndarray::*;
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_3, PI};

pub(super) fn calculate_helmholtz_energy_density_polar<N: DualNum<f64> + Copy + ScalarOperand>(
    parameters: &PcSaftParameters,
    temperature: N,
    density: ArrayView2<N>,
) -> Result<Array1<N>, EosError> {
    // temperature dependent segment radius
    let r = parameters.hs_diameter(temperature) * 0.5;

    // packing fraction
    let eta = density
        .outer_iter()
        .zip((&r * &r * &r * &parameters.m * 4.0 * FRAC_PI_3).into_iter())
        .fold(
            Array::zeros(density.raw_dim().remove_axis(Axis(0))),
            |acc: Array1<N>, (rho, r3m)| acc + &rho * r3m,
        );

    let mut phi = Array::zeros(eta.raw_dim());
    if parameters.ndipole > 0 {
        phi += &phi_polar_dipole(parameters, temperature, density, &eta)?;
    }
    if parameters.nquadpole > 0 {
        phi += &phi_polar_quadrupole(parameters, temperature, density, &eta)?;
    }
    if parameters.ndipole > 0 && parameters.nquadpole > 0 {
        phi += &phi_polar_dipole_quadrupole(parameters, temperature, density, &eta)?;
    }
    Ok(phi)
}

pub fn pair_integral_ij<N: DualNum<f64> + Copy + ScalarOperand>(
    mij1: f64,
    mij2: f64,
    eta: &Array1<N>,
    a: &[[f64; 3]],
    b: &[[f64; 3]],
    eps_ij_t: N,
) -> Array1<N> {
    let eta2 = eta * eta;
    let etas = [
        &Array::ones(eta.raw_dim()),
        eta,
        &eta2,
        &(&eta2 * eta),
        &(&eta2 * &eta2),
    ];
    let mut integral = Array::zeros(eta.raw_dim());
    for i in 0..a.len() {
        integral += &(etas[i]
            * (eps_ij_t * (b[i][0] + mij1 * b[i][1] + mij2 * b[i][2])
                + a[i][0]
                + mij1 * a[i][1]
                + mij2 * a[i][2]));
    }
    integral
}

pub fn triplet_integral_ijk<N: DualNum<f64> + ScalarOperand>(
    mijk1: f64,
    mijk2: f64,
    eta: &Array1<N>,
    c: &[[f64; 3]],
) -> Array1<N> {
    let eta2 = eta * eta;
    let etas = [&Array::ones(eta.raw_dim()), eta, &eta2, &(&eta2 * eta)];
    let mut integral = Array::zeros(eta.raw_dim());
    for i in 0..c.len() {
        integral += &(etas[i] * (c[i][0] + mijk1 * c[i][1] + mijk2 * c[i][2]));
    }
    integral
}

fn triplet_integral_ijk_dq<N: DualNum<f64> + ScalarOperand>(
    mijk: f64,
    eta: &Array1<N>,
    c: &[[f64; 2]],
) -> Array1<N> {
    let etas = [&Array::ones(eta.raw_dim()), eta, &(eta * eta)];
    let mut integral = Array::zeros(eta.raw_dim());
    for i in 0..c.len() {
        integral += &(etas[i] * (c[i][0] + mijk * c[i][1]));
    }
    integral
}

fn phi_polar_dipole<N: DualNum<f64> + Copy + ScalarOperand>(
    p: &PcSaftParameters,
    temperature: N,
    density: ArrayView2<N>,
    eta: &Array1<N>,
) -> Result<Array1<N>, EosError> {
    // mean segment number
    let m = MeanSegmentNumbers::new(p, Multipole::Dipole);

    let t_inv = temperature.inv();
    let eps_ij_t = p.e_k_ij.mapv(|v| t_inv * v);
    let sig_ij_3 = p.sigma_ij.mapv(|v| v.powi(3));
    let mu2_term: Array1<N> = p
        .dipole_comp
        .iter()
        .map(|&i| eps_ij_t[[i, i]] * sig_ij_3[[i, i]] * p.mu2[i])
        .collect();

    let mut phi2 = Array::zeros(eta.raw_dim());
    let mut phi3 = Array::zeros(eta.raw_dim());
    for i in 0..p.ndipole {
        let di = p.dipole_comp[i];
        phi2 -= &(&density.index_axis(Axis(0), di)
            * &density.index_axis(Axis(0), di)
            * pair_integral_ij(
                m.mij1[[i, i]],
                m.mij2[[i, i]],
                eta,
                &AD,
                &BD,
                eps_ij_t[[di, di]],
            )
            * (mu2_term[i] * mu2_term[i] / sig_ij_3[[di, di]]));
        phi3 -= &(&density.index_axis(Axis(0), di)
            * &density.index_axis(Axis(0), di)
            * density.index_axis(Axis(0), di)
            * triplet_integral_ijk(m.mijk1[[i, i, i]], m.mijk2[[i, i, i]], eta, &CD)
            * (mu2_term[i] * mu2_term[i] * mu2_term[i] / sig_ij_3[[di, di]]));
        for j in i + 1..p.ndipole {
            let dj = p.dipole_comp[j];
            phi2 -= &(&density.index_axis(Axis(0), di)
                * &density.index_axis(Axis(0), dj)
                * pair_integral_ij(
                    m.mij1[[i, j]],
                    m.mij2[[i, j]],
                    eta,
                    &AD,
                    &BD,
                    eps_ij_t[[di, dj]],
                )
                * (mu2_term[i] * mu2_term[j] / sig_ij_3[[di, dj]] * 2.0));
            phi3 -= &(&density.index_axis(Axis(0), di)
                * &density.index_axis(Axis(0), di)
                * density.index_axis(Axis(0), dj)
                * triplet_integral_ijk(m.mijk1[[i, i, j]], m.mijk2[[i, i, j]], eta, &CD)
                * (mu2_term[i] * mu2_term[i] * mu2_term[j]
                    / (p.sigma_ij[[di, di]] * p.sigma_ij[[di, dj]] * p.sigma_ij[[di, dj]])
                    * 3.0));
            phi3 -= &(&density.index_axis(Axis(0), di)
                * &density.index_axis(Axis(0), dj)
                * density.index_axis(Axis(0), dj)
                * triplet_integral_ijk(m.mijk1[[i, j, j]], m.mijk2[[i, j, j]], eta, &CD)
                * (mu2_term[i] * mu2_term[j] * mu2_term[j]
                    / (p.sigma_ij[[di, dj]] * p.sigma_ij[[di, dj]] * p.sigma_ij[[dj, dj]])
                    * 3.0));
            for k in j + 1..p.ndipole {
                let dk = p.dipole_comp[k];
                phi3 -= &(&density.index_axis(Axis(0), di)
                    * &density.index_axis(Axis(0), dj)
                    * density.index_axis(Axis(0), dk)
                    * triplet_integral_ijk(m.mijk1[[i, j, k]], m.mijk2[[i, j, k]], eta, &CD)
                    * (mu2_term[i] * mu2_term[j] * mu2_term[k]
                        / (p.sigma_ij[[di, dj]] * p.sigma_ij[[di, dk]] * p.sigma_ij[[dj, dk]])
                        * 6.0));
            }
        }
    }
    phi2 = phi2 * PI;
    phi3 = phi3 * PI_SQ_43;
    let mut result = &phi2 * &phi2 / (&phi2 - &phi3);
    result.iter_mut().zip(phi2.iter()).for_each(|(r, &p2)| {
        if r.re().is_nan() {
            *r = p2;
        }
    });
    Ok(result)
}

fn phi_polar_quadrupole<N: DualNum<f64> + Copy + ScalarOperand>(
    p: &PcSaftParameters,
    temperature: N,
    density: ArrayView2<N>,
    eta: &Array1<N>,
) -> Result<Array1<N>, EosError> {
    // mean segment number
    let m = MeanSegmentNumbers::new(p, Multipole::Quadrupole);

    let t_inv = temperature.inv();
    let eps_ij_t = p.e_k_ij.mapv(|v| t_inv * v);
    let sig_ij_3 = p.sigma_ij.mapv(|v| v.powi(3));
    let q2_term: Array1<N> = p
        .quadpole_comp
        .iter()
        .map(|&i| eps_ij_t[[i, i]] * p.sigma[i].powi(5) * p.q2[i])
        .collect();

    let mut phi2 = Array::zeros(eta.raw_dim());
    let mut phi3 = Array::zeros(eta.raw_dim());
    for i in 0..p.nquadpole {
        let di = p.quadpole_comp[i];
        phi2 -= &(&density.index_axis(Axis(0), di)
            * &density.index_axis(Axis(0), di)
            * pair_integral_ij(
                m.mij1[[i, i]],
                m.mij2[[i, i]],
                eta,
                &AQ,
                &BQ,
                eps_ij_t[[di, di]],
            )
            * (q2_term[i] * q2_term[i] / p.sigma_ij[[di, di]].powi(7)));
        phi3 += &(&density.index_axis(Axis(0), di)
            * &density.index_axis(Axis(0), di)
            * density.index_axis(Axis(0), di)
            * triplet_integral_ijk(m.mijk1[[i, i, i]], m.mijk2[[i, i, i]], eta, &CQ)
            * (q2_term[i] * q2_term[i] * q2_term[i] / sig_ij_3[[di, di]].powi(3)));
        for j in i + 1..p.nquadpole {
            let dj = p.quadpole_comp[j];
            phi2 -= &(&density.index_axis(Axis(0), di)
                * &density.index_axis(Axis(0), dj)
                * pair_integral_ij(
                    m.mij1[[i, j]],
                    m.mij2[[i, j]],
                    eta,
                    &AQ,
                    &BQ,
                    eps_ij_t[[di, dj]],
                )
                * (q2_term[i] * q2_term[j] / p.sigma_ij[[di, dj]].powi(7)));
            phi3 += &(&density.index_axis(Axis(0), di)
                * &density.index_axis(Axis(0), di)
                * density.index_axis(Axis(0), dj)
                * triplet_integral_ijk(m.mijk1[[i, i, j]], m.mijk2[[i, i, j]], eta, &CQ)
                * (q2_term[i] * q2_term[i] * q2_term[j]
                    / (sig_ij_3[[di, di]] * sig_ij_3[[di, dj]] * sig_ij_3[[di, dj]])
                    * 3.0));
            phi3 += &(&density.index_axis(Axis(0), di)
                * &density.index_axis(Axis(0), dj)
                * density.index_axis(Axis(0), dj)
                * triplet_integral_ijk(m.mijk1[[i, j, j]], m.mijk2[[i, j, j]], eta, &CQ)
                * (q2_term[i] * q2_term[j] * q2_term[j]
                    / (sig_ij_3[[di, dj]] * sig_ij_3[[di, dj]] * sig_ij_3[[dj, dj]])
                    * 3.0));
            for k in j + 1..p.nquadpole {
                let dk = p.quadpole_comp[k];
                phi3 += &(&density.index_axis(Axis(0), di)
                    * &density.index_axis(Axis(0), dj)
                    * density.index_axis(Axis(0), dk)
                    * triplet_integral_ijk(m.mijk1[[i, j, k]], m.mijk2[[i, j, k]], eta, &CQ)
                    * (q2_term[i] * q2_term[j] * q2_term[k]
                        / (sig_ij_3[[di, dj]] * sig_ij_3[[di, dk]] * sig_ij_3[[dj, dk]])
                        * 6.0));
            }
        }
    }
    phi2 = phi2 * (PI * 0.5625);
    phi3 = phi3 * (PI * PI * 0.5625);
    let mut result = &phi2 * &phi2 / (&phi2 - &phi3);
    result.iter_mut().zip(phi2.iter()).for_each(|(r, &p2)| {
        if r.re().is_nan() {
            *r = p2;
        }
    });
    Ok(result)
}

fn phi_polar_dipole_quadrupole<N: DualNum<f64> + Copy + ScalarOperand>(
    p: &PcSaftParameters,
    temperature: N,
    density: ArrayView2<N>,
    eta: &Array1<N>,
) -> Result<Array1<N>, EosError> {
    let t_inv = temperature.inv();
    let eps_ij_t = p.e_k_ij.mapv(|v| t_inv * v);
    let mu2_term: Array1<N> = p
        .dipole_comp
        .iter()
        .map(|&i| eps_ij_t[[i, i]] * p.sigma[i].powi(4) * p.mu2[i])
        .collect();
    let q2_term: Array1<N> = p
        .quadpole_comp
        .iter()
        .map(|&i| eps_ij_t[[i, i]] * p.sigma[i].powi(4) * p.q2[i])
        .collect();

    // mean segment number
    let mut mdq1 = Array2::zeros((p.ndipole, p.nquadpole));
    let mut mdq2 = Array2::zeros((p.ndipole, p.nquadpole));
    let mut mdqd = Array3::zeros((p.ndipole, p.nquadpole, p.ndipole));
    let mut mdqq = Array3::zeros((p.ndipole, p.nquadpole, p.nquadpole));
    for i in 0..p.ndipole {
        let di = p.dipole_comp[i];
        let mi = p.m[di].min(2.0);
        for j in 0..p.nquadpole {
            let qj = p.quadpole_comp[j];
            let mj = p.m[qj].min(2.0);
            let m = (mi * mj).sqrt();
            mdq1[[i, j]] = (m - 1.0) / m;
            mdq2[[i, j]] = mdq1[[i, j]] * (m - 2.0) / m;
            for k in 0..p.ndipole {
                let dk = p.dipole_comp[k];
                let mk = p.m[dk].min(2.0);
                let m = (mi * mj * mk).cbrt();
                mdqd[[i, j, k]] = (m - 1.0) / m;
            }
            for k in 0..p.nquadpole {
                let qk = p.quadpole_comp[k];
                let mk = p.m[qk].min(2.0);
                let m = (mi * mj * mk).cbrt();
                mdqq[[i, j, k]] = (m - 1.0) / m;
            }
        }
    }

    let mut phi2 = Array::zeros(eta.raw_dim());
    let mut phi3 = Array::zeros(eta.raw_dim());
    for i in 0..p.ndipole {
        let di = p.dipole_comp[i];
        for j in 0..p.nquadpole {
            let qj = p.quadpole_comp[j];
            phi2 -= &(&density.index_axis(Axis(0), di)
                * &density.index_axis(Axis(0), qj)
                * pair_integral_ij(
                    mdq1[[i, j]],
                    mdq2[[i, j]],
                    eta,
                    &ADQ,
                    &BDQ,
                    eps_ij_t[[di, qj]],
                )
                * (mu2_term[i] / p.sigma[di] * q2_term[j] * p.sigma[qj]
                    / p.sigma_ij[[di, qj]].powi(5)));
            for k in 0..p.ndipole {
                let dk = p.dipole_comp[k];
                phi3 += &(&density.index_axis(Axis(0), di)
                    * &density.index_axis(Axis(0), qj)
                    * density.index_axis(Axis(0), dk)
                    * triplet_integral_ijk_dq(mdqd[[i, j, k]], eta, &CDQ)
                    * (mu2_term[i] * q2_term[j] * mu2_term[k]
                        / (p.sigma_ij[[di, qj]] * p.sigma_ij[[di, dk]] * p.sigma_ij[[qj, dk]])
                            .powi(2)));
            }
            for k in 0..p.nquadpole {
                let qk = p.quadpole_comp[k];
                phi3 += &(&density.index_axis(Axis(0), di)
                    * &density.index_axis(Axis(0), qj)
                    * density.index_axis(Axis(0), qk)
                    * triplet_integral_ijk_dq(mdqq[[i, j, k]], eta, &CDQ)
                    * (mu2_term[i] * q2_term[j] * q2_term[k]
                        / (p.sigma_ij[[di, qj]] * p.sigma_ij[[di, qk]] * p.sigma_ij[[qj, qk]])
                            .powi(2)
                        * ALPHA));
            }
        }
    }
    phi2 = phi2 * (PI * 2.25);
    phi3 = phi3 * (PI * PI);
    let mut result = &phi2 * &phi2 / (&phi2 - &phi3);
    result.iter_mut().zip(phi2.iter()).for_each(|(r, &p2)| {
        if r.re().is_nan() {
            *r = p2;
        }
    });
    Ok(result)
}
