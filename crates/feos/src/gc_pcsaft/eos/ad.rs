use super::dispersion::{A0, A1, A2, B0, B1, B2};
use super::polar::{AD, BD, CD};
use feos_core::{Residual, StateHD};
use nalgebra::{Const, SMatrix, SVector};
use num_dual::DualNum;
use quantity::{JOULE, KB, KELVIN};
use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_6, PI};
use std::sync::LazyLock;

const PI_SQ_43: f64 = 4.0 / 3.0 * PI * PI;

const MAX_ETA: f64 = 0.5;

const N_GROUPS: usize = 20;
const GROUPS: [&str; N_GROUPS] = [
    "CH3", "CH2", ">CH", ">C<", "=CH2", "=CH", "=C<", "C≡CH", "CH2_hex", "CH_hex", "CH2_pent",
    "CH_pent", "CH_arom", "C_arom", "CH=O", ">C=O", "OCH3", "OCH2", "HCOO", "COO",
];
const M: [f64; N_GROUPS] = [
    0.77247, 0.7912, 0.52235, -0.70131, 0.70581, 0.90182, 0.98505, 1.1615, 0.8793, 0.42115,
    0.90057, 0.69343, 0.88259, 0.77531, 1.1889, 1.1889, 1.1907, 1.1817, 1.2789, 1.2869,
];
const SIGMA: [f64; N_GROUPS] = [
    3.6937, 3.0207, 0.99912, 0.5435, 3.163, 2.8864, 2.245, 3.3187, 2.9995, 1.3078, 3.0437, 1.2894,
    2.9475, 1.6719, 3.2948, 3.1026, 2.7795, 3.009, 3.373, 3.0643,
];
const EPSILON_K: [f64; N_GROUPS] = [
    181.49, 157.23, 269.84, 0.0, 171.34, 158.9, 146.86, 255.13, 157.93, 131.79, 158.34, 140.69,
    156.51, 178.81, 316.91, 280.43, 284.91, 203.11, 307.44, 273.9,
];
const MU: [f64; N_GROUPS] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.4126, 3.4167, 0.0,
    2.6945, 2.6808, 3.3428,
];

const K_AB_LIST: [((&str, &str), f64); 128] = [
    (("CH3", "C≡CH"), 0.1652754031504817),
    (("CH2", "C≡CH"), -0.006553805247652898),
    ((">CH", "C≡CH"), -0.45179562711827204),
    ((">CH", "CH_arom"), -0.2623133986577006),
    (("CH3", "C_arom"), 0.1029438146806397),
    (("CH2", "C_arom"), -0.019417887747868838),
    (("CH3", "CH_arom"), 0.03524440115729556),
    (("CH2", "CH_arom"), -0.005376660474268035),
    ((">CH", "C_arom"), -0.4536159179058391),
    (("CH3", "CH=O"), 0.07114002498453002),
    (("CH2", "CH=O"), 0.05810424929073859),
    ((">C=O", "CH3"), 0.06638452980811302),
    ((">C=O", "CH2"), 0.053964852486994674),
    ((">C=O", ">CH"), -0.01886821792425231),
    (("CH3", "OCH2"), 0.0040270442665658705),
    ((">CH", "OCH3"), -0.217389127174595),
    (("CH2", "OCH2"), -0.015001007000489858),
    (("CH3", "OCH3"), 0.04370031373865),
    ((">CH", "OCH2"), -0.038601031822155345),
    (("CH2", "OCH3"), 0.021857618541656035),
    (("CH3", "HCOO"), 0.046221094393257035),
    (("CH2", "HCOO"), 0.07467469076976556),
    ((">CH", "COO"), 0.17498779403987508),
    (("CH3", "COO"), 0.05050525601666908),
    (("CH2", "COO"), 0.04126150329459275),
    (("=CH2", "C≡CH"), 0.28518205265981117),
    (("=CH", "C≡CH"), -0.19471892756822953),
    (("=C<", "C≡CH"), -0.37966037059629637),
    (("=CH", "C_arom"), -0.04794957652572591),
    (("=C<", "C_arom"), -0.19059760666493372),
    (("=CH2", "C_arom"), 0.002829235390387696),
    (("=CH", "CH_arom"), 0.008744506522926979),
    (("=C<", "CH_arom"), -0.15988275905975652),
    (("=CH2", "CH_arom"), 0.039714702972518216),
    (("=C<", ">C=O"), -0.23303936628601363),
    (("=CH2", ">C=O"), 0.0451103450863995),
    (("=CH", ">C=O"), 0.0028282796817093118),
    (("=CH2", "OCH3"), -0.017336555493171858),
    (("=CH", "OCH2"), 0.028713611730255537),
    (("=C<", "OCH2"), -0.040576835006969125),
    (("=CH2", "OCH2"), 0.02792348761278379),
    (("=CH", "OCH3"), 0.021854680107075346),
    (("=C<", "OCH3"), -0.21464227012985213),
    (("=CH", "HCOO"), -0.021573291908933357),
    (("=C<", "HCOO"), -0.021791386613505864),
    (("=CH2", "COO"), -0.08063693709595326),
    (("=CH", "COO"), -0.07829355920744586),
    (("=C<", "COO"), -0.19510136763283895),
    (("CH_arom", "C≡CH"), -0.04955767628386867),
    (("C_arom", "C≡CH"), -0.04953394596854589),
    (("CH=O", "C≡CH"), -0.33948211818518437),
    ((">C=O", "C≡CH"), -0.3657376137845608),
    (("C≡CH", "OCH2"), -0.3344648388007797),
    (("C≡CH", "OCH3"), -0.38290586519600983),
    (("C≡CH", "HCOO"), -0.24272079727170506),
    (("COO", "C≡CH"), -0.3654438081227738),
    (("C≡CH", "OH"), -0.10409652963367089),
    (("CH_arom", "C_arom"), -0.12728867698722554),
    (("CH_arom", "CH_arom"), -0.023433119170883764),
    (("C_arom", "C_arom"), -0.28421918877688607),
    (("CH2_hex", "CH_arom"), 0.001447550584366187),
    (("CH_hex", "C_arom"), -0.40316115168074723),
    (("CH_arom", "CH_hex"), -0.3576321797883022),
    (("CH2_hex", "C_arom"), 0.08521616156959391),
    (("CH_arom", "CH_pent"), -0.18714862430161244),
    (("CH_pent", "C_arom"), 0.03457714936255159),
    (("CH2_pent", "C_arom"), 0.03812116290565423),
    (("CH2_pent", "CH_arom"), 0.004966663517893626),
    (("CH=O", "C_arom"), -0.025044677100339502),
    (("CH=O", "CH_arom"), -0.06309730593619743),
    ((">C=O", "CH_arom"), -0.1106708081496441),
    ((">C=O", "C_arom"), -0.04717854573415193),
    (("C_arom", "OCH3"), -0.14006221692396298),
    (("CH_arom", "OCH2"), -0.11744345395130776),
    (("CH_arom", "OCH3"), -0.06563917699710337),
    (("C_arom", "OCH2"), -0.03747169315781876),
    (("CH_arom", "HCOO"), -0.04404325532489947),
    (("C_arom", "HCOO"), 0.2898776740748664),
    (("CH_arom", "COO"), -0.10600926342624745),
    (("COO", "C_arom"), -0.11054573554364296),
    (("C_arom", "OH"), 0.2990171120344822),
    (("CH_arom", "OH"), -0.039398695604311314),
    (("C_arom", "NH2"), 0.4535791221221567),
    (("CH_arom", "NH2"), -0.06290638692043257),
    (("CH2_hex", "CH=O"), 0.09047030071006708),
    (("CH=O", "CH_hex"), -0.14747598417210014),
    ((">C=O", "CH_hex"), 0.0676825668907787),
    ((">C=O", "CH2_hex"), 0.09082375748804353),
    (("CH2_hex", "OCH3"), 0.042823701076275526),
    (("CH_hex", "OCH2"), -0.0936451919984422),
    (("CH_hex", "OCH3"), 0.12111386208387202),
    (("CH2_hex", "OCH2"), 0.013698887705260425),
    (("CH2_hex", "HCOO"), 0.08719198819954514),
    (("CH2_hex", "COO"), 0.05937938878778157),
    (("CH_hex", "COO"), -0.10319900739370075),
    (("CH2_hex", "OH"), 0.06127398203560399),
    (("CH_hex", "OH"), 0.35825180807831797),
    (("CH2_pent", "COO"), 0.05124310486288829),
    (("CH_pent", "OH"), 0.11518421254437769),
    (("CH2_pent", "OH"), 0.08215868571093943),
    (("CH=O", "CH=O"), -0.1570556622280003),
    ((">C=O", "CH=O"), -0.16452206370456918),
    (("CH=O", "OCH2"), 0.0027095251191336708),
    (("CH=O", "HCOO"), -0.07673278642721204),
    (("CH=O", "COO"), -0.16365969940991995),
    (("CH=O", "OH"), -0.12245349842770242),
    ((">C=O", ">C=O"), -0.17931771307891634),
    ((">C=O", "OCH3"), -0.20038719736411056),
    ((">C=O", "OCH2"), 0.04468736703539099),
    ((">C=O", "HCOO"), -0.13541978245760022),
    ((">C=O", "COO"), -0.14605212162381323),
    ((">C=O", "OH"), -0.1392769563372809),
    ((">C=O", "NH2"), -0.371931948310995),
    (("OCH2", "OCH2"), 0.09662571077941844),
    (("OCH2", "OCH3"), -0.2812620283200189),
    (("OCH3", "OCH3"), -0.13909723652059505),
    (("HCOO", "OCH3"), -0.0929570619422749),
    (("COO", "OCH2"), -0.11408007406963222),
    (("COO", "OCH3"), -0.21710938244245623),
    (("OCH2", "OH"), -0.014272196525878467),
    (("OCH3", "OH"), -0.039585706351111166),
    (("HCOO", "HCOO"), -0.14303475853269773),
    (("COO", "HCOO"), -0.14056680898820434),
    (("HCOO", "OH"), -0.11204049889700908),
    (("COO", "COO"), -0.1879219131496382),
    (("COO", "OH"), -0.09507071103459414),
    (("COO", "NH2"), -0.2799573216348791),
    (("NH2", "OH"), -0.42107448986356144),
];

static K_AB_MAP: LazyLock<HashMap<(&str, &str), f64>> = LazyLock::new(|| {
    K_AB_LIST
        .into_iter()
        .map(|((s1, s2), val)| ((s2, s1), val))
        .chain(K_AB_LIST)
        .collect()
});

static K_AB: LazyLock<SMatrix<f64, N_GROUPS, N_GROUPS>> = LazyLock::new(|| {
    SMatrix::from_fn(|i, j| *K_AB_MAP.get(&(GROUPS[i], GROUPS[j])).unwrap_or(&0.0))
});

/// Parameters used to instantiate [GcPcSaft].
#[derive(Clone)]
pub struct GcPcSaftADParameters<D, const N: usize> {
    pub groups: SMatrix<D, N_GROUPS, N>,
    pub bonds: [Vec<([usize; 2], D)>; N],
}

impl<D: DualNum<f64> + Copy, const N: usize> GcPcSaftADParameters<D, N> {
    pub fn re(&self) -> GcPcSaftADParameters<f64, N> {
        let Self { groups, bonds } = self;
        let groups = groups.map(|g| g.re());
        let bonds = bonds
            .each_ref()
            .map(|b| b.iter().map(|&(b, v)| (b, v.re())).collect());
        GcPcSaftADParameters { groups, bonds }
    }
}

impl<D: DualNum<f64> + Copy, const N: usize> GcPcSaftADParameters<D, N> {
    pub fn from_groups(
        group_map: [&HashMap<&'static str, D>; N],
        bond_map: [&HashMap<[&'static str; 2], D>; N],
    ) -> Self {
        let groups =
            SMatrix::from(group_map.map(|r| GROUPS.map(|g| *r.get(g).unwrap_or(&D::zero()))));
        let group_indices: HashMap<_, _> = GROUPS
            .into_iter()
            .enumerate()
            .map(|(g, i)| (i, g))
            .collect();
        let bonds = bond_map.map(|r| {
            r.iter()
                .map(|([g1, g2], &c)| ([group_indices[g1], group_indices[g2]], c))
                .collect()
        });
        Self { groups, bonds }
    }
}

/// The heterosegmented GC model for PC-SAFT by Sauer et al.
#[derive(Clone)]
pub struct GcPcSaftAD<D, const N: usize>(pub GcPcSaftADParameters<D, N>);

impl<D: DualNum<f64> + Copy, const N: usize> Residual<Const<N>, D> for GcPcSaftAD<D, N> {
    fn components(&self) -> usize {
        N
    }

    type Real = GcPcSaftAD<f64, N>;
    type Lifted<D2: DualNum<f64, Inner = D> + Copy> = GcPcSaftAD<D2, N>;
    fn re(&self) -> Self::Real {
        GcPcSaftAD(self.0.re())
    }
    fn lift<D2: DualNum<f64, Inner = D> + Copy>(&self) -> Self::Lifted<D2> {
        let GcPcSaftADParameters { groups, bonds } = &self.0;
        let groups = groups.map(|x| D2::from_inner(&x));
        let bonds = bonds
            .each_ref()
            .map(|b| b.iter().map(|&(b, v)| (b, D2::from_inner(&v))).collect());
        GcPcSaftAD(GcPcSaftADParameters { groups, bonds })
    }

    fn compute_max_density(&self, molefracs: &SVector<D, N>) -> D {
        let msigma3: SVector<f64, N_GROUPS> = SVector::from_fn(|i, _| M[i] * SIGMA[i].powi(3));
        let msigma3 = msigma3.map(D::from);
        let GcPcSaftADParameters { groups, bonds: _ } = &self.0;
        let msigma3 = apply_group_count(groups, &msigma3).row_sum();
        // let x: f64 = msigma3.iter().zip(molefracs).map(|&(ms3, x)| x * ms3).sum();
        ((msigma3 * molefracs).into_scalar() * FRAC_PI_6).recip() * MAX_ETA
    }

    fn reduced_helmholtz_energy_density_contributions(
        &self,
        state: &StateHD<D, Const<N>>,
    ) -> Vec<(&'static str, D)> {
        vec![(
            "gc-PC-SAFT",
            self.reduced_residual_helmholtz_energy_density(state),
        )]
    }

    fn reduced_residual_helmholtz_energy_density(&self, state: &StateHD<D, Const<N>>) -> D {
        let GcPcSaftADParameters { groups, bonds } = &self.0;
        let density = &state.partial_density;

        // convert parameters
        let m = apply_group_count(groups, &SVector::from(M.map(D::from)));
        let sigma = SVector::from(SIGMA.map(D::from));
        let epsilon_k = SVector::from(EPSILON_K.map(D::from));
        let mu = SVector::from(MU.map(D::from));

        // temperature dependent segment diameter
        let t_inv = state.temperature.recip();
        let diameter = (epsilon_k * (t_inv * -3.0))
            .map(|x| -x.exp() * 0.12 + 1.0)
            .component_mul(&sigma);

        // packing fractions
        let mut zeta = [D::zero(); 4];
        for c in 0..N {
            for i in 0..diameter.len() {
                for (z, &k) in zeta.iter_mut().zip([0, 1, 2, 3].iter()) {
                    *z += density[c] * diameter[i].powi(k) * m[(i, c)] * FRAC_PI_6;
                }
            }
        }
        let zeta_23 = zeta[2] / zeta[3];
        let frac_1mz3 = -(zeta[3] - 1.0).recip();

        // hard sphere
        let hs = (zeta[1] * zeta[2] * frac_1mz3 * 3.0
            + zeta[2].powi(2) * frac_1mz3.powi(2) * zeta_23
            + (zeta[2] * zeta_23.powi(2) - zeta[0]) * (zeta[3] * (-1.0)).ln_1p())
            / std::f64::consts::FRAC_PI_6;

        // hard chain
        let c = zeta[2] * frac_1mz3 * frac_1mz3;
        let hc: D = bonds
            .iter()
            .zip(density.iter())
            .flat_map(|(bonds, &rho)| {
                bonds.iter().map(move |([i, j], count)| {
                    let (di, dj) = (diameter[*i], diameter[*j]);
                    let cdij = c * di * dj / (di + dj);
                    let g = frac_1mz3 + cdij * 3.0 - cdij * cdij * (zeta[3] - 1.0) * 2.0;
                    -rho * count * g.ln()
                })
            })
            .sum();

        // packing fraction
        let eta = zeta[3];

        // mean segment number
        let molefracs = density / density.sum();
        let mbar = m.row_sum().tr_dot(&molefracs);

        // crosswise interactions of all groups on all chains
        let eps_ij = SVector::from(EPSILON_K).map(f64::sqrt);
        let eps_ij = eps_ij * eps_ij.transpose();
        let mut rho1mix = D::zero();
        let mut rho2mix = D::zero();
        for (m_i, &rho_i) in m.column_iter().zip(density.iter()) {
            for (m_j, &rho_j) in m.column_iter().zip(density.iter()) {
                for i in 0..N_GROUPS {
                    for j in 0..N_GROUPS {
                        let k_ab = if m_i != m_j { K_AB[(i, j)] } else { 0.0 };
                        let eps_ij_t = state.temperature.recip() * eps_ij[(i, j)] * (1.0 - k_ab);
                        let sigma_ij = ((SIGMA[i] + SIGMA[j]) * 0.5).powi(3);
                        let rho1 = rho_i * rho_j * (eps_ij_t * m_i[i] * m_j[j] * sigma_ij);
                        rho1mix += rho1;
                        rho2mix += rho1 * eps_ij_t;
                    }
                }
            }
        }

        // I1, I2 and C1
        let mut i1 = D::zero();
        let mut i2 = D::zero();
        let mut eta_i = D::one();
        let m1 = (mbar - 1.0) / mbar;
        let m2 = (mbar - 2.0) / mbar * m1;
        for i in 0..=6 {
            i1 += (m2 * A2[i] + m1 * A1[i] + A0[i]) * eta_i;
            i2 += (m2 * B2[i] + m1 * B1[i] + B0[i]) * eta_i;
            eta_i *= eta;
        }
        let c1 = (mbar * (eta * 8.0 - eta.powi(2) * 2.0) / (eta - 1.0).powi(4)
            + (D::one() - mbar)
                * (eta * 20.0 - eta.powi(2) * 27.0 + eta.powi(3) * 12.0 - eta.powi(4) * 2.0)
                / ((eta - 1.0) * (eta - 2.0)).powi(2)
            + 1.0)
            .recip();

        // dispersion
        let disp = (-rho1mix * i1 * 2.0 - rho2mix * mbar * c1 * i2) * PI;

        // dipoles
        let m_mix = m.row_sum();
        let sigma_mix = apply_group_count(&m, &sigma.map(|s| s.powi(3)))
            .row_sum()
            .component_div(&m_mix)
            .map(|s| s.cbrt());
        let epsilon_k_mix = apply_group_count(&m, &epsilon_k)
            .row_sum()
            .component_div(&m_mix);
        let mu2 = apply_group_count(groups, &mu.component_mul(&mu))
            .row_sum()
            .component_div(&m_mix)
            * (t_inv * 1e-19 * JOULE.convert_into(KELVIN * KB));
        let m_mix = m_mix.map(|m| if m.re() > 2.0 { D::from(2.0) } else { m });

        let mut phi2 = D::zero();
        let mut phi3 = D::zero();
        for i in 0..N {
            for j in i..N {
                let sigma_ij_3 = ((sigma_mix[i] + sigma_mix[j]) * 0.5).powi(3);
                let mij = (m_mix[i] * m_mix[j]).sqrt();
                let mij1 = (mij - 1.0) / mij;
                let mij2 = mij1 * (mij - 2.0) / mij;
                let eps_ij_t = t_inv * (epsilon_k_mix[i] * epsilon_k_mix[j]).sqrt();
                let c = if i == j { 1.0 } else { 2.0 };
                phi2 -= (density[i] * density[j] * mu2[i] * mu2[j])
                    * pair_integral(mij1, mij2, eta, eps_ij_t)
                    / sigma_ij_3
                    * c;
                for k in j..N {
                    let sigma_ij = (sigma_mix[i] + sigma_mix[j]) * 0.5;
                    let sigma_ik = (sigma_mix[i] + sigma_mix[k]) * 0.5;
                    let sigma_jk = (sigma_mix[j] + sigma_mix[k]) * 0.5;
                    let mijk = (m_mix[i] * m_mix[j] * m_mix[k]).cbrt();
                    let mijk1 = (mijk - 1.0) / mijk;
                    let mijk2 = mijk1 * (mijk - 2.0) / mijk;
                    let c = if i == j && i == k {
                        1.0
                    } else if i == j || i == k || j == k {
                        3.0
                    } else {
                        6.0
                    };
                    phi3 -= (density[i] * density[j] * density[k] * mu2[i] * mu2[j] * mu2[k])
                        * triplet_integral(mijk1, mijk2, eta)
                        / (sigma_ij * sigma_ik * sigma_jk)
                        * c;
                }
            }
        }
        phi2 *= PI;
        phi3 *= PI_SQ_43;
        let mut dipole = phi2 * phi2 / (phi2 - phi3);
        if dipole.re().is_nan() {
            dipole = phi2
        }

        hs + hc + disp + dipole
    }
}

fn apply_group_count<D: DualNum<f64> + Copy, const N: usize>(
    groups: &SMatrix<D, N_GROUPS, N>,
    x: &SVector<D, N_GROUPS>,
) -> SMatrix<D, N_GROUPS, N> {
    let mut ms = *groups;
    ms.column_iter_mut()
        .for_each(|mut s| s.component_mul_assign(x));
    ms
}

fn pair_integral<D: DualNum<f64> + Copy>(mij1: D, mij2: D, eta: D, eps_ij_t: D) -> D {
    let mut eta_i = D::one();
    let mut j = D::zero();
    for (ad, bd) in AD.into_iter().zip(BD) {
        j += (eps_ij_t * (mij2 * bd[2] + mij1 * bd[1] + bd[0])
            + (mij2 * ad[2] + mij1 * ad[1] + ad[0]))
            * eta_i;
        eta_i *= eta;
    }
    j
}

fn triplet_integral<D: DualNum<f64> + Copy>(mij1: D, mij2: D, eta: D) -> D {
    let mut eta_i = D::one();
    let mut j = D::zero();
    for cd in CD {
        j += (mij2 * cd[2] + mij1 * cd[1] + cd[0]) * eta_i;
        eta_i *= eta;
    }
    j
}

#[cfg(test)]
pub mod test {
    use super::{EPSILON_K, GROUPS, GcPcSaftAD, GcPcSaftADParameters, M, MU, SIGMA};
    use crate::gc_pcsaft::{GcPcSaft, GcPcSaftParameters as GcPcSaftEosParameters, GcPcSaftRecord};
    use approx::assert_relative_eq;
    use feos_core::parameter::{ChemicalRecord, SegmentRecord};
    use feos_core::{Contributions::Total, FeosResult, State};
    use nalgebra::{dvector, vector};
    use quantity::{KELVIN, KILO, METER, MOL};
    use std::collections::HashMap;

    pub fn gcpcsaft() -> FeosResult<(GcPcSaftADParameters<f64, 1>, GcPcSaft)> {
        let cr = ChemicalRecord::new(
            Default::default(),
            vec!["CH3".into(), ">C=O".into(), "CH2".into(), "CH3".into()],
            None,
        );
        let segment_records: Vec<_> = M
            .into_iter()
            .zip(SIGMA)
            .zip(EPSILON_K)
            .zip(MU)
            .zip(GROUPS)
            .map(|((((m, sigma), epsilon_k), mu), g)| {
                SegmentRecord::new(
                    g.into(),
                    0.0,
                    GcPcSaftRecord::new(m, sigma, epsilon_k, mu, None),
                )
            })
            .collect();
        let params = GcPcSaftEosParameters::from_segments_hetero(vec![cr], &segment_records, None)?;
        let eos = GcPcSaft::new(params);
        let mut groups = HashMap::new();
        groups.insert("CH3", 2.0);
        groups.insert(">C=O", 1.0);
        groups.insert("CH2", 1.0);
        let mut bonds = HashMap::new();
        bonds.insert(["CH3", ">C=O"], 1.0);
        bonds.insert([">C=O", "CH2"], 1.0);
        bonds.insert(["CH2", "CH3"], 1.0);
        let params = GcPcSaftADParameters::from_groups([&groups], [&bonds]);
        Ok((params, eos))
    }

    #[test]
    fn test_gcpcsaft() -> FeosResult<()> {
        let (params, eos) = gcpcsaft()?;

        let temperature = 300.0 * KELVIN;
        let volume = 2.3 * METER * METER * METER;
        let moles = dvector![1.3] * KILO * MOL;

        let state = State::new_nvt(&&eos, temperature, volume, &moles)?;
        let a_feos = state.residual_molar_helmholtz_energy();
        let mu_feos = state.residual_chemical_potential();
        let p_feos = state.pressure(Total);
        let s_feos = state.residual_molar_entropy();
        let h_feos = state.residual_molar_enthalpy();

        let eos_ad = GcPcSaftAD(params);
        let moles = vector![1.3] * KILO * MOL;
        let state = State::new_nvt(&eos_ad, temperature, volume, &moles)?;
        let a_ad = state.residual_molar_helmholtz_energy();
        let mu_ad = state.residual_chemical_potential();
        let p_ad = state.pressure(Total);
        let s_ad = state.residual_molar_entropy();
        let h_ad = state.residual_molar_enthalpy();

        println!("\nHelmholtz energy density:\n{a_feos}",);
        println!("{a_ad}");
        assert_relative_eq!(a_feos, a_ad, max_relative = 1e-14);

        println!("\nChemical potential:\n{}", mu_feos.get(0));
        println!("{}", mu_ad.get(0));
        assert_relative_eq!(mu_feos.get(0), mu_ad.get(0), max_relative = 1e-14);

        println!("\nPressure:\n{p_feos}");
        println!("{p_ad}");
        assert_relative_eq!(p_feos, p_ad, max_relative = 1e-14);

        println!("\nMolar entropy:\n{s_feos}");
        println!("{s_ad}");
        assert_relative_eq!(s_feos, s_ad, max_relative = 1e-14);

        println!("\nMolar enthalpy:\n{h_feos}");
        println!("{h_ad}");
        assert_relative_eq!(h_feos, h_ad, max_relative = 1e-14);

        Ok(())
    }
}
