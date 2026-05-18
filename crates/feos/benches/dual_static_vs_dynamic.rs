//! Compare statically sized and dynamically sized vector dual numbers for a
//! PC-SAFT-like Helmholtz energy density evaluation.
//!
//! This intentionally copies the optimized pure-component PC-SAFT expressions
//! instead of calling the production implementation: the production `Residual`
//! stack requires `D: Copy`, while dynamic dual vectors (`DualDVec64`) are not
//! `Copy`. The copied expression is close enough to expose the arithmetic and
//! allocation cost of realistic PC-SAFT parameter derivatives.

use criterion::{Criterion, criterion_group, criterion_main};
use num_dual::{DualDVec64, DualNum, DualSVec64};
use std::f64::consts::{FRAC_PI_6, PI};
use std::hint::black_box;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const PI_SQ_43: f64 = 4.0 / 3.0 * PI * PI;

// PC-SAFT parameters for an associating, polar pure component-ish model:
// m, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab, na, nb
const PARAMS: [f64; 8] = [1.5, 3.4, 180.0, 2.2, 0.03, 2500.0, 2.0, 1.0];
const TEMPERATURE: f64 = 300.0;
const DENSITY: f64 = 0.01;

// Dispersion coefficients copied from pcsaft/eos/dispersion.rs.
const A0: [f64; 7] = [
    0.91056314451539,
    0.63612814494991,
    2.68613478913903,
    -26.5473624914884,
    97.7592087835073,
    -159.591540865600,
    91.2977740839123,
];
const A1: [f64; 7] = [
    -0.30840169182720,
    0.18605311591713,
    -2.50300472586548,
    21.4197936296668,
    -65.2558853303492,
    83.3186804808856,
    -33.7469229297323,
];
const A2: [f64; 7] = [
    -0.09061483509767,
    0.45278428063920,
    0.59627007280101,
    -1.72418291311787,
    -4.13021125311661,
    13.7766318697211,
    -8.67284703679646,
];
const B0: [f64; 7] = [
    0.72409469413165,
    2.23827918609380,
    -4.00258494846342,
    -21.00357681484648,
    26.8556413626615,
    206.5513384066188,
    -355.60235612207947,
];
const B1: [f64; 7] = [
    -0.57554980753450,
    0.69950955214436,
    3.89256733895307,
    -17.21547164777212,
    192.6722644652495,
    -161.8264616487648,
    -165.2076934555607,
];
const B2: [f64; 7] = [
    0.09768831158356,
    -0.25575749816100,
    -9.15585615297321,
    20.64207597439724,
    -38.80443005206285,
    93.6267740770146,
    -29.66690558514725,
];

// Dipole coefficients copied from pcsaft/eos/polar.rs.
const AD: [[f64; 3]; 5] = [
    [0.30435038064, 0.95346405973, -1.16100802773],
    [-0.13585877707, -1.83963831920, 4.52586067320],
    [1.44933285154, 2.01311801180, 0.97512223853],
    [0.35569769252, -7.37249576667, -12.2810377713],
    [-2.06533084541, 8.23741345333, 5.93975747420],
];
const BD: [[f64; 3]; 5] = [
    [0.21879385627, -0.58731641193, 3.48695755800],
    [-1.18964307357, 1.24891317047, -14.9159739347],
    [1.16268885692, -0.50852797392, 15.3720218600],
    [0.0; 3],
    [0.0; 3],
];
const CD: [[f64; 3]; 4] = [
    [-0.06467735252, -0.95208758351, -0.62609792333],
    [0.19758818347, 2.99242575222, 1.29246858189],
    [-0.80875619458, -2.38026356489, 1.65427830900],
    [0.69028490492, -0.27012609786, -3.43967436378],
];

fn helmholtz_energy_density_non_assoc<D>(
    m: D,
    sigma: D,
    epsilon_k: D,
    mu: D,
    temperature: D,
    density: D,
) -> (D, [D; 2])
where
    D: DualNum<f64> + Clone,
{
    // temperature dependent segment diameter
    let diameter =
        sigma.clone() * (D::one() - (epsilon_k.clone() * -3.0 / temperature.clone()).exp() * 0.12);

    let eta = m.clone() * density.clone() * diameter.clone().powi(3) * FRAC_PI_6;
    let eta2 = eta.clone() * eta.clone();
    let eta3 = eta2.clone() * eta.clone();
    let eta_m1 = (D::one() - eta.clone()).recip();
    let eta_m2 = eta_m1.clone() * eta_m1.clone();
    let etas = [
        D::one(),
        eta.clone(),
        eta2.clone(),
        eta3.clone(),
        eta2.clone() * eta2.clone(),
        eta2.clone() * eta3.clone(),
        eta3.clone() * eta3.clone(),
    ];

    // hard sphere
    let hs =
        m.clone() * density.clone() * (eta.clone() * 4.0 - eta2.clone() * 3.0) * eta_m2.clone();

    // hard chain
    let g = (D::one() - eta.clone() * 0.5) * eta_m1.clone() * eta_m2.clone();
    let hc = -(density.clone() * (m.clone() - 1.0) * g.ln());

    // dispersion
    let e = epsilon_k.clone() / temperature.clone();
    let s3 = sigma.clone().powi(3);
    let mut i1 = D::zero();
    let mut i2 = D::zero();
    let m1 = (m.clone() - 1.0) / m.clone();
    let m2 = (m.clone() - 2.0) / m.clone();
    for i in 0..7 {
        i1 += (m1.clone() * (m2.clone() * A2[i] + A1[i]) + A0[i]) * etas[i].clone();
        i2 += (m1.clone() * (m2.clone() * B2[i] + B1[i]) + B0[i]) * etas[i].clone();
    }
    let c1 =
        (m.clone() * (eta.clone() * 8.0 - eta2.clone() * 2.0) * eta_m2.clone() * eta_m2.clone()
            + 1.0
            - (m.clone() - 1.0)
                * (eta.clone() * 20.0 - eta2.clone() * 27.0 + eta3.clone() * 12.0
                    - eta2.clone() * eta2.clone() * 2.0)
                / ((eta.clone() - 1.0) * (eta.clone() - 2.0)).powi(2))
        .recip();
    let i = i1 * 2.0 + c1 * i2 * m.clone() * e.clone();
    let disp =
        -(density.clone() * density.clone() * m.clone().powi(2) * e.clone() * s3.clone() * i * PI);

    // dipoles
    let mu2 = mu.clone().powi(2) / (m.clone() * temperature * 1.380649e-4);
    let m_dipole = if m.re() > 2.0 {
        D::from(2.0)
    } else {
        m.clone()
    };
    let m1 = (m_dipole.clone() - 1.0) / m_dipole.clone();
    let m2 = m1.clone() * (m_dipole.clone() - 2.0) / m_dipole;
    let mut j1 = D::zero();
    let mut j2 = D::zero();
    for i in 0..5 {
        let a = m2.clone() * AD[i][2] + m1.clone() * AD[i][1] + AD[i][0];
        let b = m2.clone() * BD[i][2] + m1.clone() * BD[i][1] + BD[i][0];
        j1 += (a + b * e.clone()) * etas[i].clone();
        if i < 4 {
            j2 += (m2.clone() * CD[i][2] + m1.clone() * CD[i][1] + CD[i][0]) * etas[i].clone();
        }
    }

    // mu is factored out of these expressions to deal with the case where mu=0
    let phi2 = -(density.clone() * density.clone() * j1 / s3.clone() * PI);
    let phi3 = -(density.clone() * density.clone() * density * j2 / s3 * PI_SQ_43);
    let dipole = phi2.clone() * phi2.clone() * mu2.clone() * mu2.clone() / (phi2 - phi3 * mu2);

    (hs + hc + disp + dipole, [eta, eta_m1])
}

fn helmholtz_energy_density<D>(parameters: &[D; 8], temperature: D, density: D) -> D
where
    D: DualNum<f64> + Clone,
{
    let [m, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab, na, nb] =
        parameters.each_ref().map(Clone::clone);
    let (non_assoc, [eta, eta_m1]) = helmholtz_energy_density_non_assoc(
        m,
        sigma.clone(),
        epsilon_k,
        mu,
        temperature.clone(),
        density.clone(),
    );

    // association
    let delta_assoc = ((epsilon_k_ab / temperature).exp() - 1.0) * sigma.powi(3) * kappa_ab;
    let k = eta * eta_m1.clone();
    let delta = (k.clone() * (k * 0.5 + 1.5) + 1.0) * eta_m1 * delta_assoc;
    let rhoa = na * density.clone();
    let rhob = nb * density;
    let aux = (rhoa.clone() - rhob.clone()) * delta.clone() + 1.0;
    let sqrt = (aux.clone() * aux + rhob.clone() * delta.clone() * 4.0).sqrt();
    let xa = (sqrt.clone() + 1.0 + (rhob.clone() - rhoa.clone()) * delta.clone()).recip() * 2.0;
    let xb = (sqrt + 1.0 - (rhob.clone() - rhoa.clone()) * delta).recip() * 2.0;
    let assoc =
        rhoa * (xa.clone().ln() - xa * 0.5 + 0.5) + rhob * (xb.clone().ln() - xb * 0.5 + 0.5);

    non_assoc + assoc
}

fn static_parameters<const P: usize>(params: [f64; 8]) -> [DualSVec64<P>; 8] {
    std::array::from_fn(|i| {
        let x = DualSVec64::<P>::from_re(params[i]);
        if i < P { x.derivative(i) } else { x }
    })
}

fn dynamic_parameters(params: [f64; 8], p: usize) -> [DualDVec64; 8] {
    std::array::from_fn(|i| {
        let x = DualDVec64::from_re(params[i]);
        if i < p { x.derivative(p, i) } else { x }
    })
}

fn eval_static<const P: usize>(params: [f64; 8], temperature: f64, density: f64) -> DualSVec64<P> {
    helmholtz_energy_density(
        &static_parameters::<P>(params),
        DualSVec64::<P>::from_re(temperature),
        DualSVec64::<P>::from_re(density),
    )
}

fn eval_dynamic(params: [f64; 8], temperature: f64, density: f64, p: usize) -> DualDVec64 {
    helmholtz_energy_density(
        &dynamic_parameters(params, p),
        DualDVec64::from_re(temperature),
        DualDVec64::from_re(density),
    )
}

fn bench_pair<const P: usize>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    group.bench_function(format!("static_p{P}"), |b| {
        b.iter(|| {
            black_box(eval_static::<P>(
                black_box(PARAMS),
                black_box(TEMPERATURE),
                black_box(DENSITY),
            ))
        })
    });
    group.bench_function(format!("dynamic_p{P}"), |b| {
        b.iter(|| {
            black_box(eval_dynamic(
                black_box(PARAMS),
                black_box(TEMPERATURE),
                black_box(DENSITY),
                P,
            ))
        })
    });
}

fn static_vs_dynamic_pcsaft(c: &mut Criterion) {
    let mut group = c.benchmark_group("dual_static_vs_dynamic_pcsaft_helmholtz");
    bench_pair::<1>(&mut group);
    bench_pair::<2>(&mut group);
    bench_pair::<3>(&mut group);
    bench_pair::<4>(&mut group);
    bench_pair::<6>(&mut group);
    bench_pair::<8>(&mut group);
    group.finish();
}

criterion_group!(benches, static_vs_dynamic_pcsaft);
criterion_main!(benches);
