//! Benchmarks for the evaluation of the Helmholtz energy function
//! for a given `StateHD` for different types of dual numbers.
//! These should give an idea about the expected slow-down depending
//! on the dual number type used without the overhead of the `State`
//! creation.
use criterion::{criterion_group, criterion_main, Criterion};
use feos::hard_sphere::HardSphereProperties;
#[cfg(feature = "pcsaft")]
use feos::pcsaft::{PcSaft, PcSaftParameters};
#[cfg(feature = "saftvrmie")]
use feos::saftvrmie::{SaftVRMie, SaftVRMieParameters};
use feos_core::si::*;
use feos_core::{
    parameter::{IdentifierOption, Parameter},
    Derivative, Residual, State, StateHD,
};
use ndarray::{arr1, Array, ScalarOperand};
use num_dual::{Dual2_64, Dual3_64, Dual64, DualNum, HyperDual64};
use std::sync::Arc;
use typenum::P3;

/// Helper function to create a state for given parameters.
/// - temperature is 80% of critical temperature,
/// - volume is critical volume,
/// - molefracs (or moles) for equimolar mixture.
#[cfg(feature = "pcsaft")]
fn state_pcsaft(parameters: PcSaftParameters) -> State<PcSaft> {
    let n = parameters.pure_records.len();
    let eos = Arc::new(PcSaft::new(Arc::new(parameters)));
    let moles = Array::from_elem(n, 1.0 / n as f64) * 10.0 * MOL;
    let cp = State::critical_point(&eos, Some(&moles), None, Default::default()).unwrap();
    let temperature = 0.8 * cp.temperature;
    State::new_nvt(&eos, temperature, cp.volume, &moles).unwrap()
}

#[cfg(feature = "saftvrmie")]
fn state_saftvrmie(parameters: &Arc<SaftVRMieParameters>) -> State<SaftVRMie> {
    use feos::saftvrmie::{SaftVRMie, SaftVRMieParameters};

    let n = parameters.pure_records.len();
    let eos = Arc::new(SaftVRMie::new(parameters.clone()));
    let moles = Array::from_elem(n, 1.0 / n as f64) * 10.0 * MOL;
    let cp = State::critical_point(&eos, Some(&moles), None, Default::default()).unwrap();
    let temperature = 0.8 * cp.temperature;
    State::new_nvt(&eos, temperature, cp.volume, &moles).unwrap()
}

/// Residual Helmholtz energy given an equation of state and a StateHD.
fn a_res<D: DualNum<f64> + Copy + ScalarOperand, E: Residual>(inp: (&Arc<E>, &StateHD<D>)) -> D {
    inp.0.residual_helmholtz_energy(inp.1)
}

fn d_hs<D: DualNum<f64> + Copy, P: HardSphereProperties>(inp: (&P, D)) -> D {
    inp.0.hs_diameter(inp.1)[0]
}

/// Benchmark for evaluation of the hard sphere diameter
fn bench_diameter<P: HardSphereProperties>(
    c: &mut Criterion,
    group_name: &str,
    temperature: f64,
    parameters: &P,
) {
    let mut group = c.benchmark_group(group_name);
    group.bench_function("d_f64", |b| b.iter(|| d_hs((parameters, temperature))));
    group.bench_function("d_dual", |b| {
        b.iter(|| d_hs((parameters, Dual64::from_re(temperature).derivative())))
    });
    group.bench_function("d_dual2", |b| {
        b.iter(|| d_hs((parameters, Dual2_64::from_re(temperature).derivative())))
    });
    group.bench_function("d_hyperdual", |b| {
        b.iter(|| d_hs((parameters, HyperDual64::from_re(temperature).derivative1())))
    });
    group.bench_function("d_dual3", |b| {
        b.iter(|| d_hs((parameters, Dual3_64::from_re(temperature).derivative())))
    });
}

/// Benchmark for evaluation of the Helmholtz energy for different dual number types.
fn bench_dual_numbers<E: Residual>(c: &mut Criterion, group_name: &str, state: State<E>) {
    let mut group = c.benchmark_group(group_name);
    group.bench_function("a_f64", |b| {
        b.iter(|| a_res((&state.eos, &state.derive0())))
    });
    group.bench_function("a_dual", |b| {
        b.iter(|| a_res((&state.eos, &state.derive1(Derivative::DV))))
    });
    group.bench_function("a_dual2", |b| {
        b.iter(|| a_res((&state.eos, &state.derive2(Derivative::DV))))
    });
    group.bench_function("a_hyperdual", |b| {
        b.iter(|| {
            a_res((
                &state.eos,
                &state.derive2_mixed(Derivative::DV, Derivative::DV),
            ))
        })
    });
    group.bench_function("a_dual3", |b| {
        b.iter(|| a_res((&state.eos, &state.derive3(Derivative::DV))))
    });
}

/// Benchmark for the PC-SAFT equation of state
#[cfg(feature = "pcsaft")]
fn pcsaft(c: &mut Criterion) {
    // methane
    let parameters = PcSaftParameters::from_json(
        vec!["methane"],
        "./parameters/pcsaft/gross2001.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    bench_dual_numbers(c, "dual_numbers_pcsaft_methane", state_pcsaft(parameters));

    // water (4C, polar)
    let parameters = PcSaftParameters::from_json(
        vec!["water_4C_polar"],
        "./parameters/pcsaft/rehner2020.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    bench_dual_numbers(
        c,
        "dual_numbers_pcsaft_water_4c_polar",
        state_pcsaft(parameters),
    );

    // methane, ethane, propane
    let parameters = PcSaftParameters::from_json(
        vec!["methane", "ethane", "propane"],
        "./parameters/pcsaft/gross2001.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    bench_dual_numbers(
        c,
        "dual_numbers_pcsaft_methane_ethane_propane",
        state_pcsaft(parameters),
    );
}

/// Benchmark for the SAFT VR Mie equation of state
#[cfg(feature = "saftvrmie")]
fn saftvrmie(c: &mut Criterion) {
    use feos::saftvrmie::{ethane, methane};

    let parameters = Arc::new(methane());
    // bench_dual_numbers(
    //     c,
    //     "dual_numbers_saftvrmie_methane",
    //     state_saftvrmie(&parameters),
    // );
    bench_diameter(
        c,
        "dual_numbers_diameter_saftvrmie_methane",
        200.0,
        parameters.as_ref(),
    );
}

/// Benchmark for the PC-SAFT equation of state.
/// Binary system of methane and co2 used to model biogas.
#[cfg(feature = "pcsaft")]
fn methane_co2_pcsaft(c: &mut Criterion) {
    let parameters = PcSaftParameters::from_multiple_json(
        &[
            (vec!["methane"], "./parameters/pcsaft/gross2001.json"),
            (
                vec!["carbon dioxide"],
                "./parameters/pcsaft/gross2005_fit.json",
            ),
        ],
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    let k_ij = -0.0192211646;
    let parameters =
        PcSaftParameters::new_binary(parameters.pure_records, Some(k_ij.into())).unwrap();
    let eos = Arc::new(PcSaft::new(Arc::new(parameters)));

    // 230 K, 50 bar, x0 = 0.15
    let temperature = 230.0 * KELVIN;
    let density = 24.16896 * KILO * MOL / METER.powi::<P3>();
    let volume = 10.0 * MOL / density;
    let x = arr1(&[0.15, 0.85]);
    let moles = &x * 10.0 * MOL;
    let state = State::new_nvt(&eos, temperature, volume, &moles).unwrap();
    bench_dual_numbers(c, "dual_numbers_pcsaft_methane_co2", state);
}

#[cfg(feature = "pcsaft")]
criterion_group!(bench, pcsaft, methane_co2_pcsaft);
#[cfg(feature = "saftvrmie")]
criterion_group!(bench, saftvrmie);
criterion_main!(bench);
