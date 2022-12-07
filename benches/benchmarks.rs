use criterion::{criterion_group, criterion_main, Criterion};
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::{
    parameter::{IdentifierOption, Parameter},
    Contributions, EquationOfState, HelmholtzEnergy, HelmholtzEnergyDual, State, StateBuilder,
    StateHD,
};
use ndarray::{arr1, Array1};
use num_dual::{Dual64, DualNum, HyperDual64, StaticMat, StaticVec};
use quantity::{si::*, QuantityArray1};
use std::sync::Arc;

/// State generation using `State::new`
fn state(inp: (&Arc<PcSaft>, SINumber, SINumber, &Array1<f64>)) -> State<SIUnit, PcSaft> {
    State::new(
        inp.0,
        Some(inp.1),
        None,
        Some(inp.2),
        None,
        None,
        None,
        Some(inp.3),
        None,
        None,
        None,
        None,
        feos_core::DensityInitialization::None,
        None,
    )
    .unwrap()
}

/// Residual Helmholtz energy given a StateHD
fn a_res(inp: (&Arc<PcSaft>, SINumber, SINumber, &SIArray1)) -> f64 {
    let t = inp.1.to_reduced(KELVIN).unwrap();
    let v = inp.2.to_reduced(ANGSTROM.powi(3)).unwrap();
    let m = inp.3.to_reduced(MOL).unwrap();
    let s = StateHD::new(t, v, m);
    inp.0.evaluate_residual(&s)
}

/// Residual Helmholtz energy given a StateHD
fn a_res_shd<D: DualNum<f64>>(inp: (&Arc<PcSaft>, &StateHD<D>)) -> D
where
    (dyn HelmholtzEnergy + 'static): HelmholtzEnergyDual<D>,
{
    inp.0.evaluate_residual(inp.1)
}

fn da_dv(inp: (&Arc<PcSaft>, SINumber, SINumber, &SIArray1)) -> f64 {
    let t = Dual64::from_re(inp.1.to_reduced(KELVIN).unwrap());
    let v = Dual64::from_re(inp.2.to_reduced(ANGSTROM.powi(3)).unwrap()).derive();
    let m = inp
        .3
        .to_reduced(MOL)
        .unwrap()
        .iter()
        .map(|&mi| Dual64::from_re(mi))
        .collect();
    let s = StateHD::new(t, v, m);
    inp.0.evaluate_residual(&s).eps[0]
}

fn compressibility(inp: (&Arc<PcSaft>, SINumber, SINumber, &Array1<f64>)) -> f64 {
    State::new(
        inp.0,
        Some(inp.1),
        None,
        Some(inp.2),
        None,
        None,
        None,
        Some(inp.3),
        None,
        None,
        None,
        None,
        feos_core::DensityInitialization::None,
        None,
    )
    .unwrap()
    .compressibility(Contributions::Total)
}

fn compressibility_(state: &State<SIUnit, PcSaft>) -> f64 {
    state.compressibility(Contributions::Total)
}

fn ln_phi(inp: (&Arc<PcSaft>, SINumber, SINumber, &Array1<f64>)) -> Array1<f64> {
    State::new(
        inp.0,
        Some(inp.1),
        None,
        Some(inp.2),
        None,
        None,
        None,
        Some(inp.3),
        None,
        None,
        None,
        None,
        feos_core::DensityInitialization::None,
        None,
    )
    .unwrap()
    .ln_phi()
}

fn da_dni(inp: (&Arc<PcSaft>, SINumber, SINumber, &SIArray1)) -> Array1<f64> {
    let t = Dual64::from_re(inp.1.to_reduced(KELVIN).unwrap());
    let v = Dual64::from_re(inp.2.to_reduced(ANGSTROM.powi(3)).unwrap());
    let m: Array1<_> = inp
        .3
        .to_reduced(MOL)
        .unwrap()
        .iter()
        .map(|&mi| Dual64::from_re(mi))
        .collect();
    Array1::from_shape_fn(inp.3.len(), |i| {
        let mut n = m.clone();
        n[i] = n[i].derive();
        let s = StateHD::new(t, v, n);
        inp.0.evaluate_residual(&s).eps[0]
    })
}

fn c_v(inp: (&Arc<PcSaft>, SINumber, SINumber, &Array1<f64>)) -> SINumber {
    State::new(
        inp.0,
        Some(inp.1),
        None,
        Some(inp.2),
        None,
        None,
        None,
        Some(inp.3),
        None,
        None,
        None,
        None,
        feos_core::DensityInitialization::None,
        None,
    )
    .unwrap()
    .c_v(Contributions::ResidualNvt)
}

fn d2a_dt2(inp: (&Arc<PcSaft>, SINumber, SINumber, &SIArray1)) -> f64 {
    let t = HyperDual64::from_re(inp.1.to_reduced(KELVIN).unwrap()).derive2();
    let v = HyperDual64::from_re(inp.2.to_reduced(ANGSTROM.powi(3)).unwrap());
    let m: Array1<_> = inp
        .3
        .to_reduced(MOL)
        .unwrap()
        .iter()
        .map(|&mi| HyperDual64::from_re(mi))
        .collect();

    let s = StateHD::new(t, v, m);
    inp.0.evaluate_residual(&s).eps1eps2[0]
}

fn molar_volume(inp: (&Arc<PcSaft>, SINumber, SINumber, &Array1<f64>)) -> SIArray1 {
    State::new(
        inp.0,
        Some(inp.1),
        None,
        Some(inp.2),
        None,
        None,
        None,
        Some(inp.3),
        None,
        None,
        None,
        None,
        feos_core::DensityInitialization::None,
        None,
    )
    .unwrap()
    .molar_volume(Contributions::ResidualNvt)
}

fn benchmark_helmholtz_energy(c: &mut Criterion) {
    let parameters = PcSaftParameters::from_json(
        vec!["methane", "ethane", "propane"],
        "./parameters/pcsaft/gross2001.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    let eos = Arc::new(PcSaft::new(Arc::new(parameters)));
    let t = 300.0 * KELVIN;
    let density = 71.18 * KILO * MOL / METER.powi(3);
    let volume = 100.0 * MOL / density;
    let x = arr1(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    let rho_i = &x * density;
    let m = &x * 100.0 * MOL;
    let s = StateHD::new(
        300.0,
        volume.to_reduced(ANGSTROM.powi(3)).unwrap(),
        &x * 100.0,
    );

    let mut group = c.benchmark_group("helmholtz_energy");
    group.bench_function("state", |b| b.iter(|| state((&eos, t, density, &x))));
    group.bench_function("a_res", |b| b.iter(|| a_res((&eos, t, volume, &m))));
    group.bench_function("a_res_shd", |b| b.iter(|| a_res_shd((&eos, &s))));
}

fn benchmark_first_derivative(c: &mut Criterion) {
    let parameters = PcSaftParameters::from_json(
        vec!["methane", "ethane", "propane"],
        "./parameters/pcsaft/gross2001.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    let eos = Arc::new(PcSaft::new(Arc::new(parameters)));
    let t = 300.0 * KELVIN;
    let density = 71.18 * KILO * MOL / METER.powi(3);
    let volume = 100.0 * MOL / density;
    let x = arr1(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    let rho_i = &x * density;
    let m = &x * 100.0 * MOL;

    let s = StateBuilder::new(&eos)
        .temperature(t)
        .partial_density(&rho_i)
        .total_moles(m.sum())
        .build()
        .unwrap();

    let mut dual = c.benchmark_group("first_derivative");
    // group.bench_function("state", |b| b.iter(|| state((&eos, t, density, &x))));
    dual.bench_function("compressibility", |b| {
        b.iter(|| compressibility((&eos, t, density, &x)))
    });
    dual.bench_function("compressibility_c", |b| {
        b.iter(|| compressibility((&eos, t, density, &x)))
    });
    dual.bench_function("da_dv", |b| b.iter(|| da_dv((&eos, t, volume, &m))));
    dual.bench_function("ln_phi", |b| b.iter(|| ln_phi((&eos, t, density, &x))));
    dual.bench_function("da_dni", |b| b.iter(|| da_dni((&eos, t, volume, &m))));
}

fn benchmark_second_derivative(c: &mut Criterion) {
    let parameters = PcSaftParameters::from_json(
        vec!["methane", "ethane", "propane"],
        "./parameters/pcsaft/gross2001.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    let eos = Arc::new(PcSaft::new(Arc::new(parameters)));
    let t = 300.0 * KELVIN;
    let density = 71.18 * KILO * MOL / METER.powi(3);
    let volume = 100.0 * MOL / density;
    let x = arr1(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    let rho_i = &x * density;
    let m = &x * 100.0 * MOL;

    let s = StateBuilder::new(&eos)
        .temperature(t)
        .partial_density(&rho_i)
        .total_moles(m.sum())
        .build()
        .unwrap();

    let mut hyper_dual = c.benchmark_group("second_derivative");
    hyper_dual.bench_function("c_v", |b| b.iter(|| c_v((&eos, t, density, &x))));
    hyper_dual.bench_function("d2a_dt2", |b| b.iter(|| d2a_dt2((&eos, t, volume, &m))));

    hyper_dual.bench_function("molar_volume", |b| {
        b.iter(|| molar_volume((&eos, t, density, &x)))
    });
}

criterion_group!(
    bench,
    benchmark_helmholtz_energy,
    benchmark_first_derivative,
    benchmark_second_derivative
);
criterion_main!(bench);
