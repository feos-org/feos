use criterion::{criterion_group, criterion_main, Criterion};
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::{
    parameter::{IdentifierOption, Parameter},
    Contributions, State,
};
use ndarray::{arr1, Array1};
use quantity::si::*;
use std::sync::Arc;

fn helmholtz_energy(inp: (&Arc<PcSaft>, SINumber, SINumber, &SIArray1)) -> f64 {
    State::new(
        inp.0,
        Some(inp.1),
        Some(inp.2),
        None,
        None,
        None,
        Some(inp.3),
        None,
        None,
        None,
        None,
        None,
        feos_core::DensityInitialization::None,
        None,
    )
    .unwrap()
    .helmholtz_energy(Contributions::Total)
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

fn properties_pcsaft(c: &mut Criterion) {
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
    let m = &x * 100.0 * MOL;

    let mut group = c.benchmark_group("PC-SAFT methane + ethane + propane");
    group.bench_function("a", |b| b.iter(|| helmholtz_energy((&eos, t, volume, &m))));
    group.bench_function("compressibility", |b| {
        b.iter(|| compressibility((&eos, t, density, &x)))
    });
    group.bench_function("ln_phi", |b| b.iter(|| ln_phi((&eos, t, density, &x))));
    group.bench_function("c_v", |b| b.iter(|| c_v((&eos, t, density, &x))));
    group.bench_function("molar_volume", |b| {
        b.iter(|| molar_volume((&eos, t, density, &x)))
    });
}

criterion_group!(bench, properties_pcsaft);
criterion_main!(bench);
