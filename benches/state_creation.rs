use criterion::{criterion_group, criterion_main, Criterion};
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::{
    parameter::{IdentifierOption, Parameter},
    DensityInitialization, PhaseEquilibrium, State,
};
use ndarray::arr1;
use quantity::si::*;
use std::sync::Arc;

/// Evaluate NPT constructor
fn npt(
    (eos, t, p, n, rho0): (
        &Arc<PcSaft>,
        SINumber,
        SINumber,
        &SIArray1,
        DensityInitialization<SIUnit>,
    ),
) {
    State::new_npt(eos, t, p, n, rho0).unwrap();
}

/// Evaluate critical point constructor
fn critical_point((eos, n): (&Arc<PcSaft>, Option<&SIArray1>)) {
    State::critical_point(eos, n, None, Default::default()).unwrap();
}

fn tp_flash((eos, t, p, feed): (&Arc<PcSaft>, SINumber, SINumber, &SIArray1)) {
    PhaseEquilibrium::tp_flash(eos, t, p, feed, None, Default::default(), None).unwrap();
}

fn states_pcsaft(c: &mut Criterion) {
    let parameters = PcSaftParameters::from_json(
        vec!["methane", "ethane", "propane"],
        "./parameters/pcsaft/gross2001.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    let eos = Arc::new(PcSaft::new(Arc::new(parameters)));
    let t = 300.0 * KELVIN;
    let x = arr1(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    let n = &x * 100.0 * MOL;
    let p = BAR;

    let mut group = c.benchmark_group("methane_ethane_propane");
    group.bench_function("NPT", |b| {
        b.iter(|| npt((&eos, t, p, &n, DensityInitialization::None)))
    });
    group.bench_function("critical_point", |b| {
        b.iter(|| critical_point((&eos, Some(&n))))
    });
    group.bench_function("tp_flash", |b| b.iter(|| tp_flash((&eos, 315.0 * KELVIN, 72.0 * BAR, &n))));
}

criterion_group!(bench, states_pcsaft);
criterion_main!(bench);
