use criterion::{criterion_group, criterion_main, Criterion};
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::{
    parameter::{IdentifierOption, Parameter},
    Contributions, EquationOfState, State,
};
use ndarray::arr1;
use quantity::si::*;
use std::sync::Arc;

type S = State<SIUnit, PcSaft>;

/// Evaluate a property of a state given the EoS, the property to compute,
/// temperature, volume, moles, and the contributions to consider.
fn property<E: EquationOfState, T, F: Fn(&State<SIUnit, E>, Contributions) -> T>(
    (eos, property, t, v, n, contributions): (
        &Arc<E>,
        F,
        SINumber,
        SINumber,
        &SIArray1,
        Contributions,
    ),
) -> T {
    let state = State::new_nvt(eos, t, v, n).unwrap();
    property(&state, contributions)
}

/// Evaluate a property with of a state given the EoS, the property to compute,
/// temperature, volume, moles.
fn property_no_contributions<E: EquationOfState, T, F: Fn(&State<SIUnit, E>) -> T>(
    (eos, property, t, v, n): (&Arc<E>, F, SINumber, SINumber, &SIArray1),
) -> T {
    let state = State::new_nvt(eos, t, v, n).unwrap();
    property(&state)
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
    let v = 100.0 * MOL / density;
    let x = arr1(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    let m = &x * 100.0 * MOL;

    let mut group = c.benchmark_group("state_properties_pcsaft_methane_ethane_propane");
    group.bench_function("a", |b| {
        b.iter(|| property((&eos, S::helmholtz_energy, t, v, &m, Contributions::Total)))
    });
    group.bench_function("compressibility", |b| {
        b.iter(|| property((&eos, S::compressibility, t, v, &m, Contributions::Total)))
    });
    group.bench_function("ln_phi", |b| {
        b.iter(|| property_no_contributions((&eos, S::ln_phi, t, v, &m)))
    });
    group.bench_function("c_v", |b| {
        b.iter(|| property((&eos, S::c_v, t, v, &m, Contributions::ResidualNvt)))
    });
    group.bench_function("molar_volume", |b| {
        b.iter(|| property((&eos, S::molar_volume, t, v, &m, Contributions::ResidualNvt)))
    });
}

criterion_group!(bench, properties_pcsaft);
criterion_main!(bench);
