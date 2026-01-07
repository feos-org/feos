#![allow(clippy::type_complexity)]
use criterion::{Criterion, criterion_group, criterion_main};
use feos::core::parameter::IdentifierOption;
use feos::core::{Contributions, Residual, State};
use feos::pcsaft::{PcSaft, PcSaftParameters};
use nalgebra::{DVector, dvector};
use quantity::*;

/// Evaluate a property of a state given the EoS, the property to compute,
/// temperature, volume, moles, and the contributions to consider.
fn property<E: Residual, T, F: Fn(&State<E>, Contributions) -> T>(
    (eos, property, t, v, n, contributions): (
        &E,
        F,
        Temperature,
        Volume,
        &Moles<DVector<f64>>,
        Contributions,
    ),
) -> T {
    let state = State::new_nvt(eos, t, v, n).unwrap();
    property(&state, contributions)
}

/// Evaluate a property with of a state given the EoS, the property to compute,
/// temperature, volume, moles.
fn property_no_contributions<E: Residual, T, F: Fn(&State<E>) -> T>(
    (eos, property, t, v, n): (&E, F, Temperature, Volume, &Moles<DVector<f64>>),
) -> T {
    let state = State::new_nvt(eos, t, v, n).unwrap();
    property(&state)
}

fn properties_pcsaft(c: &mut Criterion) {
    let parameters = PcSaftParameters::from_json(
        vec!["methane", "ethane", "propane"],
        "../../parameters/pcsaft/gross2001.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    let eos = PcSaft::new(parameters);
    let t = 300.0 * KELVIN;
    let density = 71.18 * KILO * MOL / METER.powi::<3>();
    let v = 100.0 * MOL / density;
    let x = dvector![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let m = &x * 100.0 * MOL;

    let mut group = c.benchmark_group("state_properties_pcsaft_methane_ethane_propane");
    group.bench_function("a", |b| {
        b.iter(|| property_no_contributions((&&eos, State::residual_helmholtz_energy, t, v, &m)))
    });
    group.bench_function("compressibility", |b| {
        b.iter(|| {
            property((
                &&eos,
                State::compressibility,
                t,
                v,
                &m,
                Contributions::Total,
            ))
        })
    });
    group.bench_function("ln_phi", |b| {
        b.iter(|| property_no_contributions((&&eos, State::ln_phi, t, v, &m)))
    });
    group.bench_function("c_v", |b| {
        b.iter(|| {
            property_no_contributions((
                &&eos,
                State::residual_molar_isochoric_heat_capacity,
                t,
                v,
                &m,
            ))
        })
    });
    group.bench_function("partial_molar_volume", |b| {
        b.iter(|| property_no_contributions((&&eos, State::partial_molar_volume, t, v, &m)))
    });
}

fn properties_pcsaft_polar(c: &mut Criterion) {
    let parameters = PcSaftParameters::from_json(
        vec!["acetone", "butanal", "dimethyl ether"],
        "../../parameters/pcsaft/gross2006.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    let eos = PcSaft::new(parameters);
    let t = 300.0 * KELVIN;
    let density = 71.18 * KILO * MOL / METER.powi::<3>();
    let v = 100.0 * MOL / density;
    let x = dvector![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let m = &x * 100.0 * MOL;

    let mut group = c.benchmark_group("state_properties_pcsaft_polar");
    group.bench_function("a", |b| {
        b.iter(|| property_no_contributions((&&eos, State::residual_helmholtz_energy, t, v, &m)))
    });
    group.bench_function("compressibility", |b| {
        b.iter(|| {
            property((
                &&eos,
                State::compressibility,
                t,
                v,
                &m,
                Contributions::Total,
            ))
        })
    });
    group.bench_function("ln_phi", |b| {
        b.iter(|| property_no_contributions((&&eos, State::ln_phi, t, v, &m)))
    });
    group.bench_function("c_v", |b| {
        b.iter(|| {
            property_no_contributions((
                &&eos,
                State::residual_molar_isochoric_heat_capacity,
                t,
                v,
                &m,
            ))
        })
    });
    group.bench_function("partial_molar_volume", |b| {
        b.iter(|| property_no_contributions((&&eos, State::partial_molar_volume, t, v, &m)))
    });
}

criterion_group!(bench, properties_pcsaft, properties_pcsaft_polar);
criterion_main!(bench);
