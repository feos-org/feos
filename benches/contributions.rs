//! Benchmarks for the evaluation of the first derivative of the
//! Helmholtz energy function for various binary mixtures.
//! The mixtures contain fluids of different polarities that are
//! modeled using additional Helmholtz energy contributions.
//! It is supposed to demonstrate the expected reduction in
//! performance when more complex physical interactions are
//! modeled.
use criterion::{criterion_group, criterion_main, Criterion};
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::parameter::{IdentifierOption, Parameter};
use feos_core::si::*;
use feos_core::{DensityInitialization, Derivative, Residual, State};
use ndarray::arr1;
use std::sync::Arc;

/// Benchmark for the PC-SAFT equation of state
fn pcsaft(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcsaft");

    // non-polar components
    let mut records = PcSaftParameters::from_json(
        vec!["hexane", "heptane"],
        "./parameters/pcsaft/gross2001.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap()
    .pure_records;
    let hexane = records.remove(0);
    let heptane = records.remove(0);

    // dipolar components
    records = PcSaftParameters::from_json(
        vec!["acetone", "dimethyl ether"],
        "./parameters/pcsaft/gross2006.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap()
    .pure_records;
    let acetone = records.remove(0);
    let dme = records.remove(0);

    // quadrupolar components
    records = PcSaftParameters::from_json(
        vec!["carbon dioxide", "acetylene"],
        "./parameters/pcsaft/gross2005_literature.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap()
    .pure_records;
    let co2 = records.remove(0);
    let acetylene = records.remove(0);

    // associating components
    records = PcSaftParameters::from_json(
        vec!["ethanol", "1-propanol"],
        "./parameters/pcsaft/gross2002.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap()
    .pure_records;
    let ethanol = records.remove(0);
    let propanol = records.remove(0);

    let t = 300.0 * KELVIN;
    let p = BAR;
    let moles = arr1(&[1.0, 1.0]) * MOL;
    for comp1 in &[hexane, acetone, co2, ethanol] {
        for comp2 in [&heptane, &dme, &acetylene, &propanol] {
            let params =
                PcSaftParameters::new_binary(vec![comp1.clone(), comp2.clone()], None).unwrap();
            let eos = Arc::new(PcSaft::new(Arc::new(params)));
            let state = State::new_npt(&eos, t, p, &moles, DensityInitialization::Liquid).unwrap();
            let state_hd = state.derive1(Derivative::DT);
            let name1 = comp1.identifier.name.as_deref().unwrap();
            let name2 = comp2.identifier.name.as_deref().unwrap();
            let mix = format!("{name1}_{name2}");
            group.bench_function(mix, |b| b.iter(|| eos.evaluate_residual(&state_hd)));
        }
    }
}

criterion_group!(bench, pcsaft);
criterion_main!(bench);
