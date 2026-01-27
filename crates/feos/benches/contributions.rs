//! Benchmarks for the evaluation of the first derivative of the
//! Helmholtz energy function for various binary mixtures.
//! The mixtures contain fluids of different polarities that are
//! modeled using additional Helmholtz energy contributions.
//! It is supposed to demonstrate the expected reduction in
//! performance when more complex physical interactions are
//! modeled.
use ::num_dual::Dual64;
use criterion::{Criterion, criterion_group, criterion_main};
use feos::core::parameter::IdentifierOption;
use feos::core::{DensityInitialization, Residual, State};
use feos::pcsaft::{PcSaft, PcSaftAssociationRecord, PcSaftParameters, PcSaftRecord};
use feos_core::ReferenceSystem;
use feos_core::parameter::PureRecord;
use nalgebra::dvector;
use num_dual::Dual;
use quantity::*;

type Pure = PureRecord<PcSaftRecord, PcSaftAssociationRecord>;

/// Benchmark for the PC-SAFT equation of state
fn pcsaft(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcsaft");

    // non-polar components
    let mut records = Pure::from_json(
        &["hexane", "heptane"],
        "../../parameters/pcsaft/gross2001.json",
        IdentifierOption::Name,
    )
    .unwrap();
    let hexane = records.remove(0);
    let heptane = records.remove(0);

    // dipolar components
    records = Pure::from_json(
        &["acetone", "dimethyl ether"],
        "../../parameters/pcsaft/gross2006.json",
        IdentifierOption::Name,
    )
    .unwrap();
    let acetone = records.remove(0);
    let dme = records.remove(0);

    // quadrupolar components
    records = Pure::from_json(
        &["carbon dioxide", "acetylene"],
        "../../parameters/pcsaft/gross2005_literature.json",
        IdentifierOption::Name,
    )
    .unwrap();
    let co2 = records.remove(0);
    let acetylene = records.remove(0);

    // associating components
    records = Pure::from_json(
        &["ethanol", "1-propanol"],
        "../../parameters/pcsaft/gross2002.json",
        IdentifierOption::Name,
    )
    .unwrap();
    let ethanol = records.remove(0);
    let propanol = records.remove(0);

    let t = 300.0 * KELVIN;
    let p = BAR;
    let moles = dvector![1.0, 1.0] * MOL;
    for comp1 in &[hexane, acetone, co2, ethanol] {
        for comp2 in [&heptane, &dme, &acetylene, &propanol] {
            let params =
                PcSaftParameters::new_binary([comp1.clone(), comp2.clone()], None, vec![]).unwrap();
            let eos = PcSaft::new(params);
            let state =
                State::new_npt(&&eos, t, p, &moles, Some(DensityInitialization::Liquid)).unwrap();
            let temperature = Dual64::from(state.temperature.into_reduced()).derivative();
            let molar_volume = Dual::from(1.0 / state.density.into_reduced());
            let moles = state.moles().unwrap().to_reduced().map(Dual::from);
            // let state_hd = state.derive1(Derivative::DT);
            let name1 = comp1.identifier.name.as_deref().unwrap();
            let name2 = comp2.identifier.name.as_deref().unwrap();
            let mix = format!("{name1}_{name2}");
            group.bench_function(mix, |b| {
                b.iter(|| (&eos).residual_helmholtz_energy(temperature, molar_volume, &moles))
            });
        }
    }
}

criterion_group!(bench, pcsaft);
criterion_main!(bench);
