//! Benchmarks for the evaluation of the Helmholtz energy function
//! for a given `StateHD` for different types of dual numbers.
//! These should give an idea about the expected slow-down depending
//! on the dual number type used without the overhead of the `State`
//! creation.
use criterion::{criterion_group, criterion_main, Criterion};
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::parameter::{IdentifierOption, Parameter};
use feos_core::{Contributions, DensityInitialization, Derivative, EquationOfState, State};
use ndarray::arr1;
use quantity::si::*;
use std::sync::Arc;

// /// Helper function to create a state for given parameters.
// /// - temperature is 80% of critical temperature,
// /// - volume is critical volume,
// /// - molefracs (or moles) for equimolar mixture.
// fn state_pcsaft(parameters: PcSaftParameters) -> State<PcSaft> {
//     let n = parameters.pure_records.len();
//     let eos = Arc::new(PcSaft::new(Arc::new(parameters)));
//     let moles = Array::from_elem(n, 1.0 / n as f64) * 10.0 * MOL;
//     let cp = State::critical_point(&eos, Some(&moles), None, Default::default()).unwrap();
//     let temperature = 0.8 * cp.temperature;
//     State::new_nvt(&eos, temperature, cp.volume, &moles).unwrap()
// }

// /// Residual Helmholtz energy given an equation of state and a StateHD.
// fn a_res<D: DualNum<f64>, E: EquationOfState>(inp: (&Arc<E>, &StateHD<D>)) -> D
// where
//     (dyn HelmholtzEnergy + 'static): HelmholtzEnergyDual<D>,
// {
//     inp.0.evaluate_residual(inp.1)
// }

// /// Benchmark for evaluation of the Helmholtz energy for different dual number types.
// fn bench_fugacity<E: EquationOfState>(c: &mut Criterion, group_name: &str, state: State<E>) {
//     let mut group = c.benchmark_group(group_name);
//     group.bench_function("a_f64", |b| {
//         b.iter(|| a_res((&state.eos, &state.derive0())))
//     });
//     group.bench_function("a_dual", |b| {
//         b.iter(|| a_res((&state.eos, &state.derive1(Derivative::DV))))
//     });
//     group.bench_function("a_dual2", |b| {
//         b.iter(|| a_res((&state.eos, &state.derive2(Derivative::DV))))
//     });
//     group.bench_function("a_hyperdual", |b| {
//         b.iter(|| {
//             a_res((
//                 &state.eos,
//                 &state.derive2_mixed(Derivative::DV, Derivative::DV),
//             ))
//         })
//     });
//     group.bench_function("a_dual3", |b| {
//         b.iter(|| a_res((&state.eos, &state.derive3(Derivative::DV))))
//     });
// }

/// Benchmark for the PC-SAFT equation of state
fn pcsaft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fugacity");

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
            let params = PcSaftParameters::new_binary(vec![comp1.clone(), comp2.clone()], None);
            let eos = Arc::new(PcSaft::new(Arc::new(params)));
            let state = State::new_npt(&eos, t, p, &moles, DensityInitialization::Liquid).unwrap();
            let state_hd = state.derive1(Derivative::DT);
            let name1 = comp1.identifier.name.as_deref().unwrap();
            let name2 = comp2.identifier.name.as_deref().unwrap();
            let mix = format!("{name1}_{name2}");
            group.bench_function(mix, |b| b.iter(|| eos.evaluate_residual(&state_hd)));
        }
    }

    // // water (4C, polar)
    // let parameters = PcSaftParameters::from_json(
    //     vec!["water_4C_polar"],
    //     "./parameters/pcsaft/rehner2020.json",
    //     None,
    //     IdentifierOption::Name,
    // )
    // .unwrap();
    // bench_dual_numbers(
    //     c,
    //     "dual_numbers_pcsaft_water_4c_polar",
    //     state_pcsaft(parameters),
    // );

    // // methane, ethane, propane
    // let parameters = PcSaftParameters::from_json(
    //     vec!["methane", "ethane", "propane"],
    //     "./parameters/pcsaft/gross2001.json",
    //     None,
    //     IdentifierOption::Name,
    // )
    // .unwrap();
    // bench_dual_numbers(
    //     c,
    //     "dual_numbers_pcsaft_methane_ethane_propane",
    //     state_pcsaft(parameters),
    // );
}

criterion_group!(bench, pcsaft);
criterion_main!(bench);
