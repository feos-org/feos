//! Benchmarks for the evaluation of the Helmholtz energy function
//! for a given `StateHD` for different types of dual numbers.
//! These should give an idea about the expected slow-down depending
//! on the dual number type used without the overhead of the `State`
//! creation.
//!
//! The example system is the binary mixture of CH4/CO2 modelled
//! with the PCP-SAFT equation of state. The considered Helmholtz
//! energy contributions are: hard-sphere, hard-chain, dispersion,
//! and polar (dipolar)..
use criterion::{criterion_group, criterion_main, Criterion};
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::{
    parameter::{IdentifierOption, Parameter},
    EquationOfState, HelmholtzEnergy, HelmholtzEnergyDual, StateHD,
};
use ndarray::arr1;
use num_dual::{Dual3, Dual64, DualNum, HyperDual64};
use quantity::si::*;
use std::sync::Arc;

/// Residual Helmholtz energy given a StateHD
fn a_res<D: DualNum<f64>>(inp: (&Arc<PcSaft>, &StateHD<D>)) -> D
where
    (dyn HelmholtzEnergy + 'static): HelmholtzEnergyDual<D>,
{
    inp.0.evaluate_residual(inp.1)
}

fn benchmark_dual_numbers(c: &mut Criterion) {
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
        PcSaftParameters::new_binary(parameters.pure_records.clone(), Some(k_ij.into()));
    let eos = Arc::new(PcSaft::new(Arc::new(parameters)));

    // 230 K, 50 bar, x0 = 0.15
    let t = 230.0 * KELVIN;
    let density = 24.16896 * KILO * MOL / METER.powi(3);
    let volume = 10.0 * MOL / density;
    let x = arr1(&[0.15, 0.85]);
    let m = &x * 10.0 * MOL;

    let mut group = c.benchmark_group("dual_numbers");

    // real valued evaluation
    let s = StateHD::new(
        t.to_reduced(KELVIN).unwrap(),
        volume.to_reduced(ANGSTROM.powi(3)).unwrap(),
        m.to_reduced(MOL).unwrap(),
    );
    group.bench_function("a_f64", |b| b.iter(|| a_res((&eos, &s))));

    // da_dv - dual number
    let s = StateHD::new(
        Dual64::from_re(t.to_reduced(KELVIN).unwrap()),
        Dual64::from_re(volume.to_reduced(ANGSTROM.powi(3)).unwrap()).derive(),
        m.to_reduced(MOL)
            .unwrap()
            .iter()
            .map(|&mi| Dual64::from_re(mi))
            .collect(),
    );
    group.bench_function("a_dual", |b| b.iter(|| a_res((&eos, &s))));

    // d2a_dv2 - hyperdual number
    let s = StateHD::new(
        HyperDual64::from_re(t.to_reduced(KELVIN).unwrap()).derive1(),
        HyperDual64::from_re(volume.to_reduced(ANGSTROM.powi(3)).unwrap()).derive2(),
        m.to_reduced(MOL)
            .unwrap()
            .iter()
            .map(|&mi| HyperDual64::from_re(mi))
            .collect(),
    );
    group.bench_function("a_hyperdual", |b| b.iter(|| a_res((&eos, &s))));

    // d3a_dv3 - dual3 number
    let s = StateHD::new(
        Dual3::from_re(t.to_reduced(KELVIN).unwrap()),
        Dual3::from_re(volume.to_reduced(ANGSTROM.powi(3)).unwrap()).derive(),
        m.to_reduced(MOL)
            .unwrap()
            .iter()
            .map(|&mi| Dual3::from_re(mi))
            .collect(),
    );
    group.bench_function("a_dual3", |b| b.iter(|| a_res((&eos, &s))));
}

criterion_group!(bench, benchmark_dual_numbers);
criterion_main!(bench);
