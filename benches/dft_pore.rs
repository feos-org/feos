//! Benchmarks for the calculation of density profiles
//! in pores at different conditions.
use criterion::{criterion_group, criterion_main, Criterion};
use feos::gc_pcsaft::{GcPcSaftFunctional, GcPcSaftFunctionalParameters};
use feos::hard_sphere::{FMTFunctional, FMTVersion};
use feos::pcsaft::{PcSaftFunctional, PcSaftParameters};
use feos_core::parameter::{IdentifierOption, Parameter, ParameterHetero};
use feos_core::si::{ANGSTROM, KELVIN, NAV};
use feos_core::{PhaseEquilibrium, State, StateBuilder};
use feos_dft::adsorption::{ExternalPotential, Pore1D, PoreSpecification};
use feos_dft::{DFTSolver, Geometry};
use ndarray::arr1;
use std::sync::Arc;
use typenum::P3;

fn fmt(c: &mut Criterion) {
    let mut group = c.benchmark_group("DFT_pore_fmt");

    let func = Arc::new(FMTFunctional::new(&arr1(&[1.0]), FMTVersion::WhiteBear));
    let pore = Pore1D::new(
        Geometry::Cartesian,
        10.0 * ANGSTROM,
        ExternalPotential::HardWall { sigma_ss: 1.0 },
        None,
        None,
    );
    let bulk = State::new_pure(&func, KELVIN, 0.75 / NAV / ANGSTROM.powi::<P3>()).unwrap();
    group.bench_function("liquid", |b| {
        b.iter(|| pore.initialize(&bulk, None, None).solve(None))
    });
}

fn pcsaft(c: &mut Criterion) {
    let mut group = c.benchmark_group("DFT_pore_pcsaft");
    let parameters = PcSaftParameters::from_json(
        vec!["butane"],
        "./parameters/pcsaft/gross2001.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    let func = Arc::new(PcSaftFunctional::new(Arc::new(parameters)));
    let pore = Pore1D::new(
        Geometry::Cartesian,
        20.0 * ANGSTROM,
        ExternalPotential::LJ93 {
            sigma_ss: 3.0,
            epsilon_k_ss: 100.0,
            rho_s: 0.08,
        },
        None,
        None,
    );
    let vle = PhaseEquilibrium::pure(&func, 300.0 * KELVIN, None, Default::default()).unwrap();
    let bulk = vle.liquid();
    group.bench_function("butane_liquid", |b| {
        b.iter(|| pore.initialize(bulk, None, None).solve(None))
    });
    let bulk = State::new_pure(&func, 300.0 * KELVIN, vle.vapor().density * 0.2).unwrap();
    group.bench_function("butane_vapor", |b| {
        b.iter(|| pore.initialize(&bulk, None, None).solve(None))
    });

    let parameters = PcSaftParameters::from_json(
        vec!["butane", "pentane"],
        "./parameters/pcsaft/gross2001.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    let func = Arc::new(PcSaftFunctional::new(Arc::new(parameters)));
    let vle = PhaseEquilibrium::bubble_point(
        &func,
        300.0 * KELVIN,
        &arr1(&[0.5, 0.5]),
        None,
        None,
        Default::default(),
    )
    .unwrap();
    let bulk = vle.liquid();
    group.bench_function("butane_pentane_liquid", |b| {
        b.iter(|| pore.initialize(bulk, None, None).solve(None))
    });
    let bulk = StateBuilder::new(&func)
        .temperature(300.0 * KELVIN)
        .partial_density(&(&vle.vapor().partial_density * 0.2))
        .build()
        .unwrap();
    group.bench_function("butane_pentane_vapor", |b| {
        b.iter(|| pore.initialize(&bulk, None, None).solve(None))
    });
}

fn gc_pcsaft(c: &mut Criterion) {
    let mut group = c.benchmark_group("DFT_pore_gc_pcsaft");
    group.sample_size(20);

    let parameters = GcPcSaftFunctionalParameters::from_json_segments(
        &["butane"],
        "./parameters/pcsaft/gc_substances.json",
        "./parameters/pcsaft/sauer2014_hetero.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    let func = Arc::new(GcPcSaftFunctional::new(Arc::new(parameters)));
    let pore = Pore1D::new(
        Geometry::Cartesian,
        20.0 * ANGSTROM,
        ExternalPotential::LJ93 {
            sigma_ss: 3.0,
            epsilon_k_ss: 100.0,
            rho_s: 0.08,
        },
        None,
        None,
    );
    let vle = PhaseEquilibrium::pure(&func, 300.0 * KELVIN, None, Default::default()).unwrap();
    let bulk = vle.liquid();
    let solver = DFTSolver::new(None)
        .picard_iteration(None, None, Some(1e-5), None)
        .anderson_mixing(None, None, None, None, None);
    group.bench_function("butane_liquid", |b| {
        b.iter(|| pore.initialize(bulk, None, None).solve(Some(&solver)))
    });
}

criterion_group!(bench, fmt, pcsaft, gc_pcsaft);
criterion_main!(bench);
