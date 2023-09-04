#![allow(clippy::type_complexity)]
use criterion::{criterion_group, criterion_main, Criterion};
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::si::*;
use feos_core::{
    parameter::{IdentifierOption, Parameter},
    Contributions, DensityInitialization, PhaseEquilibrium, Residual, State, TPSpec,
};
use ndarray::{Array, Array1};
use std::sync::Arc;

/// Evaluate NPT constructor
fn npt<E: Residual>(
    (eos, t, p, n, rho0): (
        &Arc<E>,
        Temperature,
        Pressure,
        &Moles<Array1<f64>>,
        DensityInitialization,
    ),
) {
    State::new_npt(eos, t, p, n, rho0).unwrap();
}

/// Evaluate critical point constructor
fn critical_point<E: Residual>((eos, n): (&Arc<E>, Option<&Moles<Array1<f64>>>)) {
    State::critical_point(eos, n, None, Default::default()).unwrap();
}

/// Evaluate critical point constructor for binary systems at given T or p
fn critical_point_binary<E: Residual, TP>((eos, tp): (&Arc<E>, TP))
where
    TPSpec: From<TP>,
{
    State::critical_point_binary(eos, tp, None, None, Default::default()).unwrap();
}

/// VLE for pure substance for given temperature or pressure
fn pure<E: Residual, TP>((eos, t_or_p): (&Arc<E>, TP))
where
    TPSpec: From<TP>,
{
    PhaseEquilibrium::pure(eos, t_or_p, None, Default::default()).unwrap();
}

/// Evaluate temperature, pressure flash.
fn tp_flash<E: Residual>((eos, t, p, feed): (&Arc<E>, Temperature, Pressure, &Moles<Array1<f64>>)) {
    PhaseEquilibrium::tp_flash(eos, t, p, feed, None, Default::default(), None).unwrap();
}

fn bubble_point<E: Residual>((eos, t, x): (&Arc<E>, Temperature, &Array1<f64>)) {
    PhaseEquilibrium::bubble_point(
        eos,
        t,
        x,
        None,
        None,
        (Default::default(), Default::default()),
    )
    .unwrap();
}

fn dew_point<E: Residual>((eos, t, y): (&Arc<E>, Temperature, &Array1<f64>)) {
    PhaseEquilibrium::dew_point(
        eos,
        t,
        y,
        None,
        None,
        (Default::default(), Default::default()),
    )
    .unwrap();
}

fn bench_states<E: Residual>(c: &mut Criterion, group_name: &str, eos: &Arc<E>) {
    let ncomponents = eos.components();
    let x = Array::from_elem(ncomponents, 1.0 / ncomponents as f64);
    let n = &x * 100.0 * MOL;
    let crit = State::critical_point(eos, Some(&n), None, Default::default()).unwrap();
    let vle = if ncomponents == 1 {
        PhaseEquilibrium::pure(eos, crit.temperature * 0.95, None, Default::default()).unwrap()
    } else {
        PhaseEquilibrium::tp_flash(
            eos,
            crit.temperature,
            crit.pressure(Contributions::Total) * 0.95,
            &crit.moles,
            None,
            Default::default(),
            None,
        )
        .unwrap()
    };

    let mut group = c.benchmark_group(group_name);
    group.bench_function("new_npt_liquid", |b| {
        b.iter(|| {
            npt((
                eos,
                vle.liquid().temperature,
                vle.liquid().pressure(Contributions::Total) * 1.01,
                &n,
                DensityInitialization::Liquid,
            ))
        })
    });
    group.bench_function("new_npt_vapor", |b| {
        b.iter(|| {
            npt((
                eos,
                vle.vapor().temperature,
                vle.vapor().pressure(Contributions::Total) * 0.99,
                &n,
                DensityInitialization::Vapor,
            ))
        })
    });
    group.bench_function("critical_point", |b| {
        b.iter(|| critical_point((eos, Some(&n))))
    });
    if ncomponents == 2 {
        group.bench_function("critical_point_binary_t", |b| {
            b.iter(|| critical_point_binary((eos, crit.temperature)))
        });
        group.bench_function("critical_point_binary_p", |b| {
            b.iter(|| critical_point_binary((eos, crit.pressure(Contributions::Total))))
        });
    }
    if ncomponents != 1 {
        group.bench_function("tp_flash", |b| {
            b.iter(|| {
                tp_flash((
                    eos,
                    crit.temperature,
                    crit.pressure(Contributions::Total) * 0.99,
                    &n,
                ))
            })
        });

        group.bench_function("bubble_point", |b| {
            b.iter(|| bubble_point((eos, vle.liquid().temperature, &vle.liquid().molefracs)))
        });

        group.bench_function("dew_point", |b| {
            b.iter(|| dew_point((eos, vle.vapor().temperature, &vle.vapor().molefracs)))
        });
    } else {
        group.bench_function("pure_t", |b| {
            b.iter(|| pure((eos, vle.vapor().temperature)))
        });
        group.bench_function("pure_p", |b| {
            b.iter(|| pure((eos, vle.vapor().pressure(Contributions::Total))))
        });
    }
}

fn pcsaft(c: &mut Criterion) {
    let parameters = PcSaftParameters::from_json(
        vec!["methane"],
        "./parameters/pcsaft/gross2001.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    let eos = Arc::new(PcSaft::new(Arc::new(parameters)));
    bench_states(c, "state_creation_pcsaft_methane", &eos);

    let parameters = PcSaftParameters::from_json(
        vec!["methane", "ethane"],
        "./parameters/pcsaft/gross2001.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    let eos = Arc::new(PcSaft::new(Arc::new(parameters)));
    bench_states(c, "state_creation_pcsaft_methane_ethane", &eos);

    let parameters = PcSaftParameters::from_json(
        vec!["methane", "ethane", "propane"],
        "./parameters/pcsaft/gross2001.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    let eos = Arc::new(PcSaft::new(Arc::new(parameters)));
    bench_states(c, "state_creation_pcsaft_methane_ethane_propane", &eos);
}

criterion_group!(bench, pcsaft);
criterion_main!(bench);
