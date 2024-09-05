//! Benchmarks for the evaluation of the Helmholtz energy function
//! for a given `StateHD` for different types of dual numbers.
//! These should give an idea about the expected slow-down depending
//! on the dual number type used without the overhead of the `State`
//! creation.
use criterion::{criterion_group, criterion_main, Criterion};
use feos::hard_sphere::HardSphereProperties;
use feos::saftvrmie::{test_utils::test_parameters, SaftVRMie, SaftVRMieParameters};
use feos_core::si::*;
use feos_core::{Derivative, Residual, State, StateHD};
use ndarray::{Array, ScalarOperand};
use num_dual::DualNum;
use std::sync::Arc;

/// Helper function to create a state for given parameters.
/// - temperature is 80% of critical temperature,
/// - volume is critical volume,
/// - molefracs (or moles) for equimolar mixture.
fn state_saftvrmie(parameters: &Arc<SaftVRMieParameters>) -> State<SaftVRMie> {
    let n = parameters.pure_records.len();
    let eos = Arc::new(SaftVRMie::new(parameters.clone()));
    let moles = Array::from_elem(n, 1.0 / n as f64) * 10.0 * MOL;
    let cp = State::critical_point(&eos, Some(&moles), None, Default::default()).unwrap();
    let temperature = 0.8 * cp.temperature;
    State::new_nvt(&eos, temperature, cp.volume, &moles).unwrap()
}

/// Residual Helmholtz energy given an equation of state and a StateHD.
fn a_res<D: DualNum<f64> + Copy + ScalarOperand, E: Residual>(inp: (&Arc<E>, &StateHD<D>)) -> D {
    inp.0.residual_helmholtz_energy(inp.1)
}

fn d_hs<D: DualNum<f64> + Copy>(inp: (&SaftVRMieParameters, D)) -> D {
    inp.0.hs_diameter(inp.1)[0]
}

/// Benchmark for evaluation of the Helmholtz energy for different dual number types.
fn bench_dual_numbers<E: Residual>(
    c: &mut Criterion,
    group_name: &str,
    state: State<E>,
    parameters: &SaftVRMieParameters,
) {
    let mut group = c.benchmark_group(group_name);
    group.bench_function("d_f64", |b| {
        b.iter(|| d_hs((parameters, state.derive0().temperature)))
    });
    group.bench_function("d_dual", |b| {
        b.iter(|| d_hs((parameters, state.derive1(Derivative::DT).temperature)))
    });
    group.bench_function("d_dual2", |b| {
        b.iter(|| d_hs((parameters, state.derive2(Derivative::DT).temperature)))
    });
    group.bench_function("d_hyperdual", |b| {
        b.iter(|| {
            d_hs((
                parameters,
                state
                    .derive2_mixed(Derivative::DT, Derivative::DT)
                    .temperature,
            ))
        })
    });
    group.bench_function("d_dual3", |b| {
        b.iter(|| d_hs((parameters, state.derive3(Derivative::DT).temperature)))
    });

    group.bench_function("a_f64", |b| {
        b.iter(|| a_res((&state.eos, &state.derive0())))
    });
    group.bench_function("a_dual", |b| {
        b.iter(|| a_res((&state.eos, &state.derive1(Derivative::DV))))
    });
    group.bench_function("a_dual2", |b| {
        b.iter(|| a_res((&state.eos, &state.derive2(Derivative::DV))))
    });
    group.bench_function("a_hyperdual", |b| {
        b.iter(|| {
            a_res((
                &state.eos,
                &state.derive2_mixed(Derivative::DV, Derivative::DV),
            ))
        })
    });
    group.bench_function("a_dual3", |b| {
        b.iter(|| a_res((&state.eos, &state.derive3(Derivative::DV))))
    });
}

/// Benchmark for the SAFT VR Mie equation of state
fn saftvrmie(c: &mut Criterion) {
    let parameters = Arc::new(test_parameters().remove("ethane").unwrap());
    bench_dual_numbers(
        c,
        "dual_numbers_saftvrmie_ethane",
        state_saftvrmie(&parameters),
        &parameters,
    );
}

criterion_group!(bench, saftvrmie);
criterion_main!(bench);
