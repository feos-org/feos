//! Benchmarks for the evaluation of the Helmholtz energy function
//! for a given `StateHD` for different types of dual numbers.
//! These should give an idea about the expected slow-down depending
//! on the dual number type used without the overhead of the `State`
//! creation.
use criterion::{Criterion, criterion_group, criterion_main};
use feos::core::{Residual, State, StateHD};
use feos::hard_sphere::HardSphereProperties;
use feos::saftvrmie::{SaftVRMie, test_utils::test_parameters};
use feos_core::ReferenceSystem;
use nalgebra::{DVector, Dyn};
use num_dual::{Dual2_64, Dual3_64, Dual64, DualNum, HyperDual64};
use quantity::*;

/// Helper function to create a state for given parameters.
/// - temperature is 80% of critical temperature,
/// - volume is critical volume,
/// - molefracs (or moles) for equimolar mixture.
fn state_saftvrmie(n: usize, eos: &SaftVRMie) -> State<&SaftVRMie> {
    let molefracs = DVector::from_element(n, 1.0 / n as f64);
    let cp = State::critical_point(&eos, Some(&molefracs), None, None, Default::default()).unwrap();
    let temperature = 0.8 * cp.temperature;
    State::new_nvt(&eos, temperature, cp.volume, &(molefracs * 10. * MOL)).unwrap()
}

/// Residual Helmholtz energy given an equation of state and a StateHD.
fn a_res<D: DualNum<f64> + Copy, E: Residual<Dyn, D>>((eos, state): (&E, &StateHD<D>)) -> D {
    eos.reduced_residual_helmholtz_energy_density(state)
}

fn d_hs<D: DualNum<f64> + Copy>(inp: (&SaftVRMie, D)) -> D {
    inp.0.params.hs_diameter(inp.1)[0]
}

/// Benchmark for evaluation of the Helmholtz energy for different dual number types.
fn bench_dual_numbers(c: &mut Criterion, group_name: &str, state: State<&SaftVRMie>) {
    let mut group = c.benchmark_group(group_name);
    group.bench_function("d_f64", |b| {
        b.iter(|| d_hs((state.eos, derive0(&state).temperature)))
    });
    group.bench_function("d_dual", |b| {
        b.iter(|| d_hs((state.eos, derive1(&state, Derivative::DT).temperature)))
    });
    group.bench_function("d_dual2", |b| {
        b.iter(|| d_hs((state.eos, derive2(&state, Derivative::DT).temperature)))
    });
    group.bench_function("d_hyperdual", |b| {
        b.iter(|| {
            d_hs((
                state.eos,
                derive2_mixed(&state, Derivative::DT, Derivative::DT).temperature,
            ))
        })
    });
    group.bench_function("d_dual3", |b| {
        b.iter(|| d_hs((state.eos, derive3(&state, Derivative::DT).temperature)))
    });

    group.bench_function("a_f64", |b| {
        b.iter(|| a_res((&state.eos, &derive0(&state))))
    });
    group.bench_function("a_dual", |b| {
        b.iter(|| a_res((&state.eos, &derive1(&state, Derivative::DV))))
    });
    group.bench_function("a_dual2", |b| {
        b.iter(|| a_res((&state.eos, &derive2(&state, Derivative::DV))))
    });
    group.bench_function("a_hyperdual", |b| {
        b.iter(|| {
            a_res((
                &state.eos,
                &derive2_mixed(&state, Derivative::DV, Derivative::DV),
            ))
        })
    });
    group.bench_function("a_dual3", |b| {
        b.iter(|| a_res((&state.eos, &derive3(&state, Derivative::DV))))
    });
}

/// Benchmark for the SAFT VR Mie equation of state
fn saftvrmie(c: &mut Criterion) {
    let parameters = test_parameters().remove("ethane").unwrap();
    let eos = &SaftVRMie::new(parameters);
    bench_dual_numbers(c, "dual_numbers_saftvrmie_ethane", state_saftvrmie(1, eos));
}

criterion_group!(bench, saftvrmie);
criterion_main!(bench);

enum Derivative {
    /// Derivative with respect to system volume.
    DV,
    /// Derivative with respect to temperature.
    DT,
    /// Derivative with respect to component `i`.
    #[expect(dead_code)]
    DN(usize),
}

/// Creates a [StateHD] cloning temperature, volume and moles.
fn derive0<E>(state: &State<E>) -> StateHD<f64> {
    let total_moles = state.total_moles.into_reduced();
    StateHD::new(
        state.temperature.into_reduced(),
        state.volume.into_reduced() / total_moles,
        &(state.moles.to_reduced() / total_moles),
    )
}

/// Creates a [StateHD] taking the first derivative.
fn derive1<E>(state: &State<E>, derivative: Derivative) -> StateHD<Dual64> {
    let state = derive0(state);
    let mut t = Dual64::from(state.temperature);
    let mut v = Dual64::from(state.partial_density.sum().recip());
    let mut n = state.molefracs.map(Dual64::from);
    match derivative {
        Derivative::DT => t = t.derivative(),
        Derivative::DV => v = v.derivative(),
        Derivative::DN(i) => n[i] = n[i].derivative(),
    }
    StateHD::new(t, v, &n)
}

/// Creates a [StateHD] taking the first and second (partial) derivatives.
fn derive2<E>(state: &State<E>, derivative: Derivative) -> StateHD<Dual2_64> {
    let state = derive0(state);
    let mut t = Dual2_64::from(state.temperature);
    let mut v = Dual2_64::from(state.partial_density.sum().recip());
    let mut n = state.molefracs.map(Dual2_64::from);
    match derivative {
        Derivative::DT => t = t.derivative(),
        Derivative::DV => v = v.derivative(),
        Derivative::DN(i) => n[i] = n[i].derivative(),
    }
    StateHD::new(t, v, &n)
}

/// Creates a [StateHD] taking the first and second (partial) derivatives.
fn derive2_mixed<E>(
    state: &State<E>,
    derivative1: Derivative,
    derivative2: Derivative,
) -> StateHD<HyperDual64> {
    let state = derive0(state);
    let mut t = HyperDual64::from(state.temperature);
    let mut v = HyperDual64::from(state.partial_density.sum().recip());
    let mut n = state.molefracs.map(HyperDual64::from);
    match derivative1 {
        Derivative::DT => t = t.derivative1(),
        Derivative::DV => v = v.derivative1(),
        Derivative::DN(i) => n[i] = n[i].derivative1(),
    }
    match derivative2 {
        Derivative::DT => t = t.derivative2(),
        Derivative::DV => v = v.derivative2(),
        Derivative::DN(i) => n[i] = n[i].derivative2(),
    }
    StateHD::new(t, v, &n)
}

/// Creates a [StateHD] taking the first, second, and third derivative with respect to a single property.
fn derive3<E>(state: &State<E>, derivative: Derivative) -> StateHD<Dual3_64> {
    let state = derive0(state);
    let mut t = Dual3_64::from(state.temperature);
    let mut v = Dual3_64::from(state.partial_density.sum().recip());
    let mut n = state.molefracs.map(Dual3_64::from);
    match derivative {
        Derivative::DT => t = t.derivative(),
        Derivative::DV => v = v.derivative(),
        Derivative::DN(i) => n[i] = n[i].derivative(),
    };
    StateHD::new(t, v, &n)
}
