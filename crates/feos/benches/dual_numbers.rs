//! Benchmarks for the evaluation of the Helmholtz energy function
//! for a given `StateHD` for different types of dual numbers.
//! These should give an idea about the expected slow-down depending
//! on the dual number type used without the overhead of the `State`
//! creation.
use criterion::{Criterion, criterion_group, criterion_main};
use feos::core::parameter::IdentifierOption;
use feos::core::{ReferenceSystem, Residual, State, StateHD};
use feos::pcsaft::{
    PcSaft, PcSaftAssociationRecord, PcSaftBinaryRecord, PcSaftParameters, PcSaftRecord,
};
use feos_core::parameter::PureRecord;
use nalgebra::{DVector, Dyn, dvector};
use num_dual::{Dual2_64, Dual3_64, Dual64, DualNum, HyperDual64};
use quantity::*;

/// Helper function to create a state for given parameters.
/// - temperature is 80% of critical temperature,
/// - volume is critical volume,
/// - molefracs (or moles) for equimolar mixture.
fn state_pcsaft(n: usize, eos: &PcSaft) -> State<&PcSaft> {
    let moles = DVector::from_element(n, 1.0 / n as f64) * 10.0 * MOL;
    let molefracs = (&moles / moles.sum()).into_value();
    let cp = State::critical_point(&eos, Some(&molefracs), None, None, Default::default()).unwrap();
    let temperature = 0.8 * cp.temperature;
    State::new_nvt(&eos, temperature, cp.volume, &moles).unwrap()
}

/// Residual Helmholtz energy given an equation of state and a StateHD.
fn a_res<D: DualNum<f64> + Copy, E: Residual<Dyn, D>>((eos, state): (&E, &StateHD<D>)) -> D {
    eos.reduced_residual_helmholtz_energy_density(state)
}

/// Benchmark for evaluation of the Helmholtz energy for different dual number types.
fn bench_dual_numbers<E: Residual>(c: &mut Criterion, group_name: &str, state: State<E>) {
    let mut group = c.benchmark_group(group_name);
    group.bench_function("a_f64", |b| {
        b.iter(|| a_res((&state.eos, &derive0(&state))))
    });
    group.bench_function("a_dual", |b| {
        b.iter(|| a_res((&state.eos.lift(), &derive1(&state, Derivative::DV))))
    });
    group.bench_function("a_dual2", |b| {
        b.iter(|| a_res((&state.eos.lift(), &derive2(&state, Derivative::DV))))
    });
    group.bench_function("a_hyperdual", |b| {
        b.iter(|| {
            a_res((
                &state.eos.lift(),
                &derive2_mixed(&state, Derivative::DV, Derivative::DV),
            ))
        })
    });
    group.bench_function("a_dual3", |b| {
        b.iter(|| a_res((&state.eos.lift(), &derive3(&state, Derivative::DV))))
    });
}

/// Benchmark for the PC-SAFT equation of state
fn pcsaft(c: &mut Criterion) {
    // methane
    let parameters = PcSaftParameters::from_json(
        vec!["methane"],
        "../../parameters/pcsaft/gross2001.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    let eos = &PcSaft::new(parameters);
    bench_dual_numbers(c, "dual_numbers_pcsaft_methane", state_pcsaft(1, eos));

    // water (4C, polar)
    let parameters = PcSaftParameters::from_json(
        vec!["water_4C_polar"],
        "../../parameters/pcsaft/rehner2020.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    let eos = &PcSaft::new(parameters);
    bench_dual_numbers(
        c,
        "dual_numbers_pcsaft_water_4c_polar",
        state_pcsaft(1, eos),
    );

    // methane, ethane, propane
    let parameters = PcSaftParameters::from_json(
        vec!["methane", "ethane", "propane"],
        "../../parameters/pcsaft/gross2001.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    let eos = &PcSaft::new(parameters);
    bench_dual_numbers(
        c,
        "dual_numbers_pcsaft_methane_ethane_propane",
        state_pcsaft(3, eos),
    );
}

/// Benchmark for the PC-SAFT equation of state.
/// Binary system of methane and co2 used to model biogas.
fn methane_co2_pcsaft(c: &mut Criterion) {
    type Pure = PureRecord<PcSaftRecord, PcSaftAssociationRecord>;
    let methane = Pure::from_json(
        &["methane"],
        "../../parameters/pcsaft/gross2001.json",
        IdentifierOption::Name,
    )
    .unwrap()
    .pop()
    .unwrap();
    let co2 = Pure::from_json(
        &["carbon dioxide"],
        "../../parameters/pcsaft/gross2005_fit.json",
        IdentifierOption::Name,
    )
    .unwrap()
    .pop()
    .unwrap();

    let k_ij = -0.0192211646;
    let br = PcSaftBinaryRecord::new(k_ij);
    let parameters = PcSaftParameters::new_binary([methane, co2], Some(br), vec![]).unwrap();
    let eos = &PcSaft::new(parameters);

    // 230 K, 50 bar, x0 = 0.15
    let temperature = 230.0 * KELVIN;
    let density = 24.16896 * KILO * MOL / METER.powi::<3>();
    let volume = 10.0 * MOL / density;
    let x = dvector![0.15, 0.85];
    let moles = &x * 10.0 * MOL;
    let state = State::new_nvt(&eos, temperature, volume, &moles).unwrap();
    bench_dual_numbers(c, "dual_numbers_pcsaft_methane_co2", state);
}

criterion_group!(bench, pcsaft, methane_co2_pcsaft);
criterion_main!(bench);

enum Derivative {
    /// Derivative with respect to system volume.
    DV,
    /// Derivative with respect to temperature.
    #[expect(dead_code)]
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
