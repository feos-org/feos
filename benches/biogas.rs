use criterion::{criterion_group, criterion_main, Criterion};
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::{
    parameter::{IdentifierOption, Parameter},
    Contributions, EquationOfState, HelmholtzEnergy, HelmholtzEnergyDual, State, StateBuilder,
    StateHD,
};
use ndarray::{arr1, Array1};
use num_dual::{Dual3, Dual64, DualNum, HyperDual64, StaticMat, StaticVec};
use quantity::{si::*, QuantityArray1};
use std::sync::Arc;

/// State generation using `State::new`
fn state(inp: (&Arc<PcSaft>, SINumber, SINumber, &Array1<f64>)) -> State<SIUnit, PcSaft> {
    State::new(
        inp.0,
        Some(inp.1),
        None,
        Some(inp.2),
        None,
        None,
        None,
        Some(inp.3),
        None,
        None,
        None,
        None,
        feos_core::DensityInitialization::None,
        None,
    )
    .unwrap()
}

/// Residual Helmholtz energy given a StateHD
fn a_res<D: DualNum<f64>>(inp: (&Arc<PcSaft>, &StateHD<D>)) -> D
where
    (dyn HelmholtzEnergy + 'static): HelmholtzEnergyDual<D>,
{
    inp.0.evaluate_residual(inp.1)
}

fn da_dv(inp: (&Arc<PcSaft>, SINumber, SINumber, &SIArray1)) -> f64 {
    let t = Dual64::from_re(inp.1.to_reduced(KELVIN).unwrap());
    let v = Dual64::from_re(inp.2.to_reduced(ANGSTROM.powi(3)).unwrap()).derive();
    let m = inp
        .3
        .to_reduced(MOL)
        .unwrap()
        .iter()
        .map(|&mi| Dual64::from_re(mi))
        .collect();
    let s = StateHD::new(t, v, m);
    inp.0.evaluate_residual(&s).eps[0]
}

fn da_dni(inp: (&Arc<PcSaft>, SINumber, SINumber, &SIArray1)) -> Array1<f64> {
    let t = Dual64::from_re(inp.1.to_reduced(KELVIN).unwrap());
    let v = Dual64::from_re(inp.2.to_reduced(ANGSTROM.powi(3)).unwrap());
    let m: Array1<_> = inp
        .3
        .to_reduced(MOL)
        .unwrap()
        .iter()
        .map(|&mi| Dual64::from_re(mi))
        .collect();
    Array1::from_shape_fn(inp.3.len(), |i| {
        let mut n = m.clone();
        n[i] = n[i].derive();
        let s = StateHD::new(t, v, n);
        inp.0.evaluate_residual(&s).eps[0]
    })
}

fn d2a_dt2(inp: (&Arc<PcSaft>, SINumber, SINumber, &SIArray1)) -> f64 {
    let t = HyperDual64::from_re(inp.1.to_reduced(KELVIN).unwrap()).derive2();
    let v = HyperDual64::from_re(inp.2.to_reduced(ANGSTROM.powi(3)).unwrap());
    let m: Array1<_> = inp
        .3
        .to_reduced(MOL)
        .unwrap()
        .iter()
        .map(|&mi| HyperDual64::from_re(mi))
        .collect();

    let s = StateHD::new(t, v, m);
    inp.0.evaluate_residual(&s).eps1eps2[0]
}

fn benchmark_helmholtz_energy(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("helmholtz_energy");

    // real valued evaluation
    let s = StateHD::new(
        t.to_reduced(KELVIN).unwrap(),
        volume.to_reduced(ANGSTROM.powi(3)).unwrap(),
        m.to_reduced(MOL).unwrap(),
    );
    group.bench_function("a_res_f64", |b| b.iter(|| a_res((&eos, &s))));

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
    group.bench_function("a_res_dual", |b| b.iter(|| a_res((&eos, &s))));

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
    group.bench_function("a_res_hyperdual", |b| b.iter(|| a_res((&eos, &s))));

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
    group.bench_function("a_res_dual3", |b| b.iter(|| a_res((&eos, &s))));
}

criterion_group!(bench, benchmark_helmholtz_energy,);
criterion_main!(bench);
