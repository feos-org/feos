use approx::assert_relative_eq;
use feos::ideal_gas::Joback;
use feos::pcsaft::PcSaftBinary;
use feos_core::{
    Contributions, EquationOfState, FeosResult, IdealGasAD, ParametersAD, PhaseEquilibrium,
    ReferenceSystem, SolverOptions, Verbosity,
};
use nalgebra::U1;
use num_dual::{DualStruct, DualVec};
use quantity::*;

#[test]
fn test_ph_flash() -> FeosResult<()> {
    let params = [
        [1.5, 3.4, 180.0, 2.2, 0.03, 2500., 2.0, 1.0],
        [4.5, 3.6, 250.0, 1.2, 0.015, 1500., 1.0, 2.0],
    ];
    let kij = 0.15;
    let pcsaft = PcSaftBinary::new(params, kij);
    let joback = [
        Joback([380., 0.0, 0.0, 0.0, 0.0]),
        Joback([210., 0.0, 0.0, 0.0, 0.0]),
    ];
    let eos = EquationOfState::new(joback.clone(), pcsaft);
    let p = 50.0 * BAR;
    let t0 = Some(500.0 * KELVIN);
    let x = 0.3;
    let dew = PhaseEquilibrium::dew_point(&eos, p, x, t0, None, Default::default())?;
    let bubble = PhaseEquilibrium::bubble_point(&eos, p, x, t0, None, Default::default())?;
    let h = 0.2 * dew.molar_enthalpy() + 0.8 * bubble.molar_enthalpy();
    let t0 = 0.8 * dew.vapor().temperature + 0.2 * bubble.vapor().temperature;
    let options = SolverOptions {
        verbosity: Verbosity::Iter,
        ..Default::default()
    };
    let vle = PhaseEquilibrium::ph_flash(&eos, p, h, x, t0, options)?;
    println!("{vle}");
    println!("{h}\n{}", vle.molar_enthalpy());
    assert_relative_eq!(h, vle.molar_enthalpy(), max_relative = 1e-10);

    let pcsaft_ad = pcsaft.named_derivatives(["k_ij"]);
    let joback_ad = joback.each_ref().map(|j| j.lift());
    let eos_ad = EquationOfState::new(joback_ad, pcsaft_ad);
    let vle_ad = PhaseEquilibrium::ph_flash(
        &eos_ad,
        Pressure::from_inner(&p),
        MolarEnergy::from_inner(&h),
        DualVec::from_inner(&x),
        t0,
        Default::default(),
    )?;
    let [[dt]] = vle_ad
        .vapor()
        .temperature
        .into_reduced()
        .eps
        .unwrap_generic(U1, U1)
        .data
        .0;
    println!("{dt}");

    let dkij = 1e-7;
    let pcsaft_h = PcSaftBinary::new(params, kij + dkij);
    let eos_h = EquationOfState::new(joback.clone(), pcsaft_h);
    let vle_h = PhaseEquilibrium::ph_flash(&eos_h, p, h, x, t0, Default::default())?;
    let dt_h = (vle_h.vapor().temperature - vle.vapor().temperature).into_reduced() / dkij;
    println!("{dt_h}");
    assert_relative_eq!(dt, dt_h, max_relative = 1e-4);

    Ok(())
}

#[test]
fn test_ps_flash() -> FeosResult<()> {
    let params = [
        [1.5, 3.4, 180.0, 2.2, 0.03, 2500., 2.0, 1.0],
        [4.5, 3.6, 250.0, 1.2, 0.015, 1500., 1.0, 2.0],
    ];
    let kij = 0.15;
    let pcsaft = PcSaftBinary::new(params, kij);
    let joback = [
        Joback([380., 0.0, 0.0, 0.0, 0.0]),
        Joback([210., 0.0, 0.0, 0.0, 0.0]),
    ];
    let eos = EquationOfState::new(joback.clone(), pcsaft);
    let p = 50.0 * BAR;
    println!(
        "{}",
        PhaseEquilibrium::bubble_point(&eos, 500.0 * KELVIN, 0.5, None, None, Default::default())?
            .vapor()
            .pressure(Contributions::Total)
    );
    let t0 = Some(500.0 * KELVIN);
    let x = 0.3;
    let dew = PhaseEquilibrium::dew_point(&eos, p, x, t0, None, Default::default())?;
    let bubble = PhaseEquilibrium::bubble_point(&eos, p, x, t0, None, Default::default())?;
    let s = 0.2 * dew.molar_entropy() + 0.8 * bubble.molar_entropy();
    let t0 = 0.8 * dew.vapor().temperature + 0.2 * bubble.vapor().temperature;
    let options = SolverOptions {
        verbosity: Verbosity::Iter,
        ..Default::default()
    };
    let vle = PhaseEquilibrium::ps_flash(&eos, p, s, x, t0, options)?;
    println!("{vle}");
    println!("{s}\n{}", vle.molar_entropy());
    assert_relative_eq!(s, vle.molar_entropy(), max_relative = 1e-10);

    let pcsaft_ad = pcsaft.named_derivatives(["k_ij"]);
    let joback_ad = joback.each_ref().map(|j| j.lift());
    let eos_ad = EquationOfState::new(joback_ad, pcsaft_ad);
    let vle_ad = PhaseEquilibrium::ps_flash(
        &eos_ad,
        Pressure::from_inner(&p),
        MolarEntropy::from_inner(&s),
        DualVec::from_inner(&x),
        t0,
        Default::default(),
    )?;
    let [[dt]] = vle_ad
        .vapor()
        .temperature
        .into_reduced()
        .eps
        .unwrap_generic(U1, U1)
        .data
        .0;
    println!("{dt}");

    let dkij = 1e-7;
    let pcsaft_h = PcSaftBinary::new(params, kij + dkij);
    let eos_h = EquationOfState::new(joback.clone(), pcsaft_h);
    let vle_h = PhaseEquilibrium::ps_flash(&eos_h, p, s, x, t0, Default::default())?;
    let dt_h = (vle_h.vapor().temperature - vle.vapor().temperature).into_reduced() / dkij;
    println!("{dt_h}");
    assert_relative_eq!(dt, dt_h, max_relative = 1e-4);

    Ok(())
}
