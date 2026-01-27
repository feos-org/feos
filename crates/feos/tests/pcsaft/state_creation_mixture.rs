use approx::assert_relative_eq;
use feos::ideal_gas::{Joback, JobackParameters};
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::parameter::IdentifierOption;
use feos_core::{Contributions, EquationOfState, FeosResult, State};
use nalgebra::dvector;
use quantity::*;
use std::error::Error;

fn propane_butane_parameters() -> FeosResult<(PcSaftParameters, Vec<Joback>)> {
    let saft = PcSaftParameters::from_json(
        vec!["propane", "butane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?;
    let joback = Joback::new(JobackParameters::from_json(
        vec!["propane", "butane"],
        "tests/pcsaft/test_parameters_joback.json",
        None,
        IdentifierOption::Name,
    )?);
    Ok((saft, joback))
}

#[test]
fn pressure_entropy_molefracs() -> Result<(), Box<dyn Error>> {
    let (saft_params, joback) = propane_butane_parameters()?;
    let saft = PcSaft::new(saft_params);
    let eos = EquationOfState::new(joback, saft);
    let pressure = BAR;
    let temperature = 300.0 * KELVIN;
    let x = dvector![0.3, 0.7];
    let state = State::new_npt(&&eos, temperature, pressure, &x, None)?;
    let molar_entropy = state.molar_entropy(Contributions::Total);
    let state = State::new_nps(&&eos, pressure, molar_entropy, x, None, None)?;
    assert_relative_eq!(
        state.molar_entropy(Contributions::Total),
        molar_entropy,
        max_relative = 1e-8
    );
    assert_relative_eq!(state.temperature, temperature, max_relative = 1e-10);
    assert_relative_eq!(
        state.pressure(Contributions::Total),
        pressure,
        max_relative = 1e-8
    );
    Ok(())
}

#[test]
fn volume_temperature_molefracs() -> Result<(), Box<dyn Error>> {
    let saft = PcSaft::new(propane_butane_parameters()?.0);
    let temperature = 300.0 * KELVIN;
    let volume = 1.5e-3 * METER.powi::<3>();
    let moles = MOL;
    let x = dvector![0.3, 0.7];
    let state = State::new_nvt(&&saft, temperature, volume, (x, moles))?;
    assert_relative_eq!(state.volume(), volume, max_relative = 1e-10);
    Ok(())
}

#[test]
fn temperature_partial_density() -> Result<(), Box<dyn Error>> {
    let saft = PcSaft::new(propane_butane_parameters()?.0);
    let temperature = 300.0 * KELVIN;
    let x = dvector![0.3, 0.7];
    let partial_density = x.clone() * MOL / METER.powi::<3>();
    let density = partial_density.sum();
    let state = State::new_density(&&saft, temperature, partial_density)?;
    assert_relative_eq!(x, state.molefracs, max_relative = 1e-10);
    assert_relative_eq!(density, state.density, max_relative = 1e-10);
    Ok(())
}

#[test]
fn temperature_density_molefracs() -> Result<(), Box<dyn Error>> {
    let saft = PcSaft::new(propane_butane_parameters()?.0);
    let temperature = 300.0 * KELVIN;
    let x = dvector![0.3, 0.7];
    let density = MOL / METER.powi::<3>();
    let state = State::new(&&saft, temperature, density, &x)?;
    state
        .molefracs
        .iter()
        .zip(&x)
        .for_each(|(&l, &r)| assert_relative_eq!(l, r, max_relative = 1e-10));
    assert_relative_eq!(state.density, density);
    Ok(())
}

#[test]
fn temperature_pressure_molefracs() -> Result<(), Box<dyn Error>> {
    let saft = PcSaft::new(propane_butane_parameters()?.0);
    let temperature = 300.0 * KELVIN;
    let pressure = BAR;
    let x = dvector![0.3, 0.7];
    let state = State::new_npt(&&saft, temperature, pressure, &x, None)?;
    state
        .molefracs
        .iter()
        .zip(&x)
        .for_each(|(&l, &r)| assert_relative_eq!(l, r, max_relative = 1e-10));
    assert_relative_eq!(
        state.pressure(Contributions::Total),
        pressure,
        max_relative = 1e-10
    );
    Ok(())
}
