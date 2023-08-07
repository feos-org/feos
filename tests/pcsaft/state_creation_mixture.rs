use approx::assert_relative_eq;
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::joback::{Joback, JobackParameters};
use feos_core::parameter::{IdentifierOption, Parameter, ParameterError};
use feos_core::si::*;
use feos_core::{Contributions, EquationOfState, StateBuilder};
use ndarray::prelude::*;
use ndarray::Zip;
use std::error::Error;
use std::sync::Arc;
use typenum::P3;

fn propane_butane_parameters(
) -> Result<(Arc<PcSaftParameters>, Arc<JobackParameters>), ParameterError> {
    let saft = Arc::new(PcSaftParameters::from_json(
        vec!["propane", "butane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?);
    let joback = Arc::new(JobackParameters::from_json(
        vec!["propane", "butane"],
        "tests/pcsaft/test_parameters_joback.json",
        None,
        IdentifierOption::Name,
    )?);
    Ok((saft, joback))
}

#[test]
fn pressure_entropy_molefracs() -> Result<(), Box<dyn Error>> {
    let (saft_params, joback_params) = propane_butane_parameters()?;
    let saft = Arc::new(PcSaft::new(saft_params));
    let joback = Joback::new(joback_params);
    let eos = Arc::new(EquationOfState::new(Arc::new(joback), saft));
    let pressure = BAR;
    let temperature = 300.0 * KELVIN;
    let x = arr1(&[0.3, 0.7]);
    let state = StateBuilder::new(&eos)
        .temperature(temperature)
        .pressure(pressure)
        .molefracs(&x)
        .build()?;
    let molar_entropy = state.molar_entropy(Contributions::Total);
    let state = StateBuilder::new(&eos)
        .pressure(pressure)
        .molar_entropy(molar_entropy)
        .molefracs(&x)
        .build()?;
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
    let saft = Arc::new(PcSaft::new(propane_butane_parameters()?.0));
    let temperature = 300.0 * KELVIN;
    let volume = 1.5e-3 * METER.powi::<P3>();
    let moles = MOL;
    let x = arr1(&[0.3, 0.7]);
    let state = StateBuilder::new(&saft)
        .temperature(temperature)
        .volume(volume)
        .total_moles(moles)
        .molefracs(&x)
        .build()?;
    assert_relative_eq!(state.volume, volume, max_relative = 1e-10);
    Ok(())
}

#[test]
fn temperature_partial_density() -> Result<(), Box<dyn Error>> {
    let saft = Arc::new(PcSaft::new(propane_butane_parameters()?.0));
    let temperature = 300.0 * KELVIN;
    let x = arr1(&[0.3, 0.7]);
    let partial_density = x.clone() * MOL / METER.powi::<P3>();
    let density = partial_density.sum();
    let state = StateBuilder::new(&saft)
        .temperature(temperature)
        .partial_density(&partial_density)
        .build()?;
    assert_relative_eq!(x, state.molefracs, max_relative = 1e-10);
    assert_relative_eq!(density, state.density, max_relative = 1e-10);
    // Zip::from(&state.partial_density.to_reduced(reference))
    //     .and(&partial_density.into_value()?)
    //     .for_each(|&r1, &r2| assert_relative_eq!(r1, r2, max_relative = 1e-10));
    Ok(())
}

#[test]
fn temperature_density_molefracs() -> Result<(), Box<dyn Error>> {
    let saft = Arc::new(PcSaft::new(propane_butane_parameters()?.0));
    let temperature = 300.0 * KELVIN;
    let x = arr1(&[0.3, 0.7]);
    let density = MOL / METER.powi::<P3>();
    let state = StateBuilder::new(&saft)
        .temperature(temperature)
        .density(density)
        .molefracs(&x)
        .build()?;
    Zip::from(&state.molefracs)
        .and(&x)
        .for_each(|&l, &r| assert_relative_eq!(l, r, max_relative = 1e-10));
    assert_relative_eq!(state.density, density);
    Ok(())
}

#[test]
fn temperature_pressure_molefracs() -> Result<(), Box<dyn Error>> {
    let saft = Arc::new(PcSaft::new(propane_butane_parameters()?.0));
    let temperature = 300.0 * KELVIN;
    let pressure = BAR;
    let x = arr1(&[0.3, 0.7]);
    let state = StateBuilder::new(&saft)
        .temperature(temperature)
        .pressure(pressure)
        .molefracs(&x)
        .build()?;
    Zip::from(&state.molefracs)
        .and(&x)
        .for_each(|&l, &r| assert_relative_eq!(l, r, max_relative = 1e-10));
    assert_relative_eq!(
        state.pressure(Contributions::Total),
        pressure,
        max_relative = 1e-10
    );
    Ok(())
}
