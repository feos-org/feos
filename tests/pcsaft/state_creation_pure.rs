use approx::assert_relative_eq;
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::joback::{Joback, JobackParameters};
use feos_core::parameter::{IdentifierOption, Parameter, ParameterError};
use feos_core::si::*;
use feos_core::{
    Contributions, DensityInitialization, EquationOfState, IdealGas, PhaseEquilibrium, Residual,
    State, StateBuilder,
};
use std::error::Error;
use std::sync::Arc;
use typenum::P3;

fn propane_parameters() -> Result<(Arc<PcSaftParameters>, Arc<JobackParameters>), ParameterError> {
    let saft = Arc::new(PcSaftParameters::from_json(
        vec!["propane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?);
    let joback = Arc::new(JobackParameters::from_json(
        vec!["propane"],
        "tests/pcsaft/test_parameters_joback.json",
        None,
        IdentifierOption::Name,
    )?);
    Ok((saft, joback))
}

#[test]
fn temperature_volume() -> Result<(), Box<dyn Error>> {
    let saft = Arc::new(PcSaft::new(propane_parameters()?.0));
    let temperature = 300.0 * KELVIN;
    let volume = 1.5e-3 * METER.powi::<P3>();
    let moles = MOL;
    let state = StateBuilder::new(&saft)
        .temperature(temperature)
        .volume(volume)
        .total_moles(moles)
        .build()?;
    assert_relative_eq!(state.volume, volume, max_relative = 1e-10);
    Ok(())
}

#[test]
fn temperature_density() -> Result<(), Box<dyn Error>> {
    let saft = Arc::new(PcSaft::new(propane_parameters()?.0));
    let temperature = 300.0 * KELVIN;
    let density = MOL / METER.powi::<P3>();
    let state = StateBuilder::new(&saft)
        .temperature(temperature)
        .density(density)
        .build()?;
    assert_relative_eq!(state.density, density, max_relative = 1e-10);
    Ok(())
}

#[test]
fn temperature_total_moles_volume() -> Result<(), Box<dyn Error>> {
    let saft = Arc::new(PcSaft::new(propane_parameters()?.0));
    let temperature = 300.0 * KELVIN;
    let total_moles = MOL;
    let volume = METER.powi::<P3>();
    let state = StateBuilder::new(&saft)
        .temperature(temperature)
        .volume(volume)
        .total_moles(total_moles)
        .build()?;
    assert_relative_eq!(state.volume, volume, max_relative = 1e-10);
    assert_relative_eq!(state.total_moles, total_moles, max_relative = 1e-10);
    Ok(())
}

#[test]
fn temperature_total_moles_density() -> Result<(), Box<dyn Error>> {
    let saft = Arc::new(PcSaft::new(propane_parameters()?.0));
    let temperature = 300.0 * KELVIN;
    let total_moles = MOL;
    let density = MOL / METER.powi::<P3>();
    let state = StateBuilder::new(&saft)
        .temperature(temperature)
        .density(density)
        .total_moles(total_moles)
        .build()?;
    assert_relative_eq!(state.density, density, max_relative = 1e-10);
    assert_relative_eq!(state.total_moles, total_moles, max_relative = 1e-10);
    assert_relative_eq!(state.volume, total_moles / density, max_relative = 1e-10);
    Ok(())
}

// Pressure constructors

#[test]
fn pressure_temperature() -> Result<(), Box<dyn Error>> {
    let saft = Arc::new(PcSaft::new(propane_parameters()?.0));
    let pressure = BAR;
    let temperature = 300.0 * KELVIN;
    let state = StateBuilder::new(&saft)
        .temperature(temperature)
        .pressure(pressure)
        .build()?;
    assert_relative_eq!(
        state.pressure(Contributions::Total),
        pressure,
        max_relative = 1e-10
    );
    Ok(())
}

#[test]
fn pressure_temperature_phase() -> Result<(), Box<dyn Error>> {
    let saft = Arc::new(PcSaft::new(propane_parameters()?.0));
    let pressure = BAR;
    let temperature = 300.0 * KELVIN;
    let state = StateBuilder::new(&saft)
        .temperature(temperature)
        .pressure(pressure)
        .liquid()
        .build()?;
    assert_relative_eq!(
        state.pressure(Contributions::Total),
        pressure,
        max_relative = 1e-10
    );
    Ok(())
}

#[test]
fn pressure_temperature_initial_density() -> Result<(), Box<dyn Error>> {
    let saft = Arc::new(PcSaft::new(propane_parameters()?.0));
    let pressure = BAR;
    let temperature = 300.0 * KELVIN;
    let state = StateBuilder::new(&saft)
        .temperature(temperature)
        .pressure(pressure)
        .initial_density(MOL / METER.powi::<P3>())
        .build()?;
    assert_relative_eq!(
        state.pressure(Contributions::Total),
        pressure,
        max_relative = 1e-10
    );
    Ok(())
}

#[test]
fn pressure_enthalpy_vapor() -> Result<(), Box<dyn Error>> {
    let (saft_params, joback_params) = propane_parameters()?;
    let saft = Arc::new(PcSaft::new(saft_params));
    let joback = Joback::new(joback_params);
    let eos = Arc::new(EquationOfState::new(Arc::new(joback), saft));
    let pressure = 0.3 * BAR;
    let molar_enthalpy = 2000.0 * JOULE / MOL;
    let state = StateBuilder::new(&eos)
        .pressure(pressure)
        .molar_enthalpy(molar_enthalpy)
        .vapor()
        .build()?;
    assert_relative_eq!(
        state.molar_enthalpy(Contributions::Total),
        molar_enthalpy,
        max_relative = 1e-10
    );
    assert_relative_eq!(
        state.pressure(Contributions::Total),
        pressure,
        max_relative = 1e-10
    );

    let state = StateBuilder::new(&eos)
        .volume(state.volume)
        .temperature(state.temperature)
        .moles(&state.moles)
        .build()?;
    assert_relative_eq!(
        state.molar_enthalpy(Contributions::Total),
        molar_enthalpy,
        max_relative = 1e-10
    );
    assert_relative_eq!(
        state.pressure(Contributions::Total),
        pressure,
        max_relative = 1e-10
    );
    Ok(())
}

#[test]
fn density_internal_energy() -> Result<(), Box<dyn Error>> {
    let (saft_params, joback_params) = propane_parameters()?;
    let saft = Arc::new(PcSaft::new(saft_params));
    let joback = Joback::new(joback_params);
    let eos = Arc::new(EquationOfState::new(Arc::new(joback), saft));
    let pressure = 5.0 * BAR;
    let temperature = 315.0 * KELVIN;
    let total_moles = 2.5 * MOL;
    let state = StateBuilder::new(&eos)
        .pressure(pressure)
        .temperature(temperature)
        .total_moles(total_moles)
        .build()?;
    let molar_internal_energy = state.molar_internal_energy(Contributions::Total);
    let state_nvu = StateBuilder::new(&eos)
        .volume(state.volume)
        .molar_internal_energy(molar_internal_energy)
        .total_moles(total_moles)
        .build()?;
    assert_relative_eq!(
        molar_internal_energy,
        state_nvu.molar_internal_energy(Contributions::Total),
        max_relative = 1e-10
    );
    assert_relative_eq!(temperature, state_nvu.temperature, max_relative = 1e-10);
    assert_relative_eq!(state.density, state_nvu.density, max_relative = 1e-10);
    Ok(())
}

#[test]
fn pressure_enthalpy_total_moles_vapor() -> Result<(), Box<dyn Error>> {
    let (saft_params, joback_params) = propane_parameters()?;
    let saft = Arc::new(PcSaft::new(saft_params));
    let joback = Joback::new(joback_params);
    let eos = Arc::new(EquationOfState::new(Arc::new(joback), saft));
    let pressure = 0.3 * BAR;
    let molar_enthalpy = 2000.0 * JOULE / MOL;
    let total_moles = 2.5 * MOL;
    let state = StateBuilder::new(&eos)
        .pressure(pressure)
        .molar_enthalpy(molar_enthalpy)
        .total_moles(total_moles)
        .vapor()
        .build()?;
    assert_relative_eq!(
        state.molar_enthalpy(Contributions::Total),
        molar_enthalpy,
        max_relative = 1e-10
    );
    assert_relative_eq!(
        state.pressure(Contributions::Total),
        pressure,
        max_relative = 1e-10
    );

    let state = StateBuilder::new(&eos)
        .volume(state.volume)
        .temperature(state.temperature)
        .total_moles(state.total_moles)
        .build()?;
    assert_relative_eq!(
        state.molar_enthalpy(Contributions::Total),
        molar_enthalpy,
        max_relative = 1e-10
    );
    assert_relative_eq!(
        state.pressure(Contributions::Total),
        pressure,
        max_relative = 1e-10
    );
    Ok(())
}

#[test]
fn pressure_entropy_vapor() -> Result<(), Box<dyn Error>> {
    let (saft_params, joback_params) = propane_parameters()?;
    let saft = Arc::new(PcSaft::new(saft_params));
    let joback = Joback::new(joback_params);
    let eos = Arc::new(EquationOfState::new(Arc::new(joback), saft));
    let pressure = 0.3 * BAR;
    let molar_entropy = -2.0 * JOULE / MOL / KELVIN;
    let state = StateBuilder::new(&eos)
        .pressure(pressure)
        .molar_entropy(molar_entropy)
        .vapor()
        .build()?;
    assert_relative_eq!(
        state.molar_entropy(Contributions::Total),
        molar_entropy,
        max_relative = 1e-10
    );
    assert_relative_eq!(
        state.pressure(Contributions::Total),
        pressure,
        max_relative = 1e-10
    );

    let state = StateBuilder::new(&eos)
        .volume(state.volume)
        .temperature(state.temperature)
        .moles(&state.moles)
        .build()?;
    assert_relative_eq!(
        state.molar_entropy(Contributions::Total),
        molar_entropy,
        max_relative = 1e-10
    );
    assert_relative_eq!(
        state.pressure(Contributions::Total),
        pressure,
        max_relative = 1e-10
    );
    Ok(())
}

#[test]
fn temperature_entropy_vapor() -> Result<(), Box<dyn Error>> {
    let (saft_params, joback_params) = propane_parameters()?;
    let saft = Arc::new(PcSaft::new(saft_params));
    let joback = Joback::new(joback_params);
    let eos = Arc::new(EquationOfState::new(Arc::new(joback), saft));
    let pressure = 3.0 * BAR;
    let temperature = 315.15 * KELVIN;
    let total_moles = 3.0 * MOL;
    let state = StateBuilder::new(&eos)
        .temperature(temperature)
        .pressure(pressure)
        .total_moles(total_moles)
        .build()?;

    let s = State::new_nts(
        &eos,
        temperature,
        state.molar_entropy(Contributions::Total),
        &state.moles,
        DensityInitialization::None,
    )?;
    assert_relative_eq!(
        state.molar_entropy(Contributions::Total),
        s.molar_entropy(Contributions::Total),
        max_relative = 1e-10
    );
    assert_relative_eq!(state.density, s.density, max_relative = 1e-10);
    Ok(())
}

fn assert_multiple_states<E: Residual + IdealGas>(
    states: &[(&State<E>, &str)],
    pressure: Pressure,
    enthalpy: MolarEnergy,
    entropy: MolarEntropy,
    density: Density,
    max_relative: f64,
) {
    for (s, name) in states {
        println!("State: {}", name);
        assert_relative_eq!(s.density, density, max_relative = max_relative);
        assert_relative_eq!(
            s.pressure(Contributions::Total),
            pressure,
            max_relative = max_relative
        );
        assert_relative_eq!(
            s.molar_enthalpy(Contributions::Total),
            enthalpy,
            max_relative = max_relative
        );
        assert_relative_eq!(
            s.molar_entropy(Contributions::Total),
            entropy,
            max_relative = max_relative
        );
    }
}

#[test]
fn test_consistency() -> Result<(), Box<dyn Error>> {
    let (saft_params, joback_params) = propane_parameters()?;
    let saft = Arc::new(PcSaft::new(saft_params));
    let joback = Joback::new(joback_params);
    let eos = Arc::new(EquationOfState::new(Arc::new(joback), saft));
    let temperatures = [350.0 * KELVIN, 400.0 * KELVIN, 450.0 * KELVIN];
    let pressures = [1.0 * BAR, 2.0 * BAR, 3.0 * BAR];

    for (&temperature, &pressure) in temperatures.iter().zip(pressures.iter()) {
        let state = StateBuilder::new(&eos)
            .pressure(pressure)
            .temperature(temperature)
            .build()?;
        assert_relative_eq!(
            state.pressure(Contributions::Total),
            pressure,
            max_relative = 1e-10
        );
        println!(
            "temperature: {}\npressure: {}\ndensity: {}",
            temperature, pressure, state.density
        );
        let molar_enthalpy = state.molar_enthalpy(Contributions::Total);
        let molar_entropy = state.molar_entropy(Contributions::Total);
        let density = state.density;

        let state_tv = StateBuilder::new(&eos)
            .temperature(temperature)
            .density(density)
            .build()?;

        let vle = PhaseEquilibrium::pure(&eos, temperature, None, Default::default());
        let builder = if let Ok(ps) = vle {
            let p_sat = ps.liquid().pressure(Contributions::Total);
            if pressure > p_sat {
                StateBuilder::new(&eos).liquid()
            } else {
                StateBuilder::new(&eos).vapor()
            }
        } else {
            StateBuilder::new(&eos).vapor()
        };

        let state_ts = builder
            .clone()
            .temperature(temperature)
            .molar_entropy(molar_entropy)
            .build()?;

        let state_ps = builder
            .clone()
            .pressure(pressure)
            .molar_entropy(molar_entropy)
            .build()?;

        dbg!("ph");
        let state_ph = builder
            .clone()
            .pressure(pressure)
            .molar_enthalpy(molar_enthalpy)
            .build()?;

        dbg!("th");
        let state_th = builder
            .clone()
            .temperature(temperature)
            .molar_enthalpy(molar_enthalpy)
            .build()?;

        dbg!("assertions");
        assert_multiple_states(
            &[
                (&state_ph, "p, h"),
                (&state_tv, "T, V"),
                (&state_ts, "T, s"),
                (&state_th, "T, h"),
                (&state_ps, "p, s"),
            ],
            pressure,
            molar_enthalpy,
            molar_entropy,
            density,
            1e-7,
        );
    }
    Ok(())
}
