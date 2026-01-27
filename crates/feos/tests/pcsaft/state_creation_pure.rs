use approx::assert_relative_eq;
use feos::ideal_gas::{Joback, JobackParameters};
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::DensityInitialization::{InitialDensity, Liquid, Vapor};
use feos_core::parameter::IdentifierOption;
use feos_core::{Contributions, EquationOfState, FeosResult, PhaseEquilibrium, State, Total};
use quantity::*;

fn propane_parameters() -> FeosResult<(PcSaftParameters, Vec<Joback>)> {
    let saft = PcSaftParameters::from_json(
        vec!["propane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?;
    let joback = Joback::new(JobackParameters::from_json(
        vec!["propane"],
        "tests/pcsaft/test_parameters_joback.json",
        None,
        IdentifierOption::Name,
    )?);
    Ok((saft, joback))
}

#[test]
fn temperature_volume() -> FeosResult<()> {
    let saft = PcSaft::new(propane_parameters()?.0);
    let temperature = 300.0 * KELVIN;
    let volume = 1.5e-3 * METER.powi::<3>();
    let moles = MOL;
    let state = State::new_nvt(&&saft, temperature, volume, moles)?;
    assert_relative_eq!(state.volume()?, volume, max_relative = 1e-10);
    Ok(())
}

#[test]
fn temperature_density() -> FeosResult<()> {
    let saft = PcSaft::new(propane_parameters()?.0);
    let temperature = 300.0 * KELVIN;
    let density = MOL / METER.powi::<3>();
    let state = State::new_pure(&&saft, temperature, density)?;
    assert_relative_eq!(state.density, density, max_relative = 1e-10);
    Ok(())
}

#[test]
fn temperature_total_moles_volume() -> FeosResult<()> {
    let saft = PcSaft::new(propane_parameters()?.0);
    let temperature = 300.0 * KELVIN;
    let total_moles = MOL;
    let volume = METER.powi::<3>();
    let state = State::new_nvt(&&saft, temperature, volume, total_moles)?;
    assert_relative_eq!(state.volume()?, volume, max_relative = 1e-10);
    assert_relative_eq!(state.total_moles()?, total_moles, max_relative = 1e-10);
    Ok(())
}

#[test]
fn temperature_total_moles_density() -> FeosResult<()> {
    let saft = PcSaft::new(propane_parameters()?.0);
    let temperature = 300.0 * KELVIN;
    let total_moles = MOL;
    let density = MOL / METER.powi::<3>();
    let state = State::new_pure(&&saft, temperature, density)?.set_total_moles(total_moles);
    assert_relative_eq!(state.density, density, max_relative = 1e-10);
    assert_relative_eq!(state.total_moles()?, total_moles, max_relative = 1e-10);
    assert_relative_eq!(state.volume()?, total_moles / density, max_relative = 1e-10);
    Ok(())
}

// Pressure constructors

#[test]
fn pressure_temperature() -> FeosResult<()> {
    let saft = PcSaft::new(propane_parameters()?.0);
    let pressure = BAR;
    let temperature = 300.0 * KELVIN;
    let state = State::new_npt(&&saft, temperature, pressure, (), None)?;
    assert_relative_eq!(
        state.pressure(Contributions::Total),
        pressure,
        max_relative = 1e-10
    );
    Ok(())
}

#[test]
fn pressure_temperature_phase() -> FeosResult<()> {
    let saft = PcSaft::new(propane_parameters()?.0);
    let pressure = BAR;
    let temperature = 300.0 * KELVIN;
    let state = State::new_npt(&&saft, temperature, pressure, (), Some(Liquid))?;
    assert_relative_eq!(
        state.pressure(Contributions::Total),
        pressure,
        max_relative = 1e-10
    );
    Ok(())
}

#[test]
fn pressure_temperature_initial_density() -> FeosResult<()> {
    let saft = PcSaft::new(propane_parameters()?.0);
    let pressure = BAR;
    let temperature = 300.0 * KELVIN;
    let init = Some(InitialDensity(MOL / METER.powi::<3>()));
    let state = State::new_npt(&&saft, temperature, pressure, (), init)?;
    assert_relative_eq!(
        state.pressure(Contributions::Total),
        pressure,
        max_relative = 1e-10
    );
    Ok(())
}

#[test]
fn pressure_enthalpy_vapor() -> FeosResult<()> {
    let (saft_params, joback) = propane_parameters()?;
    let saft = PcSaft::new(saft_params);
    let eos = EquationOfState::new(joback, saft);
    let pressure = 0.3 * BAR;
    let molar_enthalpy = 2000.0 * JOULE / MOL;
    let state = State::new_nph(&&eos, pressure, molar_enthalpy, (), Some(Vapor), None)?;
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

    let state = State::new(&&eos, state.temperature, state.density, state.molefracs)?;
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
fn density_internal_energy() -> FeosResult<()> {
    let (saft_params, joback) = propane_parameters()?;
    let saft = PcSaft::new(saft_params);
    let eos = EquationOfState::new(joback, saft);
    let pressure = 5.0 * BAR;
    let temperature = 315.0 * KELVIN;
    let total_moles = 2.5 * MOL;
    let state = State::new_npt(&&eos, temperature, pressure, total_moles, None)?;
    let molar_internal_energy = state.molar_internal_energy(Contributions::Total);
    let state_nvu = State::new_nvu(
        &&eos,
        state.volume()?,
        molar_internal_energy,
        total_moles,
        None,
    )?;
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
fn pressure_enthalpy_total_moles_vapor() -> FeosResult<()> {
    let (saft_params, joback) = propane_parameters()?;
    let saft = PcSaft::new(saft_params);
    let eos = EquationOfState::new(joback, saft);
    let pressure = 0.3 * BAR;
    let molar_enthalpy = 2000.0 * JOULE / MOL;
    let total_moles = 2.5 * MOL;
    let state = State::new_nph(
        &&eos,
        pressure,
        molar_enthalpy,
        total_moles,
        Some(Vapor),
        None,
    )?;
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

    let state = State::new_nvt(
        &&eos,
        state.temperature,
        state.volume()?,
        state.total_moles()?,
    )?;
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
fn pressure_entropy_vapor() -> FeosResult<()> {
    let (saft_params, joback) = propane_parameters()?;
    let saft = PcSaft::new(saft_params);
    let eos = EquationOfState::new(joback, saft);
    let pressure = 0.3 * BAR;
    let molar_entropy = -2.0 * JOULE / MOL / KELVIN;
    let state = State::new_nps(&&eos, pressure, molar_entropy, (), Some(Vapor), None)?;
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

    let state = State::new(&&eos, state.temperature, state.density, state.molefracs)?;
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
fn temperature_entropy_vapor() -> FeosResult<()> {
    let (saft_params, joback) = propane_parameters()?;
    let saft = PcSaft::new(saft_params);
    let eos = EquationOfState::new(joback, saft);
    let pressure = 3.0 * BAR;
    let temperature = 315.15 * KELVIN;
    let total_moles = 3.0 * MOL;
    let state = State::new_npt(&&eos, temperature, pressure, total_moles, None)?;

    let s = State::new_nts(
        &&eos,
        temperature,
        state.molar_entropy(Contributions::Total),
        state.moles()?,
        None,
    )?;
    assert_relative_eq!(
        state.molar_entropy(Contributions::Total),
        s.molar_entropy(Contributions::Total),
        max_relative = 1e-10
    );
    assert_relative_eq!(state.density, s.density, max_relative = 1e-10);
    Ok(())
}

fn assert_multiple_states<E: Total>(
    states: &[(&State<E>, &str)],
    pressure: Pressure,
    enthalpy: MolarEnergy,
    entropy: MolarEntropy,
    density: Density,
    max_relative: f64,
) {
    for (s, name) in states {
        println!("State: {name}");
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
fn test_consistency() -> FeosResult<()> {
    let (saft_params, joback) = propane_parameters()?;
    let saft = PcSaft::new(saft_params);
    let eos = EquationOfState::new(joback, saft);
    let temperatures = [350.0 * KELVIN, 400.0 * KELVIN, 450.0 * KELVIN];
    let pressures = [1.0 * BAR, 2.0 * BAR, 3.0 * BAR];

    for (&temperature, &pressure) in temperatures.iter().zip(pressures.iter()) {
        let state = State::new_npt(&&eos, temperature, pressure, (), None)?;
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

        let state_tv = State::new_pure(&&eos, temperature, density)?;

        let vle = PhaseEquilibrium::pure(&&eos, temperature, None, Default::default());
        let eos = &eos;
        let phase = if let Ok(ps) = vle {
            let p_sat = ps.liquid().pressure(Contributions::Total);
            if pressure > p_sat { Liquid } else { Vapor }
        } else {
            Vapor
        };

        let state_ts = State::new_nts(&eos, temperature, molar_entropy, (), Some(phase))?;

        let state_ps = State::new_nps(&eos, pressure, molar_entropy, (), Some(phase), None)?;

        dbg!("ph");
        let state_ph = State::new_nph(&eos, pressure, molar_enthalpy, (), Some(phase), None)?;

        dbg!("th");
        let state_th = State::new_nth(&eos, temperature, molar_enthalpy, (), Some(phase))?;

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
