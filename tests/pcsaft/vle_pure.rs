use approx::assert_relative_eq;
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::parameter::{IdentifierOption, Parameter};
use feos_core::si::*;
use feos_core::{Contributions, PhaseEquilibrium};
use std::error::Error;
use std::sync::Arc;

#[test]
fn vle_pure_temperature() -> Result<(), Box<dyn Error>> {
    let params = PcSaftParameters::from_json(
        vec!["propane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?;
    let saft = Arc::new(PcSaft::new(Arc::new(params)));
    let temperatures = [
        170.0 * KELVIN,
        200.0 * KELVIN,
        250.0 * KELVIN,
        300.0 * KELVIN,
        350.0 * KELVIN,
    ];
    for &t in temperatures.iter() {
        let state = PhaseEquilibrium::pure(&saft, t, None, Default::default())?;
        assert_relative_eq!(state.vapor().temperature, t, max_relative = 1e-10);
        assert_relative_eq!(
            state.vapor().pressure(Contributions::Total),
            state.liquid().pressure(Contributions::Total),
            max_relative = 1e-8
        );
    }
    Ok(())
}

#[test]
fn vle_pure_pressure() -> Result<(), Box<dyn Error>> {
    let params = PcSaftParameters::from_json(
        vec!["propane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?;
    let saft = Arc::new(PcSaft::new(Arc::new(params)));
    let pressures = [0.1 * BAR, 1.0 * BAR, 10.0 * BAR, 30.0 * BAR, 44.0 * BAR];
    for &p in pressures.iter() {
        let state = PhaseEquilibrium::pure(&saft, p, None, Default::default())?;
        println!(
            "liquid-p: {} vapor-p: {} p:{}",
            state.liquid().pressure(Contributions::Total),
            state.vapor().pressure(Contributions::Total),
            p
        );
        println!(
            "liquid-T: {} vapor-T: {}",
            state.liquid().temperature,
            state.vapor().temperature
        );
        assert_relative_eq!(
            state.liquid().pressure(Contributions::Total),
            p,
            max_relative = 1e-8
        );
        assert_relative_eq!(
            state.vapor().pressure(Contributions::Total),
            state.liquid().pressure(Contributions::Total),
            max_relative = 1e-8
        );
        assert_relative_eq!(
            state.vapor().temperature,
            state.liquid().temperature,
            max_relative = 1e-10
        );
    }
    Ok(())
}
