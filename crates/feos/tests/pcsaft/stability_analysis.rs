use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::parameter::IdentifierOption;
use feos_core::{DensityInitialization, PhaseEquilibrium, SolverOptions, State};
use quantity::*;
use std::error::Error;

#[test]
fn test_stability_analysis() -> Result<(), Box<dyn Error>> {
    let params = PcSaftParameters::from_json(
        vec!["water_np", "hexane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?;
    let mix = PcSaft::new(params);
    let unstable = State::new_npt(
        &&mix,
        300.0 * KELVIN,
        1.0 * BAR,
        0.5,
        Some(DensityInitialization::Liquid),
    )?;
    let options = SolverOptions {
        verbosity: feos_core::Verbosity::Iter,
        ..Default::default()
    };
    let check = unstable.stability_analysis(options)?;
    assert!(!check.is_empty());

    let params = PcSaftParameters::from_json(
        vec!["propane", "butane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?;
    let mix = PcSaft::new(params);
    let vle = PhaseEquilibrium::bubble_point(
        &&mix,
        300.0 * KELVIN,
        0.5,
        Some(6.0 * BAR),
        None,
        (options, options),
    )?;
    let vapor_check = vle.vapor().stability_analysis(options)?;
    let liquid_check = vle.liquid().stability_analysis(options)?;
    assert!(vapor_check.is_empty());
    assert!(liquid_check.is_empty());
    Ok(())
}
