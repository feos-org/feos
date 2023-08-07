use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::parameter::{IdentifierOption, Parameter};
use feos_core::si::*;
use feos_core::{DensityInitialization, PhaseEquilibrium, State};
use ndarray::arr1;
use std::error::Error;
use std::sync::Arc;

#[test]
fn test_stability_analysis() -> Result<(), Box<dyn Error>> {
    let params = PcSaftParameters::from_json(
        vec!["water_np", "hexane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?;
    let mix = Arc::new(PcSaft::new(Arc::new(params)));
    let unstable = State::new_npt(
        &mix,
        300.0 * KELVIN,
        1.0 * BAR,
        &(arr1(&[0.5, 0.5]) * MOL),
        DensityInitialization::Liquid,
    )?;
    let check = unstable.stability_analysis(Default::default())?;
    assert!(!check.is_empty());

    let params = PcSaftParameters::from_json(
        vec!["propane", "butane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?;
    let mix = Arc::new(PcSaft::new(Arc::new(params)));
    let vle = PhaseEquilibrium::bubble_point(
        &mix,
        300.0 * KELVIN,
        &arr1(&[0.5, 0.5]),
        Some(6.0 * BAR),
        None,
        Default::default(),
    )?;
    let vapor_check = vle.vapor().stability_analysis(Default::default())?;
    let liquid_check = vle.liquid().stability_analysis(Default::default())?;
    assert!(vapor_check.is_empty());
    assert!(liquid_check.is_empty());
    Ok(())
}
