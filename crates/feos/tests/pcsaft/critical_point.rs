use approx::assert_relative_eq;
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::parameter::IdentifierOption;
use feos_core::{SolverOptions, State};
use nalgebra::dvector;
use quantity::*;
use std::error::Error;
use std::sync::Arc;

#[test]
fn test_critical_point_pure() -> Result<(), Box<dyn Error>> {
    let params = PcSaftParameters::from_json(
        vec!["propane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?;
    let saft = PcSaft::new(params);
    let t = 300.0 * KELVIN;
    let cp = State::critical_point(&&saft, (), Some(t), None, Default::default())?;
    assert_relative_eq!(cp.temperature, 375.12441 * KELVIN, max_relative = 1e-8);
    assert_relative_eq!(
        cp.density,
        4733.00377 * MOL / METER.powi::<3>(),
        max_relative = 1e-6
    );
    Ok(())
}

#[test]
fn test_critical_point_mix() -> Result<(), Box<dyn Error>> {
    let params = PcSaftParameters::from_json(
        vec!["propane", "butane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?;
    let saft = PcSaft::new(params);
    let t = 300.0 * KELVIN;
    let molefracs = dvector![0.5, 0.5];
    let cp = State::critical_point(&&saft, molefracs, Some(t), None, Default::default())?;
    assert_relative_eq!(cp.temperature, 407.93481 * KELVIN, max_relative = 1e-8);
    assert_relative_eq!(
        cp.density,
        4265.50745 * MOL / METER.powi::<3>(),
        max_relative = 1e-6
    );
    Ok(())
}

#[test]
fn test_critical_point_limits() -> Result<(), Box<dyn Error>> {
    let params = PcSaftParameters::from_json(
        vec!["propane", "butane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?;
    let options = SolverOptions {
        verbosity: feos_core::Verbosity::Iter,
        ..Default::default()
    };
    let saft = Arc::new(PcSaft::new(params));
    let cp_pure = State::critical_point_pure(&saft, None, None, options)?;
    println!("{} {}", cp_pure[0], cp_pure[1]);
    let molefracs = dvector![0.0, 1.0];
    let cp_2 = State::critical_point(&saft, &molefracs, None, None, options)?;
    println!("{}", cp_2);
    let molefracs = dvector![1.0, 0.0];
    let cp_1 = State::critical_point(&saft, &molefracs, None, None, options)?;
    println!("{}", cp_1);
    assert_eq!(cp_pure[0].temperature, cp_1.temperature);
    assert_eq!(cp_pure[0].density, cp_1.density);
    assert_eq!(cp_pure[1].temperature, cp_2.temperature);
    assert_eq!(cp_pure[1].density, cp_2.density);
    Ok(())
}
