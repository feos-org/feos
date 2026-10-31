use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::parameter::IdentifierOption;
use feos_core::{DensityInitialization, PhaseEquilibrium, Residual, SolverOptions, State};
use nalgebra::dvector;
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

#[test]
fn test_stability_analysis_failed_trial() -> Result<(), Box<dyn Error>> {
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

    // the second liquid trial phase needs more than 4 iterations; the
    // remaining trial phases still find the instability
    let options = SolverOptions {
        max_iter: Some(4),
        ..Default::default()
    };
    let check = unstable.stability_analysis(options)?;
    assert!(!check.is_empty());

    // after a single iteration, the minimizations are not converged, but the
    // negative tangent plane distances already prove the instability
    let options = SolverOptions {
        max_iter: Some(1),
        ..Default::default()
    };
    let check = unstable.stability_analysis(options)?;
    assert_eq!(check.len(), 2);

    // if the minimization fails for all trial phases without finding a
    // negative tangent plane distance, the error is returned
    let params = PcSaftParameters::from_json(
        vec!["propane", "butane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?;
    let mix = PcSaft::new(params);
    let stable = State::new_npt(
        &&mix,
        300.0 * KELVIN,
        20.0 * BAR,
        0.5,
        Some(DensityInitialization::Liquid),
    )?;
    assert!(stable.stability_analysis(options).is_err());
    Ok(())
}

#[test]
fn test_stability_analysis_sorted() -> Result<(), Box<dyn Error>> {
    let params = PcSaftParameters::from_json(
        vec!["water_np", "hexane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?;
    let mix = PcSaft::new(params);
    let unstable = State::new_npt(
        &&mix,
        350.0 * KELVIN,
        1.0 * BAR,
        0.5,
        Some(DensityInitialization::Liquid),
    )?;

    // the trial phases find a water-rich liquid, a hexane-rich liquid and a
    // vapor (in that order); the vapor has the most negative tpd
    let check = unstable.stability_analysis(SolverOptions::default())?;
    assert_eq!(check.len(), 3);
    let tpds: Vec<_> = check.iter().map(|s| tpd(&unstable, s)).collect();
    assert!(tpds.is_sorted());
    assert!(check[0].density < check[1].density);
    assert!(check[0].density < check[2].density);

    // converged and not converged candidates are sorted by the same measure
    let unstable = State::new_npt(
        &&mix,
        300.0 * KELVIN,
        5.0 * BAR,
        0.1,
        Some(DensityInitialization::Vapor),
    )?;
    let options = SolverOptions {
        max_iter: Some(3),
        ..Default::default()
    };
    let check = unstable.stability_analysis(options)?;
    assert_eq!(check.len(), 2);
    let tpds: Vec<_> = check.iter().map(|s| tpd(&unstable, s)).collect();
    assert!(tpds.is_sorted());
    Ok(())
}

#[test]
fn test_stability_analysis_zero_component() -> Result<(), Box<dyn Error>> {
    let params = PcSaftParameters::from_json(
        vec!["propane", "butane", "hexane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?;
    let mix = PcSaft::new(params);
    let unstable = State::new_npt(
        &&mix,
        300.0 * KELVIN,
        1.2 * BAR,
        &dvector![0.0, 0.5, 0.5],
        Some(DensityInitialization::Liquid),
    )?;

    // absent components must not turn the tpd into NaN, also for
    // trial phases that are not converged
    for max_iter in [1, 100] {
        let options = SolverOptions {
            max_iter: Some(max_iter),
            ..Default::default()
        };
        let check = unstable.stability_analysis(options)?;
        assert!(!check.is_empty());
        assert!(check.iter().all(|s| tpd(&unstable, s) < 0.0));
    }
    Ok(())
}

#[test]
fn test_stability_analysis_duplicates() -> Result<(), Box<dyn Error>> {
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
        20.0 * BAR,
        0.9,
        Some(DensityInitialization::Liquid),
    )?;

    // the hexane-rich liquid is found from two trial phases; it only differs
    // in the (trace) water content and must be reported once
    let check = unstable.stability_analysis(SolverOptions::default())?;
    assert_eq!(check.len(), 2);
    Ok(())
}

fn tpd<E: Residual>(feed: &State<E>, trial: &State<E>) -> f64 {
    let d = feed.molefracs.map(f64::ln) + feed.ln_phi();
    let ln_phi = trial.ln_phi();
    let w = &trial.molefracs;
    (0..w.len())
        .filter(|&i| w[i] > 0.0)
        .map(|i| w[i] * (w[i].ln() + ln_phi[i] - d[i]))
        .sum()
}
