use approx::assert_relative_eq;
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::parameter::IdentifierOption;
use feos_core::{Contributions, FeosResult, PhaseEquilibrium, SolverOptions};
use nalgebra::dvector;
use quantity::*;

fn read_params(components: Vec<&str>) -> FeosResult<PcSaftParameters> {
    PcSaftParameters::from_json(
        components,
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )
}

#[test]
fn test_tp_flash() -> FeosResult<()> {
    let propane = PcSaft::new(read_params(vec!["propane"])?);
    let butane = PcSaft::new(read_params(vec!["butane"])?);
    let t = 250.0 * KELVIN;
    let p_propane = PhaseEquilibrium::pure(&&propane, t, None, Default::default())?
        .vapor()
        .pressure(Contributions::Total);
    let p_butane = PhaseEquilibrium::pure(&&butane, t, None, Default::default())?
        .vapor()
        .pressure(Contributions::Total);
    let x1 = 0.5;
    let p = x1 * p_propane + (1.0 - x1) * p_butane;
    let y1 = (x1 * p_propane / p).into_value();
    let z1 = 0.5 * (x1 + y1);
    println!("{p_propane} {p_butane} {x1} {y1} {z1}");
    let mix = PcSaft::new(read_params(vec!["propane", "butane"])?);
    let options = SolverOptions::new().max_iter(100).tol(1e-12);
    let vle = PhaseEquilibrium::tp_flash(&&mix, t, p, z1, None, options, None)?;
    println!(
        "x1: {}, y1: {}",
        vle.liquid().molefracs[0],
        vle.vapor().molefracs[0]
    );
    assert_relative_eq!(
        vle.vapor().pressure(Contributions::Total),
        vle.liquid().pressure(Contributions::Total),
        max_relative = 1e-10
    );
    assert_relative_eq!(
        &vle.vapor()
            .molefracs
            .component_mul(&vle.vapor().ln_phi().map(f64::exp)),
        &vle.liquid()
            .molefracs
            .component_mul(&vle.liquid().ln_phi().map(f64::exp)),
        max_relative = 1e-10
    );
    Ok(())
}

#[test]
fn test_tp_flash_zero_component() -> FeosResult<()> {
    let eos_full = PcSaft::new(PcSaftParameters::from_json(
        vec!["propane", "butane", "hexane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?);
    let eos_binary = PcSaft::new(PcSaftParameters::from_json(
        vec!["butane", "hexane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?);
    let options = SolverOptions {
        verbosity: feos_core::Verbosity::Iter,
        ..Default::default()
    };
    let vle_full = PhaseEquilibrium::tp_flash(
        &&eos_full,
        300.0 * KELVIN,
        1.2 * BAR,
        dvector![0.0, 0.5, 0.5],
        None,
        options,
        None,
    )?;
    let vle_binary = PhaseEquilibrium::tp_flash(
        &&eos_binary,
        300.0 * KELVIN,
        1.2 * BAR,
        dvector![0.5, 0.5],
        None,
        options,
        None,
    )?;
    println!("{vle_full}\n{vle_binary}");
    assert_eq!(
        vle_full.liquid().molefracs[1],
        vle_binary.liquid().molefracs[0]
    );
    assert_eq!(
        vle_full.vapor().molefracs[1],
        vle_binary.vapor().molefracs[0]
    );
    Ok(())
}

#[test]
fn test_tp_flash_acceleration() -> FeosResult<()> {
    // With working acceleration, this LLE converges in ~36 steps (previously ~180).
    let eos = PcSaft::new(PcSaftParameters::from_json(
        vec!["methanol", "hexane"],
        "../../parameters/pcsaft/esper2023.json",
        Some("../../parameters/pcsaft/rehner2023_binary.json"),
        IdentifierOption::Name,
    )?);
    let options = SolverOptions::new().max_iter(15).tol(1e-12);
    let vle = PhaseEquilibrium::tp_flash(
        &&eos,
        280.0 * KELVIN,
        BAR,
        &dvector![0.5, 0.5],
        None,
        options,
        None,
    )?;
    assert!((vle.vapor().molefracs[0] - vle.liquid().molefracs[0]).abs() > 0.1);
    Ok(())
}

#[test]
fn test_tp_flash_initial_state_new_feed() -> FeosResult<()> {
    let eos = PcSaft::new(PcSaftParameters::from_json(
        vec!["propane", "n-butane", "pentane"],
        "../../parameters/pcsaft/esper2023.json",
        Some("../../parameters/pcsaft/rehner2023_binary.json"),
        IdentifierOption::Name,
    )?);
    let (t, p) = (300.0 * KELVIN, 4.0 * BAR);
    let options = SolverOptions::new().tol(1e-12);
    let mut vle: Option<PhaseEquilibrium<_, 2>> = None;
    for z1 in [0.4, 0.5, 0.6, 0.7] {
        let z = dvector![z1, 0.5 * (1.0 - z1), 0.5 * (1.0 - z1)];
        let cold = PhaseEquilibrium::tp_flash(&&eos, t, p, &z, None, options, None)?;
        let warm = PhaseEquilibrium::tp_flash(&&eos, t, p, &z, vle.as_ref(), options, None)?;
        assert_relative_eq!(
            warm.vapor_phase_fraction(),
            cold.vapor_phase_fraction(),
            max_relative = 1e-8
        );
        assert_relative_eq!(
            warm.liquid().molefracs[0],
            cold.liquid().molefracs[0],
            max_relative = 1e-8
        );
        vle = Some(warm);
    }
    Ok(())
}
