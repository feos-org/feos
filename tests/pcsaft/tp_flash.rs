use approx::assert_relative_eq;
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::parameter::{IdentifierOption, Parameter, ParameterError};
use feos_core::si::*;
use feos_core::{Contributions, PhaseEquilibrium, SolverOptions};
use ndarray::*;
use std::error::Error;
use std::sync::Arc;

fn read_params(components: Vec<&str>) -> Result<Arc<PcSaftParameters>, ParameterError> {
    Ok(Arc::new(PcSaftParameters::from_json(
        components,
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?))
}

#[test]
fn test_tp_flash() -> Result<(), Box<dyn Error>> {
    let propane = Arc::new(PcSaft::new(read_params(vec!["propane"])?));
    let butane = Arc::new(PcSaft::new(read_params(vec!["butane"])?));
    let t = 250.0 * KELVIN;
    let p_propane = PhaseEquilibrium::pure(&propane, t, None, Default::default())?
        .vapor()
        .pressure(Contributions::Total);
    let p_butane = PhaseEquilibrium::pure(&butane, t, None, Default::default())?
        .vapor()
        .pressure(Contributions::Total);
    let x1 = 0.5;
    let p = x1 * p_propane + (1.0 - x1) * p_butane;
    let y1 = (x1 * p_propane / p).into_value();
    let z1 = 0.5 * (x1 + y1);
    println!("{} {} {} {} {}", p_propane, p_butane, x1, y1, z1);
    let mix = Arc::new(PcSaft::new(read_params(vec!["propane", "butane"])?));
    let options = SolverOptions::new().max_iter(100).tol(1e-12);
    let vle = PhaseEquilibrium::tp_flash(
        &mix,
        t,
        p,
        &(arr1(&[z1, 1.0 - z1]) * MOL),
        None,
        options,
        None,
    )?;
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
        &vle.vapor().molefracs * &vle.vapor().ln_phi().mapv(f64::exp),
        &vle.liquid().molefracs * &vle.liquid().ln_phi().mapv(f64::exp),
        max_relative = 1e-10
    );
    Ok(())
}
