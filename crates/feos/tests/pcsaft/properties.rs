use approx::assert_relative_eq;
use feos::pcsaft::{PcSaft, PcSaftParameters};
use feos_core::parameter::IdentifierOption;
use feos_core::DensityInitialization::{Liquid, Vapor};
use feos_core::{Contributions, PhaseEquilibrium, Residual, State};
use nalgebra::dvector;
use quantity::*;
use std::error::Error;
use std::sync::Arc;

#[test]
fn test_dln_phi_dp() -> Result<(), Box<dyn Error>> {
    let params = PcSaftParameters::from_json(
        vec!["propane", "butane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?;
    let saft = PcSaft::new(params);
    let t = 300.0 * KELVIN;
    let p = BAR;
    let h = 1e-1 * PASCAL;
    let s = State::new_npt(&&saft, t, p, dvector![0.5, 0.5], Some(Vapor))?;
    let sh = State::new_npt(&&saft, t, p + h, dvector![0.5, 0.5], Some(Vapor))?;

    let ln_phi = s.ln_phi()[0];
    let ln_phi_h = sh.ln_phi()[0];
    let dln_phi_dp = s.dln_phi_dp().get(0);
    let dln_phi_dp_h = (ln_phi_h - ln_phi) / h;
    assert_relative_eq!(dln_phi_dp, dln_phi_dp_h, max_relative = 1e-6);
    Ok(())
}

#[test]
fn test_virial_is_not_nan() -> Result<(), Box<dyn Error>> {
    let params = PcSaftParameters::from_json(
        vec!["water_np"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?;
    let saft = &PcSaft::new(params);
    let virial_b = saft.second_virial_coefficient(300.0 * KELVIN, ())?;
    assert!(!virial_b.is_nan());
    Ok(())
}

fn esper2023(components: Vec<&str>) -> Result<PcSaft, Box<dyn Error>> {
    Ok(PcSaft::new(PcSaftParameters::from_json(
        components,
        "../../parameters/pcsaft/esper2023.json",
        Some("../../parameters/pcsaft/rehner2023_binary.json"),
        IdentifierOption::Name,
    )?))
}

#[test]
fn test_thermodynamic_factor_component_order() -> Result<(), Box<dyn Error>> {
    let ew = esper2023(vec!["ethanol", "water"])?;
    let we = esper2023(vec!["water", "ethanol"])?;
    let t = 300.0 * KELVIN;
    for x in [0.2, 0.5, 1e-6] {
        let s1 = State::new_npt(&&ew, t, BAR, dvector![x, 1.0 - x], Some(Liquid))?;
        let s2 = State::new_npt(&&we, t, BAR, dvector![1.0 - x, x], Some(Liquid))?;
        let g1 = s1.thermodynamic_factor()[(0, 0)];
        let g2 = s2.thermodynamic_factor()[(0, 0)];
        assert_relative_eq!(g1, g2, max_relative = 1e-6);
    }
    let s = State::new_npt(&&ew, t, BAR, dvector![1e-8, 1.0 - 1e-8], Some(Liquid))?;
    assert_relative_eq!(s.thermodynamic_factor()[(0, 0)], 1.0, epsilon = 1e-6);
    Ok(())
}

#[test]
fn test_henrys_law_constant_mixed_solvent() -> Result<(), Box<dyn Error>> {
    let eos = Arc::new(esper2023(vec!["methane", "propane", "hexane"])?);
    let t = 300.0 * KELVIN;
    let h = State::henrys_law_constant(&eos, t, &dvector![0.0, 0.5, 0.5])?[0];
    let x1 = 1e-8;
    let z = dvector![x1, 0.5 * (1.0 - x1), 0.5 * (1.0 - x1)];
    let vle = PhaseEquilibrium::bubble_point(&eos, t, &z, None, None, Default::default())?;
    let h_lim = vle.vapor().pressure(Contributions::Total) * vle.vapor().molefracs[0] / x1;
    assert_relative_eq!(h, h_lim, max_relative = 1e-6);
    Ok(())
}
