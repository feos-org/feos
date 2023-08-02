#![allow(clippy::excessive_precision)]
#![cfg(feature = "dft")]
use approx::assert_relative_eq;
use feos::hard_sphere::FMTVersion;
use feos::pcsaft::{PcSaft, PcSaftFunctional, PcSaftParameters};
use feos_core::joback::{Joback, JobackParameters};
use feos_core::parameter::{IdentifierOption, Parameter};
use feos_core::si::*;
use feos_core::{Contributions, PhaseEquilibrium, State, Verbosity};
use feos_dft::interface::PlanarInterface;
use feos_dft::DFTSolver;
use ndarray::{arr1, Axis};
use std::error::Error;
use std::sync::Arc;
use typenum::P3;

#[test]
#[allow(non_snake_case)]
fn test_bulk_implementations() -> Result<(), Box<dyn Error>> {
    // correct for different k_B in old code
    let KB_old = 1.38064852e-23 * JOULE / KELVIN;
    let NAV_old = 6.022140857e23 / MOL;

    let params = Arc::new(PcSaftParameters::from_json(
        vec!["water_np"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?);
    let eos = Arc::new(PcSaft::new(params.clone()));
    let func_pure = Arc::new(PcSaftFunctional::new(params.clone()));
    let func_full = Arc::new(PcSaftFunctional::new_full(
        params,
        FMTVersion::KierlikRosinberg,
    ));
    let t = 300.0 * KELVIN;
    let v = 0.002 * METER.powi::<P3>() * NAV / NAV_old;
    let n = arr1(&[1.5]) * MOL;
    let state = State::new_nvt(&eos, t, v, &n)?;
    let state_pure = State::new_nvt(&func_pure, t, v, &n)?;
    let state_full = State::new_nvt(&func_full, t, v, &n)?;
    let p = state.pressure_contributions();
    let p_pure = state_pure.pressure_contributions();
    let p_full = state_full.pressure_contributions();

    println!("{}: {}", p[0].0, p[0].1);
    println!("{}: {}", p_pure[0].0, p_pure[0].1);
    println!("{}: {}", p_full[0].0, p_full[0].1);
    println!();
    println!("{:20}: {}", p[1].0, p[1].1);
    println!("{:20}: {}", p_pure[1].0, p_pure[1].1);
    println!("{:20}: {}", p_full[1].0, p_full[1].1);
    println!();
    println!("{:21}: {}", p[2].0, p[2].1);
    println!("{:21}: {}", p_pure[2].0, p_pure[2].1);
    println!("{:21}: {}", p_full[2].0, p_full[2].1);
    println!();
    println!("{:21}: {}", p[3].0, p[3].1);
    println!("{:21}: {}", p_pure[3].0, p_pure[3].1 + p_pure[4].1);
    println!("{:21}: {}", p_full[3].0, p_full[3].1 + p_full[5].1);
    println!();
    println!("{:21}: {}", p[4].0, p[4].1);
    println!("{:21}: {}", p_full[4].0, p_full[4].1);

    let ideal_gas = 1.8707534688259309 * MEGA * PASCAL * KB / KB_old;
    assert_relative_eq!(p[0].1, ideal_gas, max_relative = 1e-14,);
    assert_relative_eq!(p_pure[0].1, ideal_gas, max_relative = 1e-14,);
    assert_relative_eq!(p_full[0].1, ideal_gas, max_relative = 1e-14,);

    let hard_sphere = 54.7102253882827583 * KILO * PASCAL * KB / KB_old;
    assert_relative_eq!(p[1].1, hard_sphere, max_relative = 1e-14,);
    assert_relative_eq!(p_full[1].1, hard_sphere, max_relative = 1e-14,);

    let hard_chains = -2.0847750028499230 * KILO * PASCAL * KB / KB_old;
    assert_relative_eq!(p[2].1, hard_chains, max_relative = 1e-14,);
    assert_relative_eq!(p_pure[2].1 + p_pure[4].1, hard_chains, max_relative = 2e-13,);
    assert_relative_eq!(p_full[2].1 + p_full[5].1, hard_chains, max_relative = 2e-13,);

    let dispersion = -262.895932352779993 * KILO * PASCAL * KB / KB_old;
    assert_relative_eq!(p[3].1, dispersion, max_relative = 1e-14,);
    assert_relative_eq!(p_pure[3].1, dispersion, max_relative = 1e-14,);
    assert_relative_eq!(p_full[3].1, dispersion, max_relative = 1e-14,);

    let association = -918.3899928262694630 * KILO * PASCAL * KB / KB_old;
    assert_relative_eq!(p[4].1, association, max_relative = 1e-14,);
    assert_relative_eq!(p_full[4].1, association, max_relative = 1e-14,);

    assert_relative_eq!(p_pure[1].1, hard_sphere + association, max_relative = 1e-14,);
    Ok(())
}

#[test]
#[allow(non_snake_case)]
fn test_dft_propane() -> Result<(), Box<dyn Error>> {
    // correct for different k_B in old code
    let KB_old = 1.38064852e-23 * JOULE / KELVIN;
    let NAV_old = 6.022140857e23 / MOL;

    let params = Arc::new(PcSaftParameters::from_json(
        vec!["propane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?);
    let func_pure = Arc::new(PcSaftFunctional::new(params.clone()));
    let func_full = Arc::new(PcSaftFunctional::new_full(
        params.clone(),
        FMTVersion::KierlikRosinberg,
    ));
    let func_full_vec = Arc::new(PcSaftFunctional::new_full(params, FMTVersion::WhiteBear));
    let t = 200.0 * KELVIN;
    let w = 150.0 * ANGSTROM;
    let points = 2048;
    let tc = State::critical_point(&func_pure, None, None, Default::default())?.temperature;
    let vle_pure = PhaseEquilibrium::pure(&func_pure, t, None, Default::default())?;
    let vle_full = PhaseEquilibrium::pure(&func_full, t, None, Default::default())?;
    let vle_full_vec = PhaseEquilibrium::pure(&func_full_vec, t, None, Default::default())?;
    let profile_pure = PlanarInterface::from_tanh(&vle_pure, points, w, tc, false).solve(None)?;
    let profile_full = PlanarInterface::from_tanh(&vle_full, points, w, tc, false).solve(None)?;
    let profile_full_vec =
        PlanarInterface::from_tanh(&vle_full_vec, points, w, tc, false).solve(None)?;
    let _ = func_pure.solve_pdgt(&vle_pure, 198, 0, None)?;
    println!(
        "pure {} {} {} {}",
        profile_pure.surface_tension.unwrap(),
        vle_pure.vapor().density,
        vle_pure.liquid().density,
        func_pure.solve_pdgt(&vle_pure, 198, 0, None)?.1
    );
    println!(
        "full {} {} {} {}",
        profile_full.surface_tension.unwrap(),
        vle_full.vapor().density,
        vle_full.liquid().density,
        func_full.solve_pdgt(&vle_full, 198, 0, None)?.1
    );
    println!(
        "vec  {} {} {} {}",
        profile_full_vec.surface_tension.unwrap(),
        vle_full_vec.vapor().density,
        vle_full_vec.liquid().density,
        func_full_vec.solve_pdgt(&vle_full_vec, 198, 0, None)?.1
    );

    let vapor_density = 12.2557486248527745 * MOL / METER.powi::<P3>() * NAV_old / NAV;
    assert_relative_eq!(
        vle_pure.vapor().density,
        vapor_density,
        max_relative = 1e-13,
    );
    assert_relative_eq!(
        vle_full.vapor().density,
        vapor_density,
        max_relative = 1e-13,
    );
    assert_relative_eq!(
        vle_full_vec.vapor().density,
        vapor_density,
        max_relative = 1e-13,
    );

    let liquid_density = 13.8941749145544549 * KILO * MOL / METER.powi::<P3>() * NAV_old / NAV;
    assert_relative_eq!(
        vle_pure.liquid().density,
        liquid_density,
        max_relative = 1e-13,
    );
    assert_relative_eq!(
        vle_full.liquid().density,
        liquid_density,
        max_relative = 1e-13,
    );
    assert_relative_eq!(
        vle_full_vec.liquid().density,
        liquid_density,
        max_relative = 1e-13,
    );

    let surface_tension = 19.9931025166113692 * MILLI * NEWTON / METER * KB / KB_old;
    let surface_tension_kr = 19.9863313312996169 * MILLI * NEWTON / METER * KB / KB_old;
    assert_relative_eq!(
        profile_pure.surface_tension.unwrap(),
        surface_tension,
        max_relative = 1e-4,
    );
    assert_relative_eq!(
        profile_full.surface_tension.unwrap(),
        surface_tension_kr,
        max_relative = 1e-4,
    );
    assert_relative_eq!(
        profile_full_vec.surface_tension.unwrap(),
        surface_tension,
        max_relative = 1e-4,
    );

    let surface_tension_pdgt = 20.2849756479219039 * MILLI * NEWTON / METER * KB / KB_old;
    let surface_tension_pdgt_kr = 20.2785079953823342 * MILLI * NEWTON / METER * KB / KB_old;
    assert_relative_eq!(
        func_pure.solve_pdgt(&vle_pure, 198, 0, None)?.1,
        surface_tension_pdgt,
        max_relative = 1e-10,
    );
    assert_relative_eq!(
        func_full.solve_pdgt(&vle_full, 198, 0, None)?.1,
        surface_tension_pdgt_kr,
        max_relative = 1e-10,
    );
    assert_relative_eq!(
        func_full_vec.solve_pdgt(&vle_full_vec, 198, 0, None)?.1,
        surface_tension_pdgt,
        max_relative = 1e-10,
    );
    Ok(())
}

#[test]
#[allow(non_snake_case)]
fn test_dft_propane_newton() -> Result<(), Box<dyn Error>> {
    let params = Arc::new(PcSaftParameters::from_json(
        vec!["propane"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?);
    let func = Arc::new(PcSaftFunctional::new(params));
    let t = 200.0 * KELVIN;
    let w = 150.0 * ANGSTROM;
    let points = 512;
    let tc = State::critical_point(&func, None, None, Default::default())?.temperature;
    let vle = PhaseEquilibrium::pure(&func, t, None, Default::default())?;
    let solver = DFTSolver::new(Some(Verbosity::Iter)).newton(None, None, None, None);
    PlanarInterface::from_tanh(&vle, points, w, tc, false).solve(Some(&solver))?;
    let solver = DFTSolver::new(Some(Verbosity::Iter)).newton(Some(true), None, None, None);
    PlanarInterface::from_tanh(&vle, points, w, tc, false).solve(Some(&solver))?;
    Ok(())
}

#[test]
#[allow(non_snake_case)]
fn test_dft_water() -> Result<(), Box<dyn Error>> {
    // correct for different k_B in old code
    let KB_old = 1.38064852e-23 * JOULE / KELVIN;
    let NAV_old = 6.022140857e23 / MOL;

    let params = Arc::new(PcSaftParameters::from_json(
        vec!["water_np"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?);
    let func_pure = Arc::new(PcSaftFunctional::new(params.clone()));
    let func_full_vec = Arc::new(PcSaftFunctional::new_full(params, FMTVersion::WhiteBear));
    let t = 400.0 * KELVIN;
    let w = 120.0 * ANGSTROM;
    let points = 2048;
    let tc = State::critical_point(&func_pure, None, None, Default::default())?.temperature;
    let vle_pure = PhaseEquilibrium::pure(&func_pure, t, None, Default::default())?;
    let vle_full_vec = PhaseEquilibrium::pure(&func_full_vec, t, None, Default::default())?;
    let profile_pure = PlanarInterface::from_tanh(&vle_pure, points, w, tc, false).solve(None)?;
    let profile_full_vec =
        PlanarInterface::from_tanh(&vle_full_vec, points, w, tc, false).solve(None)?;
    println!(
        "pure {} {} {}",
        profile_pure.surface_tension.unwrap(),
        vle_pure.vapor().density,
        vle_pure.liquid().density
    );
    println!(
        "vec  {} {} {}",
        profile_full_vec.surface_tension.unwrap(),
        vle_full_vec.vapor().density,
        vle_full_vec.liquid().density
    );

    let vapor_density = 75.8045715345905222 * MOL / METER.powi::<P3>() * NAV_old / NAV;
    assert_relative_eq!(
        vle_pure.vapor().density,
        vapor_density,
        max_relative = 1e-13,
    );
    assert_relative_eq!(
        vle_full_vec.vapor().density,
        vapor_density,
        max_relative = 1e-13,
    );

    let liquid_density = 47.8480850281608454 * KILO * MOL / METER.powi::<P3>() * NAV_old / NAV;
    assert_relative_eq!(
        vle_pure.liquid().density,
        liquid_density,
        max_relative = 1e-13,
    );
    assert_relative_eq!(
        vle_full_vec.liquid().density,
        liquid_density,
        max_relative = 1e-13,
    );

    let surface_tension = 70.1419567481980408 * MILLI * NEWTON / METER * KB / KB_old;
    assert_relative_eq!(
        profile_pure.surface_tension.unwrap(),
        surface_tension,
        max_relative = 1e-4,
    );
    assert_relative_eq!(
        profile_full_vec.surface_tension.unwrap(),
        surface_tension,
        max_relative = 1e-4,
    );

    let surface_tension_pdgt = 72.9496195135527188 * MILLI * NEWTON / METER * KB / KB_old;
    assert_relative_eq!(
        func_pure.solve_pdgt(&vle_pure, 198, 0, None)?.1,
        surface_tension_pdgt,
        max_relative = 1e-10,
    );
    assert_relative_eq!(
        func_full_vec.solve_pdgt(&vle_full_vec, 198, 0, None)?.1,
        surface_tension_pdgt,
        max_relative = 1e-10,
    );
    Ok(())
}

#[test]
fn test_entropy_bulk_values() -> Result<(), Box<dyn Error>> {
    let params = PcSaftParameters::from_json(
        vec!["water_np"],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )?;
    let joback_params = JobackParameters::from_json(
        vec!["water_np"],
        "tests/pcsaft/test_parameters_joback.json",
        None,
        IdentifierOption::Name,
    )?;
    let joback = Joback::new(Arc::new(joback_params));
    let func = Arc::new(PcSaftFunctional::new(Arc::new(params)).ideal_gas(joback));
    let vle = PhaseEquilibrium::pure(&func, 350.0 * KELVIN, None, Default::default())?;
    let profile = PlanarInterface::from_pdgt(&vle, 2048, false)?.solve(None)?;
    let s_res = profile.profile.entropy_density(Contributions::Residual)?;
    let s_tot = profile.profile.entropy_density(Contributions::Total)?;
    println!(
        "Density:\n{}",
        profile.profile.density.index_axis(Axis(0), 0).to_owned()
    );
    println!(
        "liquid: {}, vapor: {}",
        profile.vle.liquid().density,
        profile.vle.vapor().density
    );
    println!("\nResidual:\n{:?}", s_res);
    println!(
        "liquid: {:?}, vapor: {:?}",
        profile.vle.liquid().entropy(Contributions::Residual) / profile.vle.liquid().volume,
        profile.vle.vapor().entropy(Contributions::Residual) / profile.vle.vapor().volume
    );
    println!("\nTotal:\n{:?}", s_tot);
    println!(
        "liquid: {:?}, vapor: {:?}",
        profile.vle.liquid().entropy(Contributions::Total) / profile.vle.liquid().volume,
        profile.vle.vapor().entropy(Contributions::Total) / profile.vle.vapor().volume
    );
    assert_relative_eq!(
        s_res.get(0),
        profile.vle.liquid().entropy(Contributions::Residual) / profile.vle.liquid().volume,
        max_relative = 1e-8,
    );
    assert_relative_eq!(
        s_res.get(2047),
        profile.vle.vapor().entropy(Contributions::Residual) / profile.vle.vapor().volume,
        max_relative = 1e-8,
    );
    assert_relative_eq!(
        s_tot.get(0),
        profile.vle.liquid().entropy(Contributions::Total) / profile.vle.liquid().volume,
        max_relative = 1e-8,
    );
    assert_relative_eq!(
        s_tot.get(2047),
        profile.vle.vapor().entropy(Contributions::Total) / profile.vle.vapor().volume,
        max_relative = 1e-8,
    );
    Ok(())
}
