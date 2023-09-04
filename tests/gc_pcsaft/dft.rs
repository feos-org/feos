#![allow(clippy::excessive_precision)]
#![cfg(feature = "dft")]
use approx::assert_relative_eq;
use feos::gc_pcsaft::{
    GcPcSaft, GcPcSaftEosParameters, GcPcSaftFunctional, GcPcSaftFunctionalParameters,
};
use feos_core::parameter::{
    ChemicalRecord, Identifier, IdentifierOption, ParameterHetero, SegmentRecord,
};
use feos_core::si::*;
use feos_core::{PhaseEquilibrium, State, StateBuilder, Verbosity};
use feos_dft::adsorption::{ExternalPotential, Pore1D, PoreSpecification};
use feos_dft::interface::PlanarInterface;
use feos_dft::{DFTSolver, Geometry};
use ndarray::arr1;
use std::error::Error;
use std::sync::Arc;
use typenum::P3;

#[test]
#[allow(non_snake_case)]
fn test_bulk_implementation() -> Result<(), Box<dyn Error>> {
    // correct for different k_B in old code
    let KB_old = 1.38064852e-23 * JOULE / KELVIN;
    let NAV_old = 6.022140857e23 / MOL;

    let parameters = GcPcSaftEosParameters::from_json_segments(
        &["propane"],
        "parameters/pcsaft/gc_substances.json",
        "parameters/pcsaft/sauer2014_hetero.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();

    let parameters_func = GcPcSaftFunctionalParameters::from_json_segments(
        &["propane"],
        "parameters/pcsaft/gc_substances.json",
        "parameters/pcsaft/sauer2014_hetero.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();

    let eos = Arc::new(GcPcSaft::new(Arc::new(parameters)));
    let func = Arc::new(GcPcSaftFunctional::new(Arc::new(parameters_func)));
    let t = 200.0 * KELVIN;
    let v = 0.002 * METER.powi::<P3>() * NAV / NAV_old;
    let n = arr1(&[1.5]) * MOL;
    let state_eos = State::new_nvt(&eos, t, v, &n)?;
    let state_func = State::new_nvt(&func, t, v, &n)?;
    let p_eos = state_eos.pressure_contributions();
    let p_func = state_func.pressure_contributions();

    println!("{:29}: {}", p_eos[0].0, p_eos[0].1);
    println!("{:29}: {}", p_func[0].0, p_func[0].1);
    println!();
    println!("{:29}: {}", p_eos[1].0, p_eos[1].1);
    println!("{:29}: {}", p_func[1].0, p_func[1].1);
    println!();
    println!("{:29}: {}", p_eos[2].0, p_eos[2].1);
    println!("{:29}: {}", p_func[2].0, p_func[2].1);
    println!();
    println!("{:29}: {}", p_eos[3].0, p_eos[3].1);
    println!("{:29}: {}", p_func[3].0, p_func[3].1);

    assert_relative_eq!(
        p_eos[0].1,
        1.2471689792172869 * MEGA * PASCAL * KB / KB_old,
        max_relative = 1e-14,
    );
    assert_relative_eq!(
        p_eos[1].1,
        280.0635060891395938 * KILO * PASCAL * KB / KB_old,
        max_relative = 1e-14,
    );
    assert_relative_eq!(
        p_eos[2].1,
        -141.9023918353318550 * KILO * PASCAL * KB / KB_old,
        max_relative = 1e-14,
    );
    assert_relative_eq!(
        p_eos[3].1,
        -763.2289230004602132 * KILO * PASCAL * KB / KB_old,
        max_relative = 1e-14,
    );

    assert_relative_eq!(
        p_func[0].1,
        1.2471689792172869 * MEGA * PASCAL * KB / KB_old,
        max_relative = 1e-14,
    );
    assert_relative_eq!(
        p_func[1].1,
        280.0635060891395938 * KILO * PASCAL * KB / KB_old,
        max_relative = 1e-14,
    );
    assert_relative_eq!(
        p_func[2].1,
        -141.9023918353318550 * KILO * PASCAL * KB / KB_old,
        max_relative = 1e-14,
    );
    assert_relative_eq!(
        p_func[3].1,
        -763.2289230004602132 * KILO * PASCAL * KB / KB_old,
        max_relative = 1e-14,
    );
    Ok(())
}

#[test]
fn test_bulk_association() -> Result<(), Box<dyn Error>> {
    let segment_records = SegmentRecord::from_json("parameters/pcsaft/sauer2014_hetero.json")?;
    let ethylene_glycol = ChemicalRecord::new(
        Identifier::default(),
        vec!["OH".into(), "CH2".into(), "CH2".into(), "OH".into()],
        None,
    );
    let eos_parameters = Arc::new(GcPcSaftEosParameters::from_segments(
        vec![ethylene_glycol.clone()],
        segment_records.clone(),
        None,
    )?);
    let eos = Arc::new(GcPcSaft::new(eos_parameters));
    let func_parameters = Arc::new(GcPcSaftFunctionalParameters::from_segments(
        vec![ethylene_glycol],
        segment_records,
        None,
    )?);
    let func = Arc::new(GcPcSaftFunctional::new(func_parameters));

    let t = 200.0 * KELVIN;
    let v = 0.002 * METER.powi::<P3>();
    let n = arr1(&[1.5]) * MOL;
    let state_eos = State::new_nvt(&eos, t, v, &n)?;
    let state_func = State::new_nvt(&func, t, v, &n)?;
    let p_eos = state_eos.pressure_contributions();
    let p_func = state_func.pressure_contributions();
    for (s, x) in &p_eos {
        println!("{s:18}: {x:21.16}");
    }
    for (s, x) in &p_func {
        println!("{s:26}: {x:21.16}");
    }
    assert_relative_eq!(p_eos[4].1, p_func[4].1, max_relative = 1e-14);
    Ok(())
}

#[test]
#[allow(non_snake_case)]
fn test_dft() -> Result<(), Box<dyn Error>> {
    // correct for different k_B in old code
    let KB_old = 1.38064852e-23 * JOULE / KELVIN;
    let NAV_old = 6.022140857e23 / MOL;

    let parameters = GcPcSaftFunctionalParameters::from_json_segments(
        &["propane"],
        "parameters/pcsaft/gc_substances.json",
        "parameters/pcsaft/sauer2014_hetero.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();

    let func = Arc::new(GcPcSaftFunctional::new(Arc::new(parameters)));
    let t = 200.0 * KELVIN;
    let w = 150.0 * ANGSTROM;
    let points = 2048;
    let tc = State::critical_point(&func, None, None, Default::default())?.temperature;
    let vle = PhaseEquilibrium::pure(&func, t, None, Default::default())?;
    let profile = PlanarInterface::from_tanh(&vle, points, w, tc, false).solve(None)?;
    println!(
        "hetero {} {} {}",
        profile.surface_tension.unwrap(),
        vle.vapor().density,
        vle.liquid().density,
    );

    assert_relative_eq!(
        vle.vapor().density,
        12.8820179191167643 * MOL / METER.powi::<P3>() * NAV_old / NAV,
        max_relative = 1e-13,
    );

    assert_relative_eq!(
        vle.liquid().density,
        13.2705903446123212 * KILO * MOL / METER.powi::<P3>() * NAV_old / NAV,
        max_relative = 1e-13,
    );

    assert_relative_eq!(
        profile.surface_tension.unwrap(),
        21.1478901897016272 * MILLI * NEWTON / METER * KB / KB_old,
        max_relative = 6e-5,
    );
    Ok(())
}

#[test]
#[allow(non_snake_case)]
fn test_dft_assoc() -> Result<(), Box<dyn Error>> {
    let parameters = GcPcSaftFunctionalParameters::from_json_segments(
        &["1-pentanol"],
        "parameters/pcsaft/gc_substances.json",
        "parameters/pcsaft/sauer2014_hetero.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();

    let func = Arc::new(GcPcSaftFunctional::new(Arc::new(parameters)));
    let t = 300.0 * KELVIN;
    let w = 100.0 * ANGSTROM;
    let points = 4096;
    let vle = PhaseEquilibrium::pure(&func, t, None, Default::default())?;
    let profile = PlanarInterface::from_tanh(&vle, points, w, 600.0 * KELVIN, false).solve(None)?;
    println!(
        "hetero {} {} {}",
        profile.surface_tension.unwrap(),
        vle.vapor().density,
        vle.liquid().density,
    );

    let solver = DFTSolver::new(Some(Verbosity::Iter))
        .picard_iteration(None, None, Some(1e-5), Some(0.05))
        .anderson_mixing(None, None, None, None, None);
    let bulk = StateBuilder::new(&func)
        .temperature(t)
        .pressure(5.0 * BAR)
        .build()?;
    Pore1D::new(
        Geometry::Cartesian,
        20.0 * ANGSTROM,
        ExternalPotential::LJ93 {
            epsilon_k_ss: 10.0,
            sigma_ss: 3.0,
            rho_s: 0.08,
        },
        None,
        None,
    )
    .initialize(&bulk, None, None)
    .unwrap()
    .solve(Some(&solver))?;
    Ok(())
}

#[test]
#[allow(non_snake_case)]
fn test_dft_newton() -> Result<(), Box<dyn Error>> {
    let params = Arc::new(GcPcSaftFunctionalParameters::from_json_segments(
        &["propane"],
        "parameters/pcsaft/gc_substances.json",
        "parameters/pcsaft/sauer2014_hetero.json",
        None,
        IdentifierOption::Name,
    )?);
    let func = Arc::new(GcPcSaftFunctional::new(params));
    let t = 200.0 * KELVIN;
    let w = 150.0 * ANGSTROM;
    let points = 512;
    let tc = State::critical_point(&func, None, None, Default::default())?.temperature;
    let vle = PhaseEquilibrium::pure(&func, t, None, Default::default())?;
    let solver = DFTSolver::new(Some(Verbosity::Iter))
        .picard_iteration(None, Some(10), None, None)
        .newton(None, None, None, None);
    PlanarInterface::from_tanh(&vle, points, w, tc, false).solve(Some(&solver))?;
    Ok(())
}
