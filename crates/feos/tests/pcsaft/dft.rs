#![allow(clippy::excessive_precision)]
#![cfg(feature = "dft")]
use approx::assert_relative_eq;
use feos::association::AssociationVersion;
use feos::hard_sphere::FMTVersion;
use feos::ideal_gas::{Joback, JobackParameters};
use feos::pcsaft::{PcSaft, PcSaftFunctional, PcSaftOptions, PcSaftParameters};
use feos_core::parameter::IdentifierOption;
use feos_core::{
    Contributions, EquationOfState, FeosResult, PhaseEquilibrium, ReferenceSystem, State,
};
use feos_dft::adsorption::{ExternalPotential, Pore1D, PoreSpecification};
use feos_dft::interface::Interface;
use feos_dft::{AssociationFunctional, DFTSolver, Geometry, HelmholtzEnergyFunctional};
use nalgebra::dvector;
use ndarray::Axis;
use quantity::*;
use std::error::Error;

fn parameters(comp: &str) -> FeosResult<PcSaftParameters> {
    PcSaftParameters::from_json(
        vec![comp],
        "tests/pcsaft/test_parameters.json",
        None,
        IdentifierOption::Name,
    )
}

#[test]
#[allow(non_snake_case)]
fn test_bulk_implementations() -> Result<(), Box<dyn Error>> {
    // correct for different k_B in old code
    let KB_old = 1.38064852e-23 * JOULE / KELVIN;
    let NAV_old = 6.022140857e23 / MOL;

    let eos = PcSaft::new(parameters("water_np")?);
    let func_pure = PcSaftFunctional::new(parameters("water_np")?);
    let func_full = PcSaftFunctional::with_options(
        parameters("water_np")?,
        FMTVersion::KierlikRosinberg,
        AssociationVersion::ISaft,
        PcSaftOptions::default(),
    );
    let t = 300.0 * KELVIN;
    let v = 0.002 * METER.powi::<3>() * NAV / NAV_old;
    let n = dvector![1.5] * MOL;
    let state = State::new_nvt(&&eos, t, v, &n)?;
    let state_pure = State::new_nvt(&&func_pure, t, v, &n)?;
    let state_full = State::new_nvt(&&func_full, t, v, &n)?;
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
    println!("{:21}: {}", p_full[3].0, p_full[3].1 + p_full[4].1);
    println!();
    println!("{:21}: {}", p[4].0, p[4].1);
    println!("{:21}: {}", p_pure[5].0, p_pure[5].1);
    println!("{:21}: {}", p_full[5].0, p_full[5].1);

    let ideal_gas = 1.8707534688259309 * MEGA * PASCAL * KB / KB_old;
    assert_relative_eq!(p[0].1, ideal_gas, max_relative = 1e-14,);
    assert_relative_eq!(p_pure[0].1, ideal_gas, max_relative = 1e-14,);
    assert_relative_eq!(p_full[0].1, ideal_gas, max_relative = 1e-14,);

    let hard_sphere = 54.7102253882827583 * KILO * PASCAL * KB / KB_old;
    assert_relative_eq!(p[1].1, hard_sphere, max_relative = 1e-14,);
    assert_relative_eq!(p_pure[1].1, hard_sphere, max_relative = 1e-14,);
    assert_relative_eq!(p_full[1].1, hard_sphere, max_relative = 1e-14,);

    let hard_chains = -2.0847750028499230 * KILO * PASCAL * KB / KB_old;
    assert_relative_eq!(p[2].1, hard_chains, max_relative = 1e-14,);
    assert_relative_eq!(p_pure[2].1 + p_pure[4].1, hard_chains, max_relative = 2e-13,);
    assert_relative_eq!(p_full[2].1 + p_full[4].1, hard_chains, max_relative = 2e-13,);

    let dispersion = -262.895932352779993 * KILO * PASCAL * KB / KB_old;
    assert_relative_eq!(p[3].1, dispersion, max_relative = 1e-14,);
    assert_relative_eq!(p_pure[3].1, dispersion, max_relative = 1e-14,);
    assert_relative_eq!(p_full[3].1, dispersion, max_relative = 1e-14,);

    let association = -918.3899928262694630 * KILO * PASCAL * KB / KB_old;
    assert_relative_eq!(p[4].1, association, max_relative = 1e-14,);
    assert_relative_eq!(p_pure[5].1, association, max_relative = 1e-14,);
    assert_relative_eq!(p_full[5].1, association, max_relative = 1e-14,);
    Ok(())
}

#[test]
#[allow(non_snake_case)]
fn test_dft_propane() -> Result<(), Box<dyn Error>> {
    // correct for different k_B in old code
    let KB_old = 1.38064852e-23 * JOULE / KELVIN;
    let NAV_old = 6.022140857e23 / MOL;

    let func_pure = PcSaftFunctional::new(parameters("propane")?);
    let func_full = PcSaftFunctional::with_options(
        parameters("propane")?,
        FMTVersion::KierlikRosinberg,
        AssociationVersion::ISaft,
        PcSaftOptions::default(),
    );
    let t = 200.0 * KELVIN;
    let w = 150.0 * ANGSTROM;
    let points = 2048;
    let tc = State::critical_point(&&func_pure, (), None, None, Default::default())?.temperature;
    let vle_pure = PhaseEquilibrium::pure(&&func_pure, t, None, Default::default())?;
    let vle_full = PhaseEquilibrium::pure(&&func_full, t, None, Default::default())?;
    let profile_pure = Interface::planar_from_tanh(&vle_pure, points, w, tc).solve(None)?;
    let profile_full = Interface::planar_from_tanh(&vle_full, points, w, tc).solve(None)?;
    println!(
        "pure {} {} {}",
        profile_pure.surface_tension.unwrap(),
        vle_pure.vapor().density,
        vle_pure.liquid().density
    );
    println!(
        "full {} {} {}",
        profile_full.surface_tension.unwrap(),
        vle_full.vapor().density,
        vle_full.liquid().density
    );

    let vapor_density = 12.2557486248527745 * MOL / METER.powi::<3>() * NAV_old / NAV;
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

    let liquid_density = 13.8941749145544549 * KILO * MOL / METER.powi::<3>() * NAV_old / NAV;
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
    Ok(())
}

#[test]
#[allow(non_snake_case)]
fn test_dft_water() -> Result<(), Box<dyn Error>> {
    // correct for different k_B in old code
    let KB_old = 1.38064852e-23 * JOULE / KELVIN;
    let NAV_old = 6.022140857e23 / MOL;

    // reference values for the association functional of Yu and Wu
    let func_pure = PcSaftFunctional::with_options(
        parameters("water_np")?,
        FMTVersion::WhiteBear,
        AssociationVersion::YuWu,
        PcSaftOptions::default(),
    );
    let t = 400.0 * KELVIN;
    let w = 120.0 * ANGSTROM;
    let points = 2048;
    let tc = State::critical_point(&&func_pure, (), None, None, Default::default())?.temperature;
    let vle_pure = PhaseEquilibrium::pure(&&func_pure, t, None, Default::default())?;
    let profile_pure = Interface::planar_from_tanh(&vle_pure, points, w, tc).solve(None)?;
    println!(
        "pure {} {} {}",
        profile_pure.surface_tension.unwrap(),
        vle_pure.vapor().density,
        vle_pure.liquid().density
    );

    let vapor_density = 75.8045715345905222 * MOL / METER.powi::<3>() * NAV_old / NAV;
    assert_relative_eq!(
        vle_pure.vapor().density,
        vapor_density,
        max_relative = 1e-13,
    );

    let liquid_density = 47.8480850281608454 * KILO * MOL / METER.powi::<3>() * NAV_old / NAV;
    assert_relative_eq!(
        vle_pure.liquid().density,
        liquid_density,
        max_relative = 1e-13,
    );

    let surface_tension = 70.1419567481980408 * MILLI * NEWTON / METER * KB / KB_old;
    assert_relative_eq!(
        profile_pure.surface_tension.unwrap(),
        surface_tension,
        max_relative = 1e-4,
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
    let joback = Joback::new(JobackParameters::from_json(
        vec!["water_np"],
        "tests/pcsaft/test_parameters_joback.json",
        None,
        IdentifierOption::Name,
    )?);
    let func = EquationOfState::new(joback, PcSaftFunctional::new(params));
    let vle = PhaseEquilibrium::pure(&&func, 350.0 * KELVIN, None, Default::default())?;
    let tc = State::critical_point(&&func, (), None, None, Default::default())?.temperature;
    let profile = Interface::planar_from_tanh(&vle, 2048, 100.0 * ANGSTROM, tc).solve(None)?;
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
    println!("\nResidual:\n{s_res:?}");
    println!(
        "liquid: {:?}, vapor: {:?}",
        profile.vle.liquid().molar_entropy(Contributions::Residual)
            / profile.vle.liquid().molar_volume,
        profile.vle.vapor().molar_entropy(Contributions::Residual)
            / profile.vle.vapor().molar_volume
    );
    println!("\nTotal:\n{s_tot:?}");
    println!(
        "liquid: {:?}, vapor: {:?}",
        profile.vle.liquid().molar_entropy(Contributions::Total)
            / profile.vle.liquid().molar_volume,
        profile.vle.vapor().molar_entropy(Contributions::Total) / profile.vle.vapor().molar_volume
    );
    assert_relative_eq!(
        s_res.get(0),
        profile.vle.liquid().molar_entropy(Contributions::Residual)
            / profile.vle.liquid().molar_volume,
        max_relative = 1e-8,
    );
    assert_relative_eq!(
        s_res.get(2047),
        profile.vle.vapor().molar_entropy(Contributions::Residual)
            / profile.vle.vapor().molar_volume,
        max_relative = 1e-6,
    );
    assert_relative_eq!(
        s_tot.get(0),
        profile.vle.liquid().molar_entropy(Contributions::Total)
            / profile.vle.liquid().molar_volume,
        max_relative = 1e-8,
    );
    assert_relative_eq!(
        s_tot.get(2047),
        profile.vle.vapor().molar_entropy(Contributions::Total) / profile.vle.vapor().molar_volume,
        max_relative = 1e-6,
    );
    Ok(())
}

#[test]
fn test_initial_non_bonded_fractions() -> Result<(), Box<dyn Error>> {
    // the initial fractions of non-bonded sites of a profile solve the mass action equations, for
    // local (Yu-Wu, WDA) and nonlocal (iSAFT) association kernels
    for version in [
        AssociationVersion::YuWu,
        AssociationVersion::Wda,
        AssociationVersion::ISaft,
    ] {
        let func = &PcSaftFunctional::with_options(
            parameters("water_np")?,
            FMTVersion::WhiteBear,
            version,
            PcSaftOptions::default(),
        );
        let vle = PhaseEquilibrium::pure(&func, 350.0 * KELVIN, None, Default::default())?;
        let pore = Pore1D::new(
            &func,
            Geometry::Cartesian,
            20.0 * ANGSTROM,
            ExternalPotential::LJ93 {
                sigma_ss: 3.0,
                epsilon_k_ss: 100.0,
                rho_s: 0.08,
            },
            Some(512),
        );
        let profile = pore
            .initialize(vle.liquid(), None, PoreSpecification::ChemicalPotential)?
            .profile;
        let association = profile.functional.association().unwrap();
        let evaluation = association.evaluate(
            profile.temperature.to_reduced(),
            &profile.density.to_reduced(),
            &profile.non_bonded_fractions,
            profile.association_convolver.as_deref().unwrap(),
            profile.association_kernel_convolver.as_deref(),
        );
        assert!(
            evaluation
                .mass_action_residual
                .iter()
                .all(|r| r.abs() < 1e-10)
        );
    }
    Ok(())
}

#[test]
fn test_association_functional() -> Result<(), Box<dyn Error>> {
    // Reference values (reduced units) for the Yu-Wu version, calculated with the former
    // implementation of the Yu-Wu functional as functional contribution (the new implementation
    // agreed to 1e-12): surface tension, and N, dN/dmu, dN/dT, dN/dp in the pore.
    let gamma_ref = 5.89546958776040455e1;
    let pore_ref = [
        2.14194575140416427e-1,
        1.41041271950989350e-4,
        -2.54686328982616384e-4,
        4.72932743004560955e-3,
    ];

    // tight tolerance, so that the comparison is not limited by the convergence of the profiles
    let solver = DFTSolver::new(None).newton(None, None, Some(1e-13));
    let t = 350.0 * KELVIN;
    for version in [
        AssociationVersion::YuWu,
        AssociationVersion::Wda,
        AssociationVersion::ISaft,
    ] {
        let func = &PcSaftFunctional::with_options(
            parameters("water_np")?,
            FMTVersion::WhiteBear,
            version,
            PcSaftOptions::default(),
        );

        // planar interface
        let vle = PhaseEquilibrium::pure(&func, t, None, Default::default())?;
        let interface = Interface::planar_from_tanh(&vle, 1024, 100.0 * ANGSTROM, 600.0 * KELVIN)
            .solve(Some(&solver))?;
        let gamma = interface.surface_tension.unwrap().to_reduced();

        // slit pore
        let bulk = State::new_npt(&func, t, 5.0 * BAR, (), None)?;
        let mut pore = Pore1D::new(
            &func,
            Geometry::Cartesian,
            20.0 * ANGSTROM,
            ExternalPotential::LJ93 {
                epsilon_k_ss: 10.0,
                sigma_ss: 3.0,
                rho_s: 0.08,
            },
            None,
        )
        .initialize(&bulk, None, PoreSpecification::ChemicalPotential)?
        .solve(Some(&solver))?;
        let profile = &mut pore.profile;
        let n = [
            profile.moles().to_reduced()[0],
            profile.dn_dmu()?.to_reduced()[(0, 0)],
            profile.dn_dt(&bulk)?.to_reduced()[0],
            profile.dn_dp(&bulk)?.to_reduced()[0],
        ];
        println!("{version:?}: surface tension {gamma}, pore {n:?}");

        if version == AssociationVersion::YuWu {
            assert_relative_eq!(gamma, gamma_ref, max_relative = 1e-10);
            for (n, n_ref) in n.into_iter().zip(pore_ref) {
                assert_relative_eq!(n, n_ref, max_relative = 1e-10);
            }
        }
    }
    Ok(())
}
