use approx::assert_relative_eq;
#[cfg(feature = "dft")]
use feos::gc_pcsaft::GcPcSaftFunctional;
use feos::gc_pcsaft::{GcPcSaft, GcPcSaftParameters};
use feos_core::parameter::IdentifierOption;
use feos_core::{Contributions, FeosResult, State};
use ndarray::arr1;
use quantity::{KELVIN, METER, MOL};
use std::sync::Arc;
use typenum::P3;

#[test]
fn test_binary() -> FeosResult<()> {
    let parameters = GcPcSaftParameters::from_json_segments_hetero(
        &["ethanol", "methanol"],
        "../../parameters/pcsaft/gc_substances.json",
        "../../parameters/pcsaft/sauer2014_hetero.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    #[cfg(feature = "dft")]
    let parameters_func = GcPcSaftParameters::from_json_segments_hetero(
        &["ethanol", "methanol"],
        "../../parameters/pcsaft/gc_substances.json",
        "../../parameters/pcsaft/sauer2014_hetero.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    let eos = Arc::new(GcPcSaft::new(parameters));
    #[cfg(feature = "dft")]
    let func = Arc::new(GcPcSaftFunctional::new(parameters_func));
    let moles = arr1(&[0.5, 0.5]) * MOL;
    let cp = State::critical_point(&eos, Some(&moles), None, Default::default())?;
    #[cfg(feature = "dft")]
    let cp_func = State::critical_point(&func, Some(&moles), None, Default::default())?;
    println!("{}", cp.temperature);
    #[cfg(feature = "dft")]
    println!("{}", cp_func.temperature);
    assert_relative_eq!(
        cp.temperature,
        536.4129479522177 * KELVIN,
        max_relative = 1e-14
    );
    #[cfg(feature = "dft")]
    assert_relative_eq!(
        cp_func.temperature,
        536.4129479522177 * KELVIN,
        max_relative = 1e-14
    );
    Ok(())
}

#[test]
fn test_polar_term() -> FeosResult<()> {
    let parameters1 = GcPcSaftParameters::from_json_segments_hetero(
        &["CCCOC(C)=O", "CCCO"],
        "../../parameters/pcsaft/gc_substances.json",
        "../../parameters/pcsaft/sauer2014_hetero.json",
        None,
        IdentifierOption::Smiles,
    )?;
    let parameters2 = GcPcSaftParameters::from_json_segments_hetero(
        &["CCCO", "CCCOC(C)=O"],
        "../../parameters/pcsaft/gc_substances.json",
        "../../parameters/pcsaft/sauer2014_hetero.json",
        None,
        IdentifierOption::Smiles,
    )?;
    let eos1 = Arc::new(GcPcSaft::new(parameters1));
    let eos2 = Arc::new(GcPcSaft::new(parameters2));
    let moles = arr1(&[0.5, 0.5]) * MOL;
    let p1 = State::new_nvt(&eos1, 300.0 * KELVIN, METER.powi::<P3>(), &moles)?
        .pressure(Contributions::Total);
    let p2 = State::new_nvt(&eos2, 300.0 * KELVIN, METER.powi::<P3>(), &moles)?
        .pressure(Contributions::Total);
    println!("{p1} {p2}");
    assert_eq!(p1, p2);
    Ok(())
}
