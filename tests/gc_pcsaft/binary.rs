use approx::assert_relative_eq;
use feos::gc_pcsaft::{GcPcSaft, GcPcSaftEosParameters};
#[cfg(feature = "dft")]
use feos::gc_pcsaft::{GcPcSaftFunctional, GcPcSaftFunctionalParameters};
use feos_core::parameter::{IdentifierOption, ParameterHetero};
use feos_core::si::{KELVIN, METER, MOL};
use feos_core::{Contributions, EosResult, State};
use ndarray::arr1;
use std::sync::Arc;
use typenum::P3;

#[test]
fn test_binary() -> EosResult<()> {
    let parameters = GcPcSaftEosParameters::from_json_segments(
        &["ethanol", "methanol"],
        "parameters/pcsaft/gc_substances.json",
        "parameters/pcsaft/sauer2014_hetero.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    #[cfg(feature = "dft")]
    let parameters_func = GcPcSaftFunctionalParameters::from_json_segments(
        &["ethanol", "methanol"],
        "parameters/pcsaft/gc_substances.json",
        "parameters/pcsaft/sauer2014_hetero.json",
        None,
        IdentifierOption::Name,
    )
    .unwrap();
    let eos = Arc::new(GcPcSaft::new(Arc::new(parameters)));
    #[cfg(feature = "dft")]
    let func = Arc::new(GcPcSaftFunctional::new(Arc::new(parameters_func)));
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
fn test_polar_term() -> EosResult<()> {
    let parameters1 = GcPcSaftEosParameters::from_json_segments(
        &["CCCOC(C)=O", "CCCO"],
        "parameters/pcsaft/gc_substances.json",
        "parameters/pcsaft/sauer2014_hetero.json",
        None,
        IdentifierOption::Smiles,
    )?;
    let parameters2 = GcPcSaftEosParameters::from_json_segments(
        &["CCCO", "CCCOC(C)=O"],
        "parameters/pcsaft/gc_substances.json",
        "parameters/pcsaft/sauer2014_hetero.json",
        None,
        IdentifierOption::Smiles,
    )?;
    let eos1 = Arc::new(GcPcSaft::new(Arc::new(parameters1)));
    let eos2 = Arc::new(GcPcSaft::new(Arc::new(parameters2)));
    let moles = arr1(&[0.5, 0.5]) * MOL;
    let p1 = State::new_nvt(&eos1, 300.0 * KELVIN, METER.powi::<P3>(), &moles)?
        .pressure(Contributions::Total);
    let p2 = State::new_nvt(&eos2, 300.0 * KELVIN, METER.powi::<P3>(), &moles)?
        .pressure(Contributions::Total);
    println!("{p1} {p2}");
    assert_eq!(p1, p2);
    Ok(())
}
