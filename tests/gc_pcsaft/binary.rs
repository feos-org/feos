use approx::assert_relative_eq;
use feos::gc_pcsaft::{GcPcSaft, GcPcSaftEosParameters};
#[cfg(feature = "dft")]
use feos::gc_pcsaft::{GcPcSaftFunctional, GcPcSaftFunctionalParameters};
use feos_core::parameter::{IdentifierOption, ParameterHetero};
use feos_core::{EosResult, State};
use ndarray::arr1;
use quantity::si::{KELVIN, MOL};
use std::rc::Rc;

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
    let eos = Rc::new(GcPcSaft::new(Rc::new(parameters)));
    #[cfg(feature = "dft")]
    let func = Rc::new(GcPcSaftFunctional::new(Rc::new(parameters_func)));
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
