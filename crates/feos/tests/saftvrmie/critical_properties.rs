use feos::saftvrmie::{test_utils, SaftVRMie};
use feos_core::{SolverOptions, State};
use quantity::*;
use std::collections::HashMap;
use std::sync::Arc;
use typenum::P3;

/// Critical data reported in Lafitte et al.
pub fn critical_data() -> HashMap<&'static str, (Temperature, Pressure, MassDensity)> {
    let mut data = HashMap::new();
    let kg_m3 = KILOGRAM / METER.powi::<P3>();
    let mpa = MEGA * PASCAL;
    let k = KELVIN;

    _ = data.insert("methane", (195.30 * k, 5.15 * mpa, 154.45 * kg_m3));
    _ = data.insert("ethane", (311.38 * k, 5.49 * mpa, 205.84 * kg_m3));
    _ = data.insert("propane", (376.20 * k, 4.77 * mpa, 219.98 * kg_m3));
    _ = data.insert("n-butane", (432.68 * k, 4.27 * mpa, 228.04 * kg_m3));
    _ = data.insert("pentane", (476.44 * k, 3.81 * mpa, 238.14 * kg_m3));
    _ = data.insert("hexane", (515.29 * k, 3.44 * mpa, 241.16 * kg_m3));
    _ = data.insert("heptane", (547.33 * k, 3.06 * mpa, 233.76 * kg_m3));
    _ = data.insert("octane", (576.72 * k, 2.77 * mpa, 227.83 * kg_m3));
    _ = data.insert("nonane", (602.20 * k, 2.52 * mpa, 224.89 * kg_m3));
    _ = data.insert("decane", (626.37 * k, 2.31 * mpa, 219.85 * kg_m3));
    _ = data.insert("dodecane", (668.75 * k, 1.99 * mpa, 214.26 * kg_m3));
    _ = data.insert("pentadecane", (720.98 * k, 1.61 * mpa, 194.23 * kg_m3));
    _ = data.insert("eicosane", (786.33 * k, 1.21 * mpa, 174.91 * kg_m3));
    _ = data.insert("methanol", (547.73 * k, 12.21 * mpa, 260.59 * kg_m3));
    _ = data.insert("ethanol", (554.41 * k, 8.83 * mpa, 247.96 * kg_m3));
    _ = data.insert("1-propanol", (560.43 * k, 6.89 * mpa, 256.34 * kg_m3));
    _ = data.insert("1-butanol", (583.97 * k, 5.50 * mpa, 253.86 * kg_m3));
    _ = data.insert(
        "tetrafluoromethane",
        (232.77 * k, 4.14 * mpa, 644.29 * kg_m3),
    );
    _ = data.insert("hexafluoroethane", (295.46 * k, 3.24 * mpa, 634.96 * kg_m3));
    _ = data.insert("perfluoropropane", (347.88 * k, 2.79 * mpa, 648.71 * kg_m3));
    _ = data.insert("perfluorobutane", (386.86 * k, 2.38 * mpa, 635.61 * kg_m3));
    _ = data.insert("perfluoropentane", (421.36 * k, 2.13 * mpa, 634.72 * kg_m3));
    _ = data.insert("fluorine", (146.20 * k, 5.66 * mpa, 559.47 * kg_m3));
    _ = data.insert("carbon dioxide", (307.00 * k, 7.86 * mpa, 472.15 * kg_m3));
    _ = data.insert("benzene", (568.33 * k, 5.51 * mpa, 307.69 * kg_m3));
    _ = data.insert("toluene", (600.25 * k, 4.73 * mpa, 301.21 * kg_m3));
    data
}

#[test]
fn critical_properties_pure() {
    let t0 = Some(500.0 * KELVIN);
    critical_data().iter().for_each(|(name, data)| {
        dbg!(name);
        let mut parameters = test_utils::test_parameters();
        let option = SolverOptions::default();
        let p = parameters.remove(name).unwrap();
        let eos = Arc::new(SaftVRMie::new(Arc::new(p)));
        let cp = State::critical_point(&eos, None, t0, option).unwrap();
        dbg!(((data.0 - cp.temperature) / data.0).into_value() * 100.0);
        // temperature within 0.2%
        assert!(((data.0 - cp.temperature).abs() / data.0).into_value() * 100.0 < 0.2);
        // pressure within 0.5%
        assert!(
            ((data.1 - cp.pressure(feos_core::Contributions::Total)).abs() / data.1).into_value()
                * 100.0
                < 0.5
        );
        // density within 1%
        assert!(((data.2 - cp.mass_density()).abs() / data.2).into_value() * 100.0 < 1.0);
    })
}
