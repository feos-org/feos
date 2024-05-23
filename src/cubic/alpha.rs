use enum_dispatch::enum_dispatch;
use ndarray::Array1;
use num_dual::DualNum;

#[enum_dispatch]
trait AlphaFunction {
    fn alpha<D: DualNum<f64>>(
        &self,
        acentric_factor: &Array1<f64>,
        reduced_temperature: &Array1<D>,
    ) -> Array1<D>;
}

struct Soave;

impl AlphaFunction for Soave {
    fn alpha<D: DualNum<f64>>(
        &self,
        acentric_factor: &Array1<f64>,
        reduced_temperature: &Array1<D>,
    ) -> Array1<D> {
        let m = 0.48 + acentric_factor * (1.574 - acentric_factor * 0.176);
        ((-reduced_temperature.mapv(|t| t.sqrt()) + 1.0) * m + 1.0).mapv(|a| a.powi(2))
    }
}

/// Improved parameterization of the Soave alpha function for Redlich-Kwong equation of state.
///
/// https://doi.org/10.1016/j.fluid.2018.12.007
struct SoaveRedlichKwong2019;

impl AlphaFunction for SoaveRedlichKwong2019 {
    fn alpha<D: DualNum<f64>>(
        &self,
        acentric_factor: &Array1<f64>,
        reduced_temperature: &Array1<D>,
    ) -> Array1<D> {
        let m = 0.481
            + acentric_factor * (1.5963 - acentric_factor * (0.2963 - acentric_factor * 0.1223));
        ((-reduced_temperature.mapv(|t| t.sqrt()) + 1.0) * m + 1.0).mapv(|a| a.powi(2))
    }
}

/// Improved parameterization of the Soave alpha function for Peng-Robinson equation of state.
///
/// https://doi.org/10.1016/j.fluid.2018.12.007
struct SoavePengRobinson2019;

impl AlphaFunction for SoavePengRobinson2019 {
    fn alpha<D: DualNum<f64>>(
        &self,
        acentric_factor: &Array1<f64>,
        reduced_temperature: &Array1<D>,
    ) -> Array1<D> {
        let m = 0.481
            + acentric_factor * (1.5963 - acentric_factor * (0.2963 - acentric_factor * 0.1223));
        ((-reduced_temperature.mapv(|t| t.sqrt()) + 1.0) * m + 1.0).mapv(|a| a.powi(2))
    }
}

#[enum_dispatch(AlphaFunction)]
enum Alpha {
    Soave,
    SoaveRedlichKwong2019,
    SoavePengRobinson2019,
}
