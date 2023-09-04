use super::{DataSet, EstimatorError, Phase};
use feos_core::si::{
    MolarEnergy, Moles, Pressure, Quantity, Temperature, _Dimensionless, PASCAL, RGAS,
};
use feos_core::{
    Contributions, DensityInitialization, PhaseDiagram, PhaseEquilibrium, Residual, State, TPSpec,
    TemperatureOrPressure,
};
use itertools::izip;
use ndarray::{arr1, s, Array1, ArrayView1, Axis};
use std::iter::FromIterator;
use std::ops::Sub;
use std::sync::Arc;

/// Store experimental binary VLE data for the calculation of chemical potential residuals.
#[derive(Clone)]
pub struct BinaryVleChemicalPotential {
    temperature: Temperature<Array1<f64>>,
    pressure: Pressure<Array1<f64>>,
    liquid_molefracs: Array1<f64>,
    vapor_molefracs: Array1<f64>,
    target: Array1<f64>,
}

impl BinaryVleChemicalPotential {
    pub fn new(
        temperature: Temperature<Array1<f64>>,
        pressure: Pressure<Array1<f64>>,
        liquid_molefracs: Array1<f64>,
        vapor_molefracs: Array1<f64>,
    ) -> Self {
        let target = Array1::ones(temperature.len() * 2);
        Self {
            temperature,
            pressure,
            liquid_molefracs,
            vapor_molefracs,
            target,
        }
    }
}

impl<E: Residual> DataSet<E> for BinaryVleChemicalPotential {
    fn target(&self) -> &Array1<f64> {
        &self.target
    }

    fn target_str(&self) -> &str {
        "chemical potential"
    }

    fn input_str(&self) -> Vec<&str> {
        vec![
            "temperature",
            "pressure",
            "liquid molefracs",
            "vapor molefracs",
        ]
    }

    fn predict(&self, eos: &Arc<E>) -> Result<Array1<f64>, EstimatorError> {
        let mut prediction = Vec::new();
        for (&xi, &yi, t, p) in izip!(
            &self.liquid_molefracs,
            &self.vapor_molefracs,
            &self.temperature,
            &self.pressure
        ) {
            let liquid_moles = Moles::from_reduced(arr1(&[xi, 1.0 - xi]));
            let liquid = State::new_npt(eos, t, p, &liquid_moles, DensityInitialization::Liquid)?;
            let mu_res_liquid = liquid.residual_chemical_potential();
            let vapor_moles = Moles::from_reduced(arr1(&[yi, 1.0 - yi]));
            let vapor = State::new_npt(eos, t, p, &vapor_moles, DensityInitialization::Vapor)?;
            let mu_res_vapor = vapor.residual_chemical_potential();

            let kt = RGAS * t;
            let rho_frac = (&liquid.partial_density / &vapor.partial_density).into_value();
            prediction.push(mu_res_liquid.get(0) - mu_res_vapor.get(0) + kt * rho_frac[0].ln());
            prediction.push(mu_res_liquid.get(1) - mu_res_vapor.get(1) + kt * rho_frac[1].ln());
        }
        let prediction = (MolarEnergy::from_vec(prediction) / MolarEnergy::from_reduced(500.0))
            .into_value()
            + 1.0;
        Ok(prediction)
    }

    // fn get_input(&self) -> HashMap<String, SIArray1> {
    //     let mut m = HashMap::with_capacity(4);
    //     m.insert("temperature".to_owned(), self.temperature.clone());
    //     m.insert("pressure".to_owned(), self.pressure.clone());
    //     m.insert(
    //         "liquid_molefracs".to_owned(),
    //         &self.liquid_molefracs * SIUnit::reference_moles() / SIUnit::reference_moles(),
    //     );
    //     m.insert(
    //         "vapor_molefracs".to_owned(),
    //         &self.vapor_molefracs * SIUnit::reference_moles() / SIUnit::reference_moles(),
    //     );
    //     m
    // }
}

/// Store experimental binary VLE data for the calculation of pressure residuals.
#[derive(Clone)]
pub struct BinaryVlePressure {
    target: Array1<f64>,
    temperature: Temperature<Array1<f64>>,
    pressure: Pressure<Array1<f64>>,
    unit: Pressure,
    molefracs: Array1<f64>,
    phase: Phase,
}

impl BinaryVlePressure {
    pub fn new(
        temperature: Temperature<Array1<f64>>,
        pressure: Pressure<Array1<f64>>,
        molefracs: Array1<f64>,
        phase: Phase,
    ) -> Self {
        let unit = PASCAL;
        Self {
            target: (&pressure / unit).into_value(),
            temperature,
            pressure,
            unit,
            molefracs,
            phase,
        }
    }
}

impl<E: Residual> DataSet<E> for BinaryVlePressure {
    fn target(&self) -> &Array1<f64> {
        &self.target
    }

    fn target_str(&self) -> &str {
        "pressure"
    }

    fn input_str(&self) -> Vec<&str> {
        let mut vec = vec!["temperature", "pressure"];
        vec.push(match self.phase {
            Phase::Vapor => "vapor molefracs",
            Phase::Liquid => "liquid molefracs",
        });
        vec
    }

    fn predict(&self, eos: &Arc<E>) -> Result<Array1<f64>, EstimatorError> {
        let options = Default::default();
        self.molefracs
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                let vle = (match self.phase {
                    Phase::Vapor => PhaseEquilibrium::dew_point(
                        eos,
                        self.temperature.get(i),
                        &arr1(&[xi, 1.0 - xi]),
                        Some(self.pressure.get(i)),
                        None,
                        options,
                    ),
                    Phase::Liquid => PhaseEquilibrium::bubble_point(
                        eos,
                        self.temperature.get(i),
                        &arr1(&[xi, 1.0 - xi]),
                        Some(self.pressure.get(i)),
                        None,
                        options,
                    ),
                })?;

                Ok((vle.vapor().pressure(Contributions::Total) / self.unit).into_value())
            })
            .collect()
    }

    // fn get_input(&self) -> HashMap<String, SIArray1> {
    //     let mut m = HashMap::with_capacity(4);
    //     m.insert("temperature".to_owned(), self.temperature.clone());
    //     m.insert("pressure".to_owned(), self.pressure.clone());
    //     m.insert(
    //         (match self.phase {
    //             Phase::Vapor => "vapor_molefracs",
    //             Phase::Liquid => "liquid_molefracs",
    //         })
    //         .to_owned(),
    //         &self.molefracs * SIUnit::reference_moles() / SIUnit::reference_moles(),
    //     );
    //     m
    // }
}

/// Store experimental binary phase diagrams for the calculation of distance residuals.
#[derive(Clone)]
pub struct BinaryPhaseDiagram<TP: TemperatureOrPressure, U> {
    specification: TP,
    temperature_or_pressure: Quantity<Array1<f64>, U>,
    liquid_molefracs: Option<Array1<f64>>,
    vapor_molefracs: Option<Array1<f64>>,
    npoints: Option<usize>,
    target: Array1<f64>,
}

impl<TP: TemperatureOrPressure, U> BinaryPhaseDiagram<TP, U> {
    pub fn new(
        specification: TP,
        temperature_or_pressure: Quantity<Array1<f64>, U>,
        liquid_molefracs: Option<Array1<f64>>,
        vapor_molefracs: Option<Array1<f64>>,
        npoints: Option<usize>,
    ) -> Self {
        let count = liquid_molefracs.as_ref().map_or(0, |x| 2 * x.len())
            + vapor_molefracs.as_ref().map_or(0, |x| 2 * x.len());
        let target = Array1::ones(count);
        Self {
            specification,
            temperature_or_pressure,
            liquid_molefracs,
            vapor_molefracs,
            npoints,
            target,
        }
    }
}

impl<TP: TemperatureOrPressure + Sync + Send, U: Copy + Sync + Send, E: Residual> DataSet<E>
    for BinaryPhaseDiagram<TP, U>
where
    TPSpec: From<TP>,
    Quantity<Array1<f64>, U>: FromIterator<TP::Other>,
    U: Sub<U, Output = _Dimensionless>,
{
    fn target(&self) -> &Array1<f64> {
        &self.target
    }

    fn target_str(&self) -> &str {
        "distance"
    }

    fn input_str(&self) -> Vec<&str> {
        let mut vec = match TPSpec::from(self.specification) {
            TPSpec::Temperature(_) => vec!["temperature", "pressure"],
            TPSpec::Pressure(_) => vec!["pressure", "temperature"],
        };
        if self.liquid_molefracs.is_some() {
            vec.push("liquid molefracs")
        }
        if self.vapor_molefracs.is_some() {
            vec.push("vapor molefracs")
        }
        vec
    }

    fn predict(&self, eos: &Arc<E>) -> Result<Array1<f64>, EstimatorError> {
        let mut res = Vec::new();

        let dia = PhaseDiagram::binary_vle(
            eos,
            self.specification,
            self.npoints,
            None,
            Default::default(),
        )?;
        let x_liq = dia.liquid().molefracs();
        let x_vec_liq = x_liq.index_axis(Axis(1), 0);
        let x_vap = dia.vapor().molefracs();
        let x_vec_vap = x_vap.index_axis(Axis(1), 0);
        let tp_vec = dia.vapor().iter().map(|s| TP::from_state(s)).collect();
        // let tp_vec = if self
        //     .temperature_or_pressure
        //     .has_unit(&SIUnit::reference_temperature())
        // {
        //     dia.vapor().temperature()
        // } else {
        //     dia.vapor().pressure()
        // };
        for (x_exp, x_vec) in [
            (&self.liquid_molefracs, x_vec_liq),
            (&self.vapor_molefracs, x_vec_vap),
        ] {
            if let Some(x_exp) = x_exp {
                res.extend(predict_distance(
                    x_vec,
                    &tp_vec,
                    x_exp,
                    &self.temperature_or_pressure,
                ));
            }
        }
        Ok(Array1::from_vec(res))
    }

    // fn get_input(&self) -> HashMap<String, SIArray1> {
    //     let mut m = HashMap::with_capacity(4);
    //     if self
    //         .specification
    //         .has_unit(&SIUnit::reference_temperature())
    //     {
    //         m.insert(
    //             "temperature".to_owned(),
    //             SIArray1::from_vec(vec![self.specification]),
    //         );
    //         m.insert("pressure".to_owned(), self.temperature_or_pressure.clone());
    //     } else {
    //         m.insert(
    //             "pressure".to_owned(),
    //             SIArray1::from_vec(vec![self.specification]),
    //         );
    //         m.insert(
    //             "temperature".to_owned(),
    //             self.temperature_or_pressure.clone(),
    //         );
    //     };
    //     if let Some(liquid_molefracs) = &self.liquid_molefracs {
    //         m.insert(
    //             "liquid_molefracs".to_owned(),
    //             liquid_molefracs * SIUnit::reference_moles() / SIUnit::reference_moles(),
    //         );
    //     }
    //     if let Some(vapor_molefracs) = &self.vapor_molefracs {
    //         m.insert(
    //             "vapor_molefracs".to_owned(),
    //             vapor_molefracs * SIUnit::reference_moles() / SIUnit::reference_moles(),
    //         );
    //     }
    //     m
    // }
}

fn predict_distance<U: Copy + Sub<U, Output = _Dimensionless>>(
    x_vec: ArrayView1<f64>,
    tp_vec: &Quantity<Array1<f64>, U>,
    x_exp: &Array1<f64>,
    tp_exp: &Quantity<Array1<f64>, U>,
) -> Vec<f64> {
    let mut res = Vec::new();
    for (tp, &x) in tp_exp.into_iter().zip(x_exp.into_iter()) {
        let y = 1.0;
        let y_vec = (tp_vec / tp).into_value();
        let dx = &x_vec.slice(s![1..]) - &x_vec.slice(s![..-1]);
        let dy = &y_vec.slice(s![1..]) - &y_vec.slice(s![..-1]);
        let x_vec = x_vec.slice(s![..-1]);
        let y_vec = y_vec.slice(s![..-1]);

        let t = ((x - &x_vec) * &dx + (y - &y_vec) * &dy) / (&dx * &dx + &dy * &dy);
        let x0 = &t * dx + x_vec;
        let y0 = &t * dy + y_vec;

        let k = x0
            .iter()
            .zip(y0.iter())
            .enumerate()
            .filter(|(i, _)| 0.0 < t[*i] && t[*i] < 1.0)
            .map(|(i, (xx, yy))| (i, (xx - x) * (xx - x) + (yy - y) * (yy - y)))
            .reduce(|(k1, d1), (k2, d2)| if d1 < d2 { (k1, d1) } else { (k2, d2) });

        let (x0, y0) = match k {
            None => {
                let point = t
                    .iter()
                    .zip(t.iter().skip(1))
                    .enumerate()
                    .find(|&(_, (t1, t2))| *t1 > 1.0 && *t2 < 0.0);
                if let Some((point, _)) = point {
                    (x_vec[point + 1], y_vec[point + 1])
                } else {
                    let n = t.len();
                    if t[0].abs() < t[n - 1].abs() {
                        (x_vec[0], y_vec[0])
                    } else {
                        (x_vec[n - 1], y_vec[n - 1])
                    }
                }
            }
            Some((k, _)) => (x0[k], y0[k]),
        };
        res.push(x0 - x + 1.0);
        res.push(y0);
    }
    res
}
