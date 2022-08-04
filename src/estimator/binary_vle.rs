use super::{DataSet, EstimatorError};
use feos_core::{
    Contributions, DensityInitialization, EosUnit, EquationOfState, PhaseEquilibrium, State,
};
use ndarray::{arr1, Array1};
use quantity::{Quantity, QuantityArray1, QuantityScalar};
use std::collections::HashMap;
use std::fmt::LowerExp;
use std::rc::Rc;

#[derive(Clone, Copy)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
pub enum CostFunction {
    Pressure,
    ChemicalPotential,
    // Distance,
}

/// Store experimental vapor pressure data and compare to the equation of state.
#[derive(Clone)]
pub struct BinaryVle<U: EosUnit> {
    cost_function: CostFunction,
    temperature: QuantityArray1<U>,
    pressure: QuantityArray1<U>,
    liquid_molefracs: Option<Array1<f64>>,
    vapor_molefracs: Option<Array1<f64>>,
    target: QuantityArray1<U>,
}

impl<U: EosUnit> BinaryVle<U> {
    pub fn new(
        cost_function: CostFunction,
        temperature: QuantityArray1<U>,
        pressure: QuantityArray1<U>,
        liquid_molefracs: Option<Array1<f64>>,
        vapor_molefracs: Option<Array1<f64>>,
    ) -> Result<Self, EstimatorError> {
        let target = match cost_function {
            CostFunction::Pressure => {
                let mut target = pressure.clone();
                if liquid_molefracs.is_some() && vapor_molefracs.is_some() {
                    target = pressure.into_iter().chain(pressure.into_iter()).collect();
                }
                target
            }
            CostFunction::ChemicalPotential => {
                Array1::ones(temperature.len() * 2) * 500.0 * U::reference_molar_energy()
            } // CostFunction::Distance => todo!(),
        };
        Ok(Self {
            cost_function,
            temperature,
            pressure,
            liquid_molefracs,
            vapor_molefracs,
            target,
        })
    }

    fn predict_pressure<E: EquationOfState>(
        &self,
        eos: &Rc<E>,
    ) -> Result<QuantityArray1<U>, EstimatorError>
    where
        Quantity<f64, U>: std::fmt::Display,
    {
        let options = Default::default();

        let mut prediction = Vec::new();
        if let Some(liquid_molefracs) = &self.liquid_molefracs {
            for (i, &xi) in liquid_molefracs.iter().enumerate() {
                prediction.push(
                    PhaseEquilibrium::bubble_point(
                        eos,
                        self.temperature.get(i),
                        &arr1(&[xi, 1.0 - xi]),
                        Some(self.pressure.get(i)),
                        self.vapor_molefracs
                            .as_ref()
                            .map(|x| arr1(&[x[i], 1.0 - x[i]]))
                            .as_ref(),
                        options,
                    )?
                    .vapor()
                    .pressure(Contributions::Total),
                );
            }
        }
        if let Some(vapor_molefracs) = &self.vapor_molefracs {
            for (i, &xi) in vapor_molefracs.iter().enumerate() {
                prediction.push(
                    PhaseEquilibrium::dew_point(
                        eos,
                        self.temperature.get(i),
                        &arr1(&[xi, 1.0 - xi]),
                        Some(self.pressure.get(i)),
                        self.liquid_molefracs
                            .as_ref()
                            .map(|x| arr1(&[x[i], 1.0 - x[i]]))
                            .as_ref(),
                        options,
                    )?
                    .vapor()
                    .pressure(Contributions::Total),
                );
            }
        }

        Ok(QuantityArray1::from_vec(prediction))
    }

    fn predict_chemical_potential<E: EquationOfState>(
        &self,
        eos: &Rc<E>,
    ) -> Result<QuantityArray1<U>, EstimatorError>
    where
        Quantity<f64, U>: std::fmt::Display,
    {
        if let (Some(liquid_molefracs), Some(vapor_molefracs)) =
            (&self.liquid_molefracs, &self.vapor_molefracs)
        {
            let mut prediction = Vec::new();
            for (((&xi, &yi), t), p) in liquid_molefracs
                .iter()
                .zip(vapor_molefracs.iter())
                .zip(self.temperature.into_iter())
                .zip(self.pressure.into_iter())
            {
                let liquid_moles = arr1(&[xi, 1.0 - xi]) * U::reference_moles();
                let liquid =
                    State::new_npt(eos, t, p, &liquid_moles, DensityInitialization::Liquid)?;
                let mu_liquid = liquid.chemical_potential(Contributions::Total);
                let vapor_moles = arr1(&[yi, 1.0 - yi]) * U::reference_moles();
                let vapor = State::new_npt(eos, t, p, &vapor_moles, DensityInitialization::Vapor)?;
                let mu_vapor = vapor.chemical_potential(Contributions::Total);

                prediction
                    .push(mu_liquid.get(0) - mu_vapor.get(0) + 500.0 * U::reference_molar_energy());
                prediction
                    .push(mu_liquid.get(1) - mu_vapor.get(1) + 500.0 * U::reference_molar_energy());
            }
            Ok(QuantityArray1::from_vec(prediction))
        } else {
            Err(EstimatorError::IncompatibleInput)
        }
    }

    // fn predict_distance<E: EquationOfState>(
    //     &self,
    //     eos: &Rc<E>,
    // ) -> Result<QuantityArray1<U>, EstimatorError>
    // where
    //     Quantity<f64, U>: std::fmt::Display,
    // {
    //     unimplemented!()
    // }

    // fn distance_cost<E: EquationOfState>(&self, eos: &Rc<E>) -> Result<Array1<f64>, EstimatorError>
    // where
    //     Quantity<f64, U>: std::fmt::Display,
    // {
    //     unimplemented!()
    //     // let dx = 1e-4;
    //     // let tol = 1e-9;
    //     // let max_iter = 60;
    //     // let options = (SolverOptions::default(), SolverOptions::default());
    //     // let mut cost = Array1::zeros(self.datapoints);

    //     // for i in 0..self.datapoints {
    //     //     let xi = self.liquid_molefracs[i];
    //     //     let mut dxi = if xi < 0.5 { dx } else { -dx };
    //     //     let temperature = self.temperature.get(i);
    //     //     let pressure = self.pressure.get(i);
    //     //     let mut shift = 0.0;
    //     //     'iteration: for i in 0..max_iter {
    //     //         let damping = match i {
    //     //             i if i <= 2 => 0.75,
    //     //             i if i > 8 && shift < 1e-5 => 0.5,
    //     //             i if i > 25 => 0.25,
    //     //             _ => 1.0,
    //     //         };

    //     //         let xi_f = xi + shift * damping;
    //     //         let prediction = PhaseEquilibrium::bubble_point(
    //     //             eos,
    //     //             temperature,
    //     //             &arr1(&vec![xi_f, 1.0 - xi_f]),
    //     //             Some(pressure),
    //     //             None,
    //     //             options,
    //     //         );
    //     //         if prediction.is_err() {
    //     //             cost[i] = 10.0;
    //     //             break 'iteration;
    //     //         }
    //     //         let p1 = prediction.unwrap().vapor().pressure(Contributions::Total);

    //     //         if xi_f > 1.0 - dxi {
    //     //             dxi *= -1.0
    //     //         };

    //     //         let xi_b = xi_f + dxi;
    //     //         let prediction = PhaseEquilibrium::bubble_point(
    //     //             eos,
    //     //             temperature,
    //     //             &arr1(&vec![xi_b, 1.0 - xi_b]),
    //     //             Some(pressure),
    //     //             None,
    //     //             options,
    //     //         );
    //     //         if prediction.is_err() {
    //     //             cost[i] = 10.0;
    //     //             break 'iteration;
    //     //         }
    //     //         let p2 = prediction.unwrap().vapor().pressure(Contributions::Total);
    //     //         let mut line_vec = arr1(&[dxi, (p2 - p1).to_reduced(pressure)?]);
    //     //         line_vec /= line_vec.mapv(|li| li * li).sum().sqrt();
    //     //         let exp_vec = arr1(&[xi - xi_f, (pressure - p1).to_reduced(pressure)?]);
    //     //         cost[i] = (&exp_vec * &exp_vec).sum().sqrt();
    //     //         shift = line_vec[0] * (&line_vec * &exp_vec).sum().sqrt();
    //     //         if shift > xi_f {
    //     //             shift = xi_f
    //     //         }
    //     //         if shift < -xi_f {
    //     //             shift = -xi_f
    //     //         }
    //     //         if shift.abs() <= tol {
    //     //             break 'iteration;
    //     //         }
    //     //     }
    //     // }
    //     // Ok(cost)
    // }
}

impl<U: EosUnit, E: EquationOfState> DataSet<U, E> for BinaryVle<U>
where
    Quantity<f64, U>: std::fmt::Display + LowerExp,
{
    fn target(&self) -> &QuantityArray1<U> {
        &self.target
    }

    fn target_str(&self) -> &str {
        match self.cost_function {
            CostFunction::Pressure => "pressure",
            CostFunction::ChemicalPotential => "chemical potential",
            // CostFunction::Distance => "distance",
        }
    }

    fn input_str(&self) -> Vec<&str> {
        let mut vec = vec!["temperature", "pressure"];
        if self.liquid_molefracs.is_some() {
            vec.push("liquid molefracs")
        }
        if self.vapor_molefracs.is_some() {
            vec.push("vapor molefracs")
        }
        vec
    }

    fn predict(&self, eos: &Rc<E>) -> Result<QuantityArray1<U>, EstimatorError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        match self.cost_function {
            CostFunction::Pressure => self.predict_pressure(eos),
            CostFunction::ChemicalPotential => self.predict_chemical_potential(eos),
            // CostFunction::Distance => self.predict_distance(eos),
        }
    }

    fn get_input(&self) -> HashMap<String, QuantityArray1<U>> {
        let mut m = HashMap::with_capacity(4);
        m.insert("temperature".to_owned(), self.temperature.clone());
        m.insert("pressure".to_owned(), self.pressure.clone());
        if let Some(liquid_molefracs) = &self.liquid_molefracs {
            m.insert(
                "liquid_molefracs".to_owned(),
                liquid_molefracs * U::reference_moles() / U::reference_moles(),
            );
        }
        if let Some(vapor_molefracs) = &self.vapor_molefracs {
            m.insert(
                "vapor_molefracs".to_owned(),
                vapor_molefracs * U::reference_moles() / U::reference_moles(),
            );
        }
        m
    }
}

// #[derive(Clone)]
// pub struct BinaryTPy<U: EosUnit> {
//     temperature: QuantityArray1<U>,
//     pressure: QuantityArray1<U>,
//     vapor_molefracs: Array1<f64>,
//     datapoints: usize,
// }

// impl<U: EosUnit> BinaryTPy<U> {
//     pub fn new(
//         temperature: QuantityArray1<U>,
//         pressure: QuantityArray1<U>,
//         vapor_molefracs: Array1<f64>,
//     ) -> Result<Self, EstimatorError> {
//         let datapoints = temperature.len();
//         Ok(Self {
//             temperature,
//             pressure,
//             vapor_molefracs,
//             datapoints,
//         })
//     }

//     fn pressure_cost<E: EquationOfState>(&self, eos: &Rc<E>) -> Result<Array1<f64>, EstimatorError>
//     where
//         Quantity<f64, U>: std::fmt::Display,
//     {
//         let options = (SolverOptions::default(), SolverOptions::default());

//         let mut cost = Array1::zeros(self.datapoints);
//         for i in 0..self.datapoints {
//             let yi = self.vapor_molefracs[i];
//             let prediction = PhaseEquilibrium::dew_point(
//                 eos,
//                 self.temperature.get(i),
//                 &arr1(&vec![yi, 1.0 - yi]),
//                 Some(self.pressure.get(i)),
//                 None,
//                 options,
//             )?
//             .vapor()
//             .pressure(Contributions::Total);

//             cost[i] = ((self.pressure.get(i) - prediction) / self.pressure.get(i)).into_value()?
//         }
//         Ok(cost)
//     }
// }

// impl<U: EosUnit, E: EquationOfState> DataSet<U, E> for BinaryTPy<U>
// where
//     Quantity<f64, U>: std::fmt::Display + LowerExp,
// {
//     fn cost(&self, eos: &Rc<E>) -> Result<Array1<f64>, EstimatorError> {
//         self.pressure_cost(eos)
//     }
// }

// #[derive(Clone)]
// pub struct BinaryTPxy<U: EosUnit> {
//     temperature: QuantityArray1<U>,
//     pressure: QuantityArray1<U>,
//     liquid_molefracs: Array1<f64>,
//     vapor_molefracs: Array1<f64>,
//     datapoints: usize,
// }

// impl<U: EosUnit> BinaryTPxy<U> {
//     pub fn new(
//         temperature: QuantityArray1<U>,
//         pressure: QuantityArray1<U>,
//         liquid_molefracs: Array1<f64>,
//         vapor_molefracs: Array1<f64>,
//     ) -> Result<Self, EstimatorError> {
//         let datapoints = temperature.len();
//         Ok(Self {
//             temperature,
//             pressure,
//             liquid_molefracs,
//             vapor_molefracs,
//             datapoints,
//         })
//     }

//     fn pressure_cost<E: EquationOfState>(&self, eos: &Rc<E>) -> Result<Array1<f64>, EstimatorError>
//     where
//         Quantity<f64, U>: std::fmt::Display,
//     {
//         let options = (SolverOptions::default(), SolverOptions::default());

//         let mut cost = Array1::zeros(2 * self.datapoints);
//         for i in 0..self.datapoints {
//             let xi = self.liquid_molefracs[i];
//             let yi = self.vapor_molefracs[i];

//             let prediction_liquid = PhaseEquilibrium::bubble_point(
//                 eos,
//                 self.temperature.get(i),
//                 &arr1(&vec![xi, 1.0 - xi]),
//                 Some(self.pressure.get(i)),
//                 Some(&arr1(&vec![yi, 1.0 - yi])),
//                 options,
//             )?
//             .vapor()
//             .pressure(Contributions::Total);

//             let prediction_vapor = PhaseEquilibrium::dew_point(
//                 eos,
//                 self.temperature.get(i),
//                 &arr1(&vec![yi, 1.0 - yi]),
//                 Some(self.pressure.get(i)),
//                 Some(&arr1(&vec![xi, 1.0 - xi])),
//                 options,
//             )?
//             .vapor()
//             .pressure(Contributions::Total);

//             cost[i] =
//                 ((self.pressure.get(i) - prediction_liquid) / self.pressure.get(i)).into_value()?;
//             cost[self.datapoints + i] =
//                 ((self.pressure.get(i) - prediction_vapor) / self.pressure.get(i)).into_value()?;
//         }
//         Ok(cost)
//     }

//     fn chemical_potential_cost<E: EquationOfState>(
//         &self,
//         eos: &Rc<E>,
//     ) -> Result<Array1<f64>, EstimatorError>
//     where
//         Quantity<f64, U>: std::fmt::Display,
//     {
//         let mut cost = Array1::zeros(self.datapoints);
//         for i in 0..self.datapoints {
//             let xi = self.liquid_molefracs[i];
//             let yi = self.vapor_molefracs[i];
//             let temperature = self.temperature.get(i);
//             let pressure = self.pressure.get(i);
//             let mu_liquid = State::new_npt(
//                 eos,
//                 temperature,
//                 pressure,
//                 &(arr1(&[xi, 1.0 - xi]) * U::reference_moles()),
//                 feos_core::DensityInitialization::Liquid,
//             )?
//             .chemical_potential(Contributions::Total)
//             .to_reduced(U::reference_molar_energy())?;

//             let mu_vapor = State::new_npt(
//                 eos,
//                 temperature,
//                 pressure,
//                 &(arr1(&[yi, 1.0 - yi]) * U::reference_moles()),
//                 feos_core::DensityInitialization::Vapor,
//             )?
//             .chemical_potential(Contributions::Total)
//             .to_reduced(U::reference_molar_energy())?;
//             cost[i] = (mu_liquid - mu_vapor).mapv(|dmu| dmu * dmu).sum().sqrt();
//         }
//         Ok(cost)
//     }
// }

// impl<U: EosUnit, E: EquationOfState> DataSet<U, E> for BinaryTPxy<U>
// where
//     Quantity<f64, U>: std::fmt::Display + LowerExp,
// {
//     fn cost(&self, eos: &Rc<E>) -> Result<Array1<f64>, EstimatorError> {
//         self.pressure_cost(eos)
//     }
// }
