/// Store experimental vapor pressure data and compare to the equation of state.
#[derive(Clone)]
pub struct VaporPressure<U: EosUnit> {
    pub target: QuantityArray1<U>,
    temperature: QuantityArray1<U>,
    max_temperature: QuantityScalar<U>,
    datapoints: usize,
    std_parameters: Vec<f64>,
}

impl<U: EosUnit> VaporPressure<U> {
    /// Create a new vapor pressure data set.
    ///
    /// Takes the temperature as input and possibly parameters
    /// that describe the standard deviation of vapor pressure as
    /// function of temperature. This standard deviation can be used
    /// as inverse weights in the cost function.
    pub fn new(
        target: QuantityArray1<U>,
        temperature: QuantityArray1<U>,
        std_parameters: Vec<f64>,
    ) -> Result<Self, FitError> {
        let datapoints = target.len();
        let max_temperature = *temperature
            .to_reduced(U::reference_temperature())?
            .max()
            .map_err(|_| FitError::IncompatibleInput)?
            * U::reference_temperature();
        Ok(Self {
            target,
            temperature,
            max_temperature,
            datapoints,
            std_parameters,
        })
    }

    /// Return temperature.
    pub fn temperature(&self) -> QuantityArray1<U> {
        self.temperature.clone()
    }

    /// Returns inverse standard deviation as weights for cost function.
    fn weight_from_std(&self, reduced_temperature: &Array1<f64>) -> Array1<f64> {
        reduced_temperature.map(|t| {
            1.0 / ((-self.std_parameters[0] * t + self.std_parameters[1]).exp()
                + self.std_parameters[2])
        })
    }
}

impl<U: EosUnit, E: EquationOfState> DataSet<U, E> for VaporPressure<U> {
    fn target(&self) -> QuantityArray1<U> {
        self.target.clone()
    }

    fn target_str(&self) -> &str {
        "vapor pressure"
        // r"$p^\text{sat}$"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature"]
    }

    fn predict(&self, eos: &Rc<E>) -> Result<QuantityArray1<U>, FitError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let tc =
            State::critical_point(eos, None, Some(self.max_temperature), VLEOptions::default())?
                .temperature;

        let unit = self.target.get(0);
        let mut prediction = Array1::zeros(self.datapoints) * unit;
        for i in 0..self.datapoints {
            let t = self.temperature.get(i);
            if t < tc {
                let state = PhaseEquilibrium::pure_t(
                    eos,
                    self.temperature.get(i),
                    None,
                    VLEOptions::default(),
                );
                if let Ok(s) = state {
                    prediction.try_set(i, s.liquid().pressure(Contributions::Total))?;
                } else {
                    println!(
                        "Failed to compute vapor pressure, T = {}",
                        self.temperature.get(i)
                    );
                    prediction.try_set(i, f64::NAN * unit)?;
                }
            } else {
                prediction.try_set(i, f64::NAN * unit)?;
            }
        }
        Ok(prediction)
    }

    fn cost(&self, eos: &Rc<E>) -> Result<Array1<f64>, FitError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let tc_inv = 1.0
            / State::critical_point(eos, None, Some(self.max_temperature), VLEOptions::default())?
                .temperature;

        let reduced_temperatures = (0..self.datapoints)
            .map(|i| (self.temperature.get(i) * tc_inv).into_value()?)
            .collect();
        let mut weights = self.weight_from_std(&reduced_temperatures);
        weights /= weights.sum();

        let prediction = &self.predict(eos)?;
        let mut cost = Array1::zeros(self.datapoints);
        for i in 0..self.datapoints {
            if prediction.get(i).is_nan() {
                cost[i] = weights[i]
                    * 5.0
                    * (self.temperature.get(i) - 1.0 / tc_inv)
                        .to_reduced(U::reference_temperature())?;
            } else {
                cost[i] = weights[i]
                    * ((self.target.get(i) - prediction.get(i)) / self.target.get(i))
                        .into_value()?
            }
        }
        Ok(cost)
    }

    fn get_input(&self) -> HashMap<String, QuantityArray1<U>> {
        let mut m = HashMap::with_capacity(1);
        m.insert("temperature".to_owned(), self.temperature());
        m
    }
}

/// Store experimental data of liquid densities and compare to the equation of state.
#[derive(Clone)]
pub struct LiquidDensity<U: EosUnit> {
    pub target: QuantityArray1<U>,
    temperature: QuantityArray1<U>,
    pressure: QuantityArray1<U>,
    datapoints: usize,
}

impl<U: EosUnit> LiquidDensity<U> {
    /// A new data set for liquid densities with pressures and temperatures as input.
    pub fn new(
        target: QuantityArray1<U>,
        temperature: QuantityArray1<U>,
        pressure: QuantityArray1<U>,
    ) -> Result<Self, FitError> {
        let datapoints = target.len();
        Ok(Self {
            target,
            temperature,
            pressure,
            datapoints,
        })
    }

    /// Returns temperature of data points.
    pub fn temperature(&self) -> QuantityArray1<U> {
        self.temperature.clone()
    }

    /// Returns pressure of data points.
    pub fn pressure(&self) -> QuantityArray1<U> {
        self.pressure.clone()
    }
}

impl<U: EosUnit, E: EquationOfState + MolarWeight<U>> DataSet<U, E> for LiquidDensity<U> {
    fn target(&self) -> QuantityArray1<U> {
        self.target.clone()
    }

    fn target_str(&self) -> &str {
        "liquid density"
        // r"$\rho^\text{liquid}$"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature", "pressure"]
    }

    fn predict(&self, eos: &Rc<E>) -> Result<QuantityArray1<U>, FitError> {
        assert_eq!(1, eos.components());
        let moles = arr1(&[1.0]) * U::reference_moles();
        let unit = self.target.get(0);
        let mut prediction = Array1::zeros(self.datapoints) * unit;
        for i in 0..self.datapoints {
            let state = State::new_npt(
                eos,
                self.temperature.get(i),
                self.pressure.get(i),
                &moles,
                DensityInitialization::Liquid,
            );
            if let Ok(s) = state {
                prediction.try_set(i, s.mass_density())?;
            } else {
                prediction.try_set(i, 1.0e10 * unit)?;
            }
        }
        Ok(prediction)
    }

    fn cost(&self, eos: &Rc<E>) -> Result<Array1<f64>, FitError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let n_inv = 1.0 / self.datapoints as f64;
        let prediction = &self.predict(eos)?;
        let mut cost = Array1::zeros(self.datapoints);
        for i in 0..self.datapoints {
            cost[i] = n_inv
                * ((self.target.get(i) - prediction.get(i)) / self.target.get(i)).into_value()?
        }
        Ok(cost)
    }

    fn get_input(&self) -> HashMap<String, QuantityArray1<U>> {
        let mut m = HashMap::with_capacity(2);
        m.insert("temperature".to_owned(), self.temperature());
        m.insert("pressure".to_owned(), self.pressure());
        m
    }
}

/// Store experimental data of liquid densities at VLE and compare to the equation of state.
#[derive(Clone)]
pub struct EquilibriumLiquidDensity<U: EosUnit> {
    pub target: QuantityArray1<U>,
    temperature: QuantityArray1<U>,
    max_temperature: QuantityScalar<U>,
    datapoints: usize,
}

impl<U: EosUnit> EquilibriumLiquidDensity<U> {
    /// A new data set of liquid densities at VLE given temperatures.
    pub fn new(
        target: QuantityArray1<U>,
        temperature: QuantityArray1<U>,
    ) -> Result<Self, FitError> {
        let datapoints = target.len();
        let max_temperature = *temperature
            .to_reduced(U::reference_temperature())?
            .max()
            .map_err(|_| FitError::IncompatibleInput)?
            * U::reference_temperature();
        Ok(Self {
            target,
            temperature,
            max_temperature,
            datapoints,
        })
    }

    /// Returns the temperature of data points.
    pub fn temperature(&self) -> QuantityArray1<U> {
        self.temperature.clone()
    }
}

impl<U: EosUnit, E: EquationOfState + MolarWeight<U>> DataSet<U, E>
    for EquilibriumLiquidDensity<U>
{
    fn target(&self) -> QuantityArray1<U> {
        self.target.clone()
    }

    fn target_str(&self) -> &str {
        "liquid density (equilibrium)"
        // r"$\rho^\text{liquid}_\text{equil}$"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature"]
    }

    fn predict(&self, eos: &Rc<E>) -> Result<QuantityArray1<U>, FitError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let tc =
            State::critical_point(eos, None, Some(self.max_temperature), VLEOptions::default())?
                .temperature;

        let unit = self.target.get(0);
        let mut prediction = Array1::zeros(self.datapoints) * unit;
        for i in 0..self.datapoints {
            let t: QuantityScalar<U> = self.temperature.get(i);
            if t < tc {
                let state: PhaseEquilibrium<U, E, 2> =
                    PhaseEquilibrium::pure_t(eos, t, None, VLEOptions::default())?;
                prediction.try_set(i, state.liquid().mass_density())?;
            } else {
                prediction.try_set(i, f64::NAN * unit)?;
            }
        }
        Ok(prediction)
    }

    fn cost(&self, eos: &Rc<E>) -> Result<Array1<f64>, FitError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let tc =
            State::critical_point(eos, None, Some(self.max_temperature), VLEOptions::default())?
                .temperature;
        let n_inv = 1.0 / self.datapoints as f64;
        let prediction = &self.predict(eos)?;
        let mut cost = Array1::zeros(self.datapoints);
        for i in 0..self.datapoints {
            if prediction.get(i).is_nan() {
                cost[i] = n_inv
                    * 5.0
                    * (self.temperature.get(i) - tc).to_reduced(U::reference_temperature())?;
            } else {
                cost[i] = n_inv
                    * ((self.target.get(i) - prediction.get(i)) / self.target.get(i))
                        .into_value()?
            }
        }
        Ok(cost)
    }

    fn get_input(&self) -> HashMap<String, QuantityArray1<U>> {
        let mut m = HashMap::with_capacity(2);
        m.insert("temperature".to_owned(), self.temperature());
        m
    }
}
