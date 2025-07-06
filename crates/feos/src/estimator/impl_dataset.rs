/// Store experimental vapor pressure data and compare to the equation of state.
#[derive(Clone)]
pub struct VaporPressure {
    pub target: SIArray1,
    temperature: SIArray1,
    max_temperature: SINumber,
    datapoints: usize,
    std_parameters: Vec<f64>,
}

impl VaporPressure {
    /// Create a new vapor pressure data set.
    ///
    /// Takes the temperature as input and possibly parameters
    /// that describe the standard deviation of vapor pressure as
    /// function of temperature. This standard deviation can be used
    /// as inverse weights in the cost function.
    pub fn new(
        target: SIArray1,
        temperature: SIArray1,
        std_parameters: Vec<f64>,
    ) -> Result<Self, FitError> {
        let datapoints = target.len();
        let max_temperature = *temperature
            .to_reduced(SIUnit::reference_temperature())?
            .max()
            .map_err(|_| FitError::IncompatibleInput)?
            * SIUnit::reference_temperature();
        Ok(Self {
            target,
            temperature,
            max_temperature,
            datapoints,
            std_parameters,
        })
    }

    /// Return temperature.
    pub fn temperature(&self) -> SIArray1 {
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

impl<E: EquationOfState> DataSet<E> for VaporPressure {
    fn target(&self) -> SIArray1 {
        self.target.clone()
    }

    fn target_str(&self) -> &str {
        "vapor pressure"
        // r"$p^\text{sat}$"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature"]
    }

    fn predict(&self, eos: &Rc<E>) -> Result<SIArray1, FitError>
    
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
                        .to_reduced(SIUnit::reference_temperature())?;
            } else {
                cost[i] = weights[i]
                    * ((self.target.get(i) - prediction.get(i)) / self.target.get(i))
                        .into_value()?
            }
        }
        Ok(cost)
    }

    fn get_input(&self) -> HashMap<String, SIArray1> {
        let mut m = HashMap::with_capacity(1);
        m.insert("temperature".to_owned(), self.temperature());
        m
    }
}

/// Store experimental data of liquid densities and compare to the equation of state.
#[derive(Clone)]
pub struct LiquidDensity {
    pub target: SIArray1,
    temperature: SIArray1,
    pressure: SIArray1,
    datapoints: usize,
}

impl LiquidDensity {
    /// A new data set for liquid densities with pressures and temperatures as input.
    pub fn new(
        target: SIArray1,
        temperature: SIArray1,
        pressure: SIArray1,
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
    pub fn temperature(&self) -> SIArray1 {
        self.temperature.clone()
    }

    /// Returns pressure of data points.
    pub fn pressure(&self) -> SIArray1 {
        self.pressure.clone()
    }
}

impl<E: EquationOfState + MolarWeight> DataSet<E> for LiquidDensity {
    fn target(&self) -> SIArray1 {
        self.target.clone()
    }

    fn target_str(&self) -> &str {
        "liquid density"
        // r"$\rho^\text{liquid}$"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature", "pressure"]
    }

    fn predict(&self, eos: &Rc<E>) -> Result<SIArray1, FitError> {
        assert_eq!(1, eos.components());
        let moles = arr1(&[1.0]) * SIUnit::reference_moles();
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

    fn get_input(&self) -> HashMap<String, SIArray1> {
        let mut m = HashMap::with_capacity(2);
        m.insert("temperature".to_owned(), self.temperature());
        m.insert("pressure".to_owned(), self.pressure());
        m
    }
}

/// Store experimental data of liquid densities at VLE and compare to the equation of state.
#[derive(Clone)]
pub struct EquilibriumLiquidDensity {
    pub target: SIArray1,
    temperature: SIArray1,
    max_temperature: SINumber,
    datapoints: usize,
}

impl EquilibriumLiquidDensity {
    /// A new data set of liquid densities at VLE given temperatures.
    pub fn new(
        target: SIArray1,
        temperature: SIArray1,
    ) -> Result<Self, FitError> {
        let datapoints = target.len();
        let max_temperature = *temperature
            .to_reduced(SIUnit::reference_temperature())?
            .max()
            .map_err(|_| FitError::IncompatibleInput)?
            * SIUnit::reference_temperature();
        Ok(Self {
            target,
            temperature,
            max_temperature,
            datapoints,
        })
    }

    /// Returns the temperature of data points.
    pub fn temperature(&self) -> SIArray1 {
        self.temperature.clone()
    }
}

impl<E: EquationOfState + MolarWeight> DataSet<E>
    for EquilibriumLiquidDensity
{
    fn target(&self) -> SIArray1 {
        self.target.clone()
    }

    fn target_str(&self) -> &str {
        "liquid density (equilibrium)"
        // r"$\rho^\text{liquid}_\text{equil}$"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature"]
    }

    fn predict(&self, eos: &Rc<E>) -> Result<SIArray1, FitError>
    
    {
        let tc =
            State::critical_point(eos, None, Some(self.max_temperature), VLEOptions::default())?
                .temperature;

        let unit = self.target.get(0);
        let mut prediction = Array1::zeros(self.datapoints) * unit;
        for i in 0..self.datapoints {
            let t: SINumber = self.temperature.get(i);
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
                    * (self.temperature.get(i) - tc).to_reduced(SIUnit::reference_temperature())?;
            } else {
                cost[i] = n_inv
                    * ((self.target.get(i) - prediction.get(i)) / self.target.get(i))
                        .into_value()?
            }
        }
        Ok(cost)
    }

    fn get_input(&self) -> HashMap<String, SIArray1> {
        let mut m = HashMap::with_capacity(2);
        m.insert("temperature".to_owned(), self.temperature());
        m
    }
}
