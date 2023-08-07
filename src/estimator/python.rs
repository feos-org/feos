use super::EstimatorError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;

impl From<EstimatorError> for PyErr {
    fn from(e: EstimatorError) -> PyErr {
        PyRuntimeError::new_err(e.to_string())
    }
}

#[macro_export]
macro_rules! impl_estimator {
    ($eos:ty, $py_eos:ty) => {
        /// Collection of loss functions that can be applied to residuals
        /// to handle outliers.
        #[pyclass(name = "Loss")]
        #[derive(Clone)]
        pub struct PyLoss(Loss);

        #[pymethods]
        impl PyLoss {
            /// Create a linear loss function.
            ///
            /// `loss = s**2 * rho(f**2 / s**2)`
            /// where `rho(z) = z` and `s = 1`.
            ///
            /// Returns
            /// -------
            /// Loss
            #[staticmethod]
            pub fn linear() -> Self {
                Self(Loss::Linear)
            }

            /// Create a loss function according to SoftL1's method.
            ///
            /// `loss = s**2 * rho(f**2 / s**2)`
            /// where `rho(z) = 2 * ((1 + z)**0.5 - 1)`.
            /// `s` is the scaling factor.
            ///
            /// Parameters
            /// ----------
            /// scaling_factor : f64
            ///     Scaling factor for SoftL1 loss function.
            ///
            /// Returns
            /// -------
            /// Loss
            #[staticmethod]
            #[pyo3(text_signature = "(scaling_factor)")]
            pub fn softl1(scaling_factor: f64) -> Self {
                Self(Loss::SoftL1(scaling_factor))
            }

            /// Create a loss function according to Huber's method.
            ///
            /// `loss = s**2 * rho(f**2 / s**2)`
            /// where `rho(z) = z if z <= 1 else 2*z**0.5 - 1`.
            /// `s` is the scaling factor.
            ///
            /// Parameters
            /// ----------
            /// scaling_factor : f64
            ///     Scaling factor for Huber loss function.
            ///
            /// Returns
            /// -------
            /// Loss
            #[staticmethod]
            #[pyo3(text_signature = "(scaling_factor)")]
            pub fn huber(scaling_factor: f64) -> Self {
                Self(Loss::Huber(scaling_factor))
            }

            /// Create a loss function according to Cauchy's method.
            ///
            /// `loss = s**2 * rho(f**2 / s**2)`
            /// where `rho(z) = ln(1 + z)`.
            /// `s` is the scaling factor.
            ///
            /// Parameters
            /// ----------
            /// scaling_factor : f64
            ///     Scaling factor for SoftL1 loss function.
            ///
            /// Returns
            /// -------
            /// Loss
            #[staticmethod]
            #[pyo3(text_signature = "(scaling_factor)")]
            pub fn cauchy(scaling_factor: f64) -> Self {
                Self(Loss::Cauchy(scaling_factor))
            }

            /// Create a loss function according to Arctan's method.
            ///
            /// `loss = s**2 * rho(f**2 / s**2)`
            /// where `rho(z) = arctan(z)`.
            /// `s` is the scaling factor.
            ///
            /// Parameters
            /// ----------
            /// scaling_factor : f64
            ///     Scaling factor for SoftL1 loss function.
            ///
            /// Returns
            /// -------
            /// Loss
            #[staticmethod]
            #[pyo3(text_signature = "(scaling_factor)")]
            pub fn arctan(scaling_factor: f64) -> Self {
                Self(Loss::Arctan(scaling_factor))
            }
        }

        /// A collection of experimental data that can be used to compute
        /// cost functions and make predictions using an equation of state.
        #[pyclass(name = "DataSet")]
        #[derive(Clone)]
        pub struct PyDataSet(Arc<dyn DataSet<$eos>>);

        #[pymethods]
        impl PyDataSet {
            /// Compute the cost function for each input value.
            ///
            /// Parameters
            /// ----------
            /// eos : EquationOfState
            ///     The equation of state that is used.
            /// loss : Loss
            ///     The loss function that is applied to residuals
            ///     to handle outliers.
            ///
            /// Returns
            /// -------
            /// numpy.ndarray[Float]
            ///     The cost function evaluated for each experimental data point.
            ///
            /// Note
            /// ----
            /// The cost function that is used depends on the
            /// property. For most properties it is the absolute relative difference.
            /// See the constructors of the respective properties
            /// to learn about the cost functions that are used.
            #[pyo3(text_signature = "($self, eos, loss)")]
            fn cost<'py>(
                &self,
                eos: &$py_eos,
                loss: PyLoss,
                py: Python<'py>,
            ) -> PyResult<&'py PyArray1<f64>> {
                Ok(self.0.cost(&eos.0, loss.0)?.view().to_pyarray(py))
            }

            /// Return the property of interest for each data point
            /// of the input as computed by the equation of state.
            ///
            /// Parameters
            /// ----------
            /// eos : EquationOfState
            ///     The equation of state that is used.
            ///
            /// Returns
            /// -------
            /// SIArray1
            #[pyo3(text_signature = "($self, eos)")]
            fn predict<'py>(&self, eos: &$py_eos, py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
                Ok(self.0.predict(&eos.0)?.view().to_pyarray(py))
            }

            /// Return the relative difference between experimental data
            /// and prediction of the equation of state.
            ///
            /// The relative difference is computed as:
            ///
            /// .. math:: \text{Relative Difference} = \frac{x_i^\text{prediction} - x_i^\text{experiment}}{x_i^\text{experiment}}
            ///
            /// Parameters
            /// ----------
            /// eos : EquationOfState
            ///     The equation of state that is used.
            ///
            /// Returns
            /// -------
            /// numpy.ndarray[Float]
            #[pyo3(text_signature = "($self, eos)")]
            fn relative_difference<'py>(
                &self,
                eos: &$py_eos,
                py: Python<'py>,
            ) -> PyResult<&'py PyArray1<f64>> {
                Ok(self.0.relative_difference(&eos.0)?.view().to_pyarray(py))
            }

            /// Return the mean absolute relative difference.
            ///
            /// The mean absolute relative difference is computed as:
            ///
            /// .. math:: \text{MARD} = \frac{1}{N}\sum_{i=1}^{N} \left|\frac{x_i^\text{prediction} - x_i^\text{experiment}}{x_i^\text{experiment}} \right|
            ///
            /// Parameters
            /// ----------
            /// eos : EquationOfState
            ///     The equation of state that is used.
            ///
            /// Returns
            /// -------
            /// Float
            #[pyo3(text_signature = "($self, eos)")]
            fn mean_absolute_relative_difference(&self, eos: &$py_eos) -> PyResult<f64> {
                Ok(self.0.mean_absolute_relative_difference(&eos.0)?)
            }

            /// Create a DataSet with experimental data for vapor pressure.
            ///
            /// Parameters
            /// ----------
            /// target : SIArray1
            ///     Experimental data for vapor pressure.
            /// temperature : SIArray1
            ///     Temperature for experimental data points.
            /// extrapolate : bool, optional
            ///     Use Antoine type equation to extrapolate vapor
            ///     pressure if experimental data is above critial
            ///     point of model. Defaults to False.
            /// critical_temperature : SINumber, optional
            ///     Estimate of the critical temperature used as initial
            ///     value for critical point calculation. Defaults to None.
            ///     For additional information, see note.
            /// max_iter : int, optional
            ///     The maximum number of iterations for critical point
            ///     and VLE algorithms.
            /// tol: float, optional
            ///     Solution tolerance for critical point
            ///     and VLE algorithms.
            /// verbosity : Verbosity, optional
            ///     Verbosity for critical point
            ///     and VLE algorithms.
            ///
            /// Returns
            /// -------
            /// ``DataSet``
            ///
            /// Note
            /// ----
            /// If no critical temperature is provided, the maximum of the `temperature` input
            /// is used. If that fails, the default temperatures of the critical point routine
            /// are used.
            #[staticmethod]
            #[pyo3(text_signature = "(target, temperature, extrapolate, critical_temperature=None, max_iter=None, verbosity=None)")]
            fn vapor_pressure(
                target: &PySIArray1,
                temperature: &PySIArray1,
                extrapolate: Option<bool>,
                critical_temperature: Option<&PySINumber>,
                max_iter: Option<usize>,
                tol: Option<f64>,
                verbosity: Option<Verbosity>,
            ) -> PyResult<Self> {
                Ok(Self(Arc::new(VaporPressure::new(
                    target.clone().try_into()?,
                    temperature.clone().try_into()?,
                    extrapolate.unwrap_or(false),
                    critical_temperature.and_then(|tc| tc.clone().try_into().ok()),
                    Some((max_iter, tol, verbosity).into()),
                ))))
            }

            /// Create a DataSet with experimental data for liquid density.
            ///
            /// Parameters
            /// ----------
            /// target : SIArray1
            ///     Experimental data for liquid density.
            /// temperature : SIArray1
            ///     Temperature for experimental data points.
            /// pressure : SIArray1
            ///     Pressure for experimental data points.
            ///
            /// Returns
            /// -------
            /// DataSet
            #[staticmethod]
            #[pyo3(text_signature = "(target, temperature, pressure)")]
            fn liquid_density(
                target: &PySIArray1,
                temperature: &PySIArray1,
                pressure: &PySIArray1,
            ) -> PyResult<Self> {
                Ok(Self(Arc::new(LiquidDensity::new(
                    target.clone().try_into()?,
                    temperature.clone().try_into()?,
                    pressure.clone().try_into()?,
                ))))
            }

            /// Create a DataSet with experimental data for liquid density
            /// for a vapor liquid equilibrium.
            ///
            /// Parameters
            /// ----------
            /// target : SIArray1
            ///     Experimental data for liquid density.
            /// temperature : SIArray1
            ///     Temperature for experimental data points.
            /// max_iter : int, optional
            ///     The maximum number of iterations for critical point
            ///     and VLE algorithms.
            /// tol: float, optional
            ///     Solution tolerance for critical point
            ///     and VLE algorithms.
            /// verbosity : Verbosity, optional
            ///     Verbosity for critical point
            ///     and VLE algorithms.
            ///
            /// Returns
            /// -------
            /// DataSet
            #[staticmethod]
            #[pyo3(text_signature = "(target, temperature)")]
            fn equilibrium_liquid_density(
                target: &PySIArray1,
                temperature: &PySIArray1,
                max_iter: Option<usize>,
                tol: Option<f64>,
                verbosity: Option<Verbosity>,
            ) -> PyResult<Self> {
                Ok(Self(Arc::new(EquilibriumLiquidDensity::new(
                    target.clone().try_into()?,
                    temperature.clone().try_into()?,
                    Some((max_iter, tol, verbosity).into()),
                ))))
            }

            /// Create a DataSet with experimental data for binary
            /// phase equilibria using the chemical potential residual.
            ///
            /// Parameters
            /// ----------
            /// temperature : SIArray1
            ///     Temperature of the experimental data points.
            /// pressure : SIArray1
            ///     Pressure of the experimental data points.
            /// liquid_molefracs : np.array[float]
            ///     Molar composition of component 1 in the liquid phase.
            /// vapor_molefracs : np.array[float]
            ///     Molar composition of component 1 in the vapor phase.
            ///
            /// Returns
            /// -------
            /// DataSet
            #[staticmethod]
            #[pyo3(text_signature = "(temperature, pressure, liquid_molefracs, vapor_molefracs)")]
            fn binary_vle_chemical_potential(
                temperature: &PySIArray1,
                pressure: &PySIArray1,
                liquid_molefracs: &PyArray1<f64>,
                vapor_molefracs: &PyArray1<f64>,
            ) -> PyResult<Self> {
                Ok(Self(Arc::new(BinaryVleChemicalPotential::new(
                    temperature.clone().try_into()?,
                    pressure.clone().try_into()?,
                    liquid_molefracs.to_owned_array(),
                    vapor_molefracs.to_owned_array(),
                ))))
            }

            /// Create a DataSet with experimental data for binary
            /// phase equilibria using the pressure residual.
            ///
            /// Parameters
            /// ----------
            /// temperature : SIArray1
            ///     Temperature of the experimental data points.
            /// pressure : SIArray1
            ///     Pressure of the experimental data points.
            /// molefracs : np.array[float]
            ///     Molar composition of component 1 in the considered phase.
            /// phase : Phase
            ///     The phase of the experimental data points.
            ///
            /// Returns
            /// -------
            /// DataSet
            #[staticmethod]
            #[pyo3(text_signature = "(temperature, pressure, molefracs, phase)")]
            fn binary_vle_pressure(
                temperature: &PySIArray1,
                pressure: &PySIArray1,
                molefracs: &PyArray1<f64>,
                phase: Phase,
            ) -> PyResult<Self> {
                Ok(Self(Arc::new(BinaryVlePressure::new(
                    temperature.clone().try_into()?,
                    pressure.clone().try_into()?,
                    molefracs.to_owned_array(),
                    phase,
                ))))
            }

            /// Create a DataSet with experimental data for binary
            /// phase diagrams using the distance residual.
            ///
            /// Parameters
            /// ----------
            /// specification : SINumber
            ///     The constant temperature/pressure of the isotherm/isobar.
            /// temperature_or_pressure : SIArray1
            ///     The temperature (isobar) or pressure (isotherm) of the
            ///     experimental data points.
            /// liquid_molefracs : np.array[float], optional
            ///     Molar composition of component 1 in the liquid phase.
            /// vapor_molefracs : np.array[float], optional
            ///     Molar composition of component 1 in the vapor phase.
            /// npoints : int, optional
            ///     The resolution of the phase diagram used to calculate
            ///     the distance residual.
            ///
            /// Returns
            /// -------
            /// DataSet
            #[staticmethod]
            #[pyo3(text_signature = "(specification, temperature_or_pressure, liquid_molefracs=None, vapor_molefracs=None, npoints=None)")]
            fn binary_phase_diagram(
                specification: PySINumber,
                temperature_or_pressure: PySIArray1,
                liquid_molefracs: Option<&PyArray1<f64>>,
                vapor_molefracs: Option<&PyArray1<f64>>,
                npoints: Option<usize>,
            ) -> PyResult<Self> {
                if let Ok(t) = Temperature::<f64>::try_from(specification) {
                    Ok(Self(Arc::new(BinaryPhaseDiagram::new(
                        t,
                        temperature_or_pressure.try_into()?,
                        liquid_molefracs.map(|x| x.to_owned_array()),
                        vapor_molefracs.map(|x| x.to_owned_array()),
                        npoints,
                    ))))
                } else if let Ok(p) = Pressure::<f64>::try_from(specification) {
                    Ok(Self(Arc::new(BinaryPhaseDiagram::new(
                        p,
                        temperature_or_pressure.try_into()?,
                        liquid_molefracs.map(|x| x.to_owned_array()),
                        vapor_molefracs.map(|x| x.to_owned_array()),
                        npoints,
                    ))))
                } else {
                    Ok(Err(EosError::WrongUnits("temperature or pressure".into(),
                        quantity::si::SINumber::from(specification).to_string()
                    ))?)
                }
            }

            /// Return `target` as ``SIArray1``.
            #[getter]
            fn get_target<'py>(&self, py: Python<'py>,) -> &'py PyArray1<f64> {
                self.0.target().view().to_pyarray(py)
            }

            /// Return number of stored data points.
            #[getter]
            fn get_datapoints(&self) -> usize {
                self.0.datapoints()
            }

            fn __repr__(&self) -> PyResult<String> {
                Ok(self.0.to_string())
            }
        }

        /// A collection of `DataSet`s that can be used to compute metrics for experimental data.
        ///
        /// Parameters
        /// ----------
        /// data : List[DataSet]
        ///     The properties and experimental data points to add to
        ///     the estimator.
        /// weights : List[float]
        ///     The weight of each property. When computing the cost function,
        ///     the weights are normalized (sum of weights equals unity).
        /// losses : List[Loss]
        ///     The loss functions for each property.
        ///
        /// Returns
        /// -------
        /// Estimator
        #[pyclass(name = "Estimator")]
        #[pyo3(text_signature = "(data, weights, losses)")]
        pub struct PyEstimator(Estimator<$eos>);

        #[pymethods]
        impl PyEstimator {
            #[new]
            fn new(data: Vec<PyDataSet>, weights: Vec<f64>, losses: Vec<PyLoss>) -> Self {
                Self(Estimator::new(
                    data.iter().map(|d| d.0.clone()).collect(),
                    weights,
                    losses.iter().map(|l| l.0.clone()).collect(),
                ))
            }

            /// Compute the cost function for each ``DataSet``.
            ///
            /// Parameters
            /// ----------
            /// eos : EquationOfState
            ///     The equation of state that is used.
            ///
            /// Returns
            /// -------
            /// numpy.ndarray[Float]
            ///     The cost function evaluated for each experimental data point
            ///     of each ``DataSet``.
            ///
            /// Note
            /// ----
            /// The cost function is:
            ///
            /// - The relative difference between prediction and target value,
            /// - to which a loss function is applied,
            /// - and which is weighted according to the number of datapoints,
            /// - and the relative weights as defined in the Estimator object.
            #[pyo3(text_signature = "($self, eos)")]
            fn cost<'py>(&self, eos: &$py_eos, py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
                Ok(self.0.cost(&eos.0)?.view().to_pyarray(py))
            }

            /// Return the properties as computed by the
            /// equation of state for each `DataSet`.
            ///
            /// Parameters
            /// ----------
            /// eos : EquationOfState
            ///     The equation of state that is used.
            ///
            /// Returns
            /// -------
            /// List[SIArray1]
            #[pyo3(text_signature = "($self, eos)")]
            fn predict<'py>(&self, eos: &$py_eos, py: Python<'py>) -> PyResult<Vec<&'py PyArray1<f64>>> {
                Ok(self
                    .0
                    .predict(&eos.0)?
                    .iter()
                    .map(|d| d.view().to_pyarray(py))
                    .collect())
            }

            /// Return the relative difference between experimental data
            /// and prediction of the equation of state for each ``DataSet``.
            ///
            /// The relative difference is computed as:
            ///
            /// .. math:: \text{Relative Difference} = \frac{x_i^\text{prediction} - x_i^\text{experiment}}{x_i^\text{experiment}}
            ///
            /// Parameters
            /// ----------
            /// eos : EquationOfState
            ///     The equation of state that is used.
            ///
            /// Returns
            /// -------
            /// List[numpy.ndarray[Float]]
            #[pyo3(text_signature = "($self, eos)")]
            fn relative_difference<'py>(
                &self,
                eos: &$py_eos,
                py: Python<'py>,
            ) -> PyResult<Vec<&'py PyArray1<f64>>> {
                Ok(self
                    .0
                    .relative_difference(&eos.0)?
                    .iter()
                    .map(|d| d.view().to_pyarray(py))
                    .collect())
            }

            /// Return the mean absolute relative difference for each ``DataSet``.
            ///
            /// The mean absolute relative difference is computed as:
            ///
            /// .. math:: \text{MARD} = \frac{1}{N}\sum_{i=1}^{N} \left|\frac{x_i^\text{prediction} - x_i^\text{experiment}}{x_i^\text{experiment}} \right|
            ///
            /// Parameters
            /// ----------
            /// eos : EquationOfState
            ///     The equation of state that is used.
            ///
            /// Returns
            /// -------
            /// numpy.ndarray[Float]
            #[pyo3(text_signature = "($self, eos)")]
            fn mean_absolute_relative_difference<'py>(
                &self,
                eos: &$py_eos,
                py: Python<'py>,
            ) -> PyResult<&'py PyArray1<f64>> {
                Ok(self
                    .0
                    .mean_absolute_relative_difference(&eos.0)?
                    .view()
                    .to_pyarray(py))
            }

            /// Return the stored ``DataSet``s.
            ///
            /// Returns
            /// -------
            /// List[DataSet]
            #[getter]
            fn get_datasets(&self) -> Vec<PyDataSet> {
                self.0
                    .datasets()
                    .iter()
                    .cloned()
                    .map(|ds| PyDataSet(ds))
                    .collect()
            }

            fn _repr_markdown_(&self) -> String {
                self.0._repr_markdownn_()
            }

            fn __repr__(&self) -> PyResult<String> {
                Ok(self.0.to_string())
            }
        }
    };
}

#[macro_export]
macro_rules! impl_estimator_entropy_scaling {
    ($eos:ty, $py_eos:ty) => {
        #[pymethods]
        impl PyDataSet {
            /// Create a DataSet with experimental data for viscosity.
            ///
            /// Parameters
            /// ----------
            /// target : SIArray1
            ///     Experimental data for viscosity.
            /// temperature : SIArray1
            ///     Temperature for experimental data points.
            /// pressure : SIArray1
            ///     Pressure for experimental data points.
            /// phase : List[Phase], optional
            ///     Phase of data. Used to determine the starting
            ///     density for the density iteration. If provided,
            ///     resulting states may not be stable.
            ///
            /// Returns
            /// -------
            /// DataSet
            #[staticmethod]
            fn viscosity(
                target: &PySIArray1,
                temperature: &PySIArray1,
                pressure: &PySIArray1,
                phase: Option<Vec<Phase>>,
            ) -> PyResult<Self> {
                Ok(Self(Arc::new(crate::estimator::Viscosity::new(
                    target.clone().try_into()?,
                    temperature.clone().try_into()?,
                    pressure.clone().try_into()?,
                    phase.as_ref(),
                ))))
            }

            /// Create a DataSet with experimental data for thermal conductivity.
            ///
            /// Parameters
            /// ----------
            /// target : SIArray1
            ///     Experimental data for thermal conductivity.
            /// temperature : SIArray1
            ///     Temperature for experimental data points.
            /// pressure : SIArray1
            ///     Pressure for experimental data points.
            /// phase : List[Phase], optional
            ///     Phase of data. Used to determine the starting
            ///     density for the density iteration. If provided,
            ///     resulting states may not be stable.
            ///
            /// Returns
            /// -------
            /// DataSet
            #[staticmethod]
            fn thermal_conductivity(
                target: &PySIArray1,
                temperature: &PySIArray1,
                pressure: &PySIArray1,
                phase: Option<Vec<Phase>>,
            ) -> PyResult<Self> {
                Ok(Self(Arc::new(crate::estimator::ThermalConductivity::new(
                    target.clone().try_into()?,
                    temperature.clone().try_into()?,
                    pressure.clone().try_into()?,
                    phase.as_ref(),
                ))))
            }

            /// Create a DataSet with experimental data for diffusion coefficient.
            ///
            /// Parameters
            /// ----------
            /// target : SIArray1
            ///     Experimental data for diffusion coefficient.
            /// temperature : SIArray1
            ///     Temperature for experimental data points.
            /// pressure : SIArray1
            ///     Pressure for experimental data points.
            /// phase : List[Phase], optional
            ///     Phase of data. Used to determine the starting
            ///     density for the density iteration. If provided,
            ///     resulting states may not be stable.
            ///
            /// Returns
            /// -------
            /// DataSet
            #[staticmethod]
            fn diffusion(
                target: &PySIArray1,
                temperature: &PySIArray1,
                pressure: &PySIArray1,
                phase: Option<Vec<Phase>>,
            ) -> PyResult<Self> {
                Ok(Self(Arc::new(Diffusion::new(
                    target.clone().try_into()?,
                    temperature.clone().try_into()?,
                    pressure.clone().try_into()?,
                    phase.as_ref(),
                ))))
            }
        }
    };
}
