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
        pub struct PyDataSet(Arc<dyn DataSet<SIUnit, $eos>>);

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
            fn predict(&self, eos: &$py_eos) -> PyResult<PySIArray1> {
                Ok(self.0.predict(&eos.0)?.into())
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
                Ok(Self(Arc::new(VaporPressure::<SIUnit>::new(
                    target.clone().into(),
                    temperature.clone().into(),
                    extrapolate.unwrap_or(false),
                    critical_temperature.and_then(|tc| Some(tc.clone().into())),
                    Some((max_iter, tol, verbosity).into()),
                )?)))
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
                Ok(Self(Arc::new(LiquidDensity::<SIUnit>::new(
                    target.clone().into(),
                    temperature.clone().into(),
                    pressure.clone().into(),
                )?)))
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
                Ok(Self(Arc::new(EquilibriumLiquidDensity::<SIUnit>::new(
                    target.clone().into(),
                    temperature.clone().into(),
                    Some((max_iter, tol, verbosity).into()),
                )?)))
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
            ) -> Self {
                Self(Arc::new(BinaryVleChemicalPotential::new(
                    temperature.clone().into(),
                    pressure.clone().into(),
                    liquid_molefracs.to_owned_array(),
                    vapor_molefracs.to_owned_array(),
                )))
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
            ) -> Self {
                Self(Arc::new(BinaryVlePressure::new(
                    temperature.clone().into(),
                    pressure.clone().into(),
                    molefracs.to_owned_array(),
                    phase,
                )))
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
                temperature_or_pressure: &PySIArray1,
                liquid_molefracs: Option<&PyArray1<f64>>,
                vapor_molefracs: Option<&PyArray1<f64>>,
                npoints: Option<usize>,
            ) -> Self {
                Self(Arc::new(BinaryPhaseDiagram::new(
                    specification.into(),
                    temperature_or_pressure.clone().into(),
                    liquid_molefracs.map(|x| x.to_owned_array()),
                    vapor_molefracs.map(|x| x.to_owned_array()),
                    npoints,
                )))
            }

            /// Return `input` as ``Dict[str, SIArray1]``.
            #[getter]
            fn get_input(&self) -> HashMap<String, PySIArray1> {
                let mut m = HashMap::with_capacity(2);
                self.0.get_input().drain().for_each(|(k, v)| {
                    m.insert(k, PySIArray1::from(v));
                });
                m
            }

            /// Return `target` as ``SIArray1``.
            #[getter]
            fn get_target(&self) -> PySIArray1 {
                PySIArray1::from(self.0.target().clone())
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
        pub struct PyEstimator(Estimator<SIUnit, $eos>);

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
            /// nthreads : int, optional
            ///     The number of threads to use. Defaults to None which deactivates
            ///     the parallel evaluation of each data set.
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
            fn cost<'py>(&self, eos: &$py_eos, nthreads: Option<usize>, py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
                if let Some(nt) = nthreads {
                    let thread_pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(nt)
                        .build().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    return Ok(self.0.par_cost(&eos.0, thread_pool)?.view().to_pyarray(py))
                }
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
            fn predict(&self, eos: &$py_eos) -> PyResult<Vec<PySIArray1>> {
                Ok(self
                    .0
                    .predict(&eos.0)?
                    .iter()
                    .map(|d| PySIArray1::from(d.clone()))
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
            ///
            /// Returns
            /// -------
            /// DataSet
            #[staticmethod]
            #[pyo3(text_signature = "(target, temperature, pressure)")]
            fn viscosity(
                target: &PySIArray1,
                temperature: &PySIArray1,
                pressure: &PySIArray1,
            ) -> PyResult<Self> {
                Ok(Self(Arc::new(Viscosity::<SIUnit>::new(
                    target.clone().into(),
                    temperature.clone().into(),
                    pressure.clone().into(),
                )?)))
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
            ///
            /// Returns
            /// -------
            /// DataSet
            #[staticmethod]
            #[pyo3(text_signature = "(target, temperature, pressure)")]
            fn thermal_conductivity(
                target: &PySIArray1,
                temperature: &PySIArray1,
                pressure: &PySIArray1,
            ) -> PyResult<Self> {
                Ok(Self(Arc::new(ThermalConductivity::<SIUnit>::new(
                    target.clone().into(),
                    temperature.clone().into(),
                    pressure.clone().into(),
                )?)))
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
            ///
            /// Returns
            /// -------
            /// DataSet
            #[staticmethod]
            #[pyo3(text_signature = "(target, temperature, pressure)")]
            fn diffusion(
                target: &PySIArray1,
                temperature: &PySIArray1,
                pressure: &PySIArray1,
            ) -> PyResult<Self> {
                Ok(Self(Arc::new(Diffusion::<SIUnit>::new(
                    target.clone().into(),
                    temperature.clone().into(),
                    pressure.clone().into(),
                )?)))
            }
        }
    };
}
