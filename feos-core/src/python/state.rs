#[macro_export]
macro_rules! impl_state {
    ($eos:ty, $py_eos:ty) => {
        /// A thermodynamic state at given conditions.
        ///
        /// Parameters
        /// ----------
        /// eos : Eos
        ///     The equation of state to use.
        /// temperature : SINumber, optional
        ///     Temperature.
        /// volume : SINumber, optional
        ///     Volume.
        /// density : SINumber, optional
        ///     Molar density.
        /// partial_density : SIArray1, optional
        ///     Partial molar densities.
        /// total_moles : SINumber, optional
        ///     Total amount of substance (of a mixture).
        /// moles : SIArray1, optional
        ///     Amount of substance for each component.
        /// molefracs : numpy.ndarray[float]
        ///     Molar fraction of each component.
        /// pressure : SINumber, optional
        ///     Pressure.
        /// molar_enthalpy : SINumber, optional
        ///     Molar enthalpy.
        /// molar_entropy : SINumber, optional
        ///     Molar entropy.
        /// molar_internal_energy: SINumber, optional
        ///     Molar internal energy
        /// density_initialization : {'vapor', 'liquid', SINumber, None}, optional
        ///     Method used to initialize density for density iteration.
        ///     'vapor' and 'liquid' are inferred from the maximum density of the equation of state.
        ///     If no density or keyword is provided, the vapor and liquid phase is tested and, if
        ///     different, the result with the lower free energy is returned.
        /// initial_temperature : SINumber, optional
        ///     Initial temperature for temperature iteration. Can improve convergence
        ///     when the state is specified with pressure and molar entropy or enthalpy.
        ///
        /// Returns
        /// -------
        /// State : state at given conditions
        ///
        /// Raises
        /// ------
        /// Error
        ///     When the state cannot be created using the combination of input.
        #[pyclass(name = "State")]
        #[derive(Clone)]
        #[pyo3(text_signature = "(eos, temperature=None, volume=None, density=None, partial_density=None, total_moles=None, moles=None, molefracs=None, pressure=None, molar_enthalpy=None, molar_entropy=None, molar_internal_energy=None, density_initialization=None, initial_temperature=None)")]
        pub struct PyState(pub State<$eos>);

        #[pymethods]
        impl PyState {
            #[new]
            pub fn new(
                eos: $py_eos,
                temperature: Option<PySINumber>,
                volume: Option<PySINumber>,
                density: Option<PySINumber>,
                partial_density: Option<PySIArray1>,
                total_moles: Option<PySINumber>,
                moles: Option<PySIArray1>,
                molefracs: Option<&PyArray1<f64>>,
                pressure: Option<PySINumber>,
                molar_enthalpy: Option<PySINumber>,
                molar_entropy: Option<PySINumber>,
                molar_internal_energy: Option<PySINumber>,
                density_initialization: Option<&PyAny>,
                initial_temperature: Option<PySINumber>,
            ) -> PyResult<Self> {
                let x = molefracs.and_then(|m| Some(m.to_owned_array()));
                let density_init = if let Some(di) = density_initialization {
                    if let Ok(d) = di.extract::<&str>() {
                        match d {
                            "vapor" => Ok(DensityInitialization::Vapor),
                            "liquid" => Ok(DensityInitialization::Liquid),
                            _ => Err(PyErr::new::<PyValueError, _>(format!(
                                "`density_initialization` must be 'vapor' or 'liquid'."
                            ))),
                        }
                    } else if let Ok(d) = di.extract::<PySINumber>() {
                        Ok(DensityInitialization::InitialDensity(d.try_into()?))
                    } else {
                        Err(PyErr::new::<PyValueError, _>(format!(
                            "`density_initialization` must be 'vapor' or 'liquid' or a molar density as `SINumber` has to be provided."
                        )))
                    }
                } else {
                    Ok(DensityInitialization::None)
                };
                let s = State::new_full(
                    &eos.0,
                    temperature.map(|t| t.try_into()).transpose()?,
                    volume.map(|v| v.try_into()).transpose()?,
                    density.map(|s| s.try_into()).transpose()?,
                    partial_density.map(|s| s.try_into()).transpose()?.as_ref(),
                    total_moles.map(|s| s.try_into()).transpose()?,
                    moles.map(|m| m.try_into()).transpose()?.as_ref(),
                    x.as_ref(),
                    pressure.map(|s| s.try_into()).transpose()?,
                    molar_enthalpy.map(|s| s.try_into()).transpose()?,
                    molar_entropy.map(|s| s.try_into()).transpose()?,
                    molar_internal_energy.map(|s| s.try_into()).transpose()?,
                    density_init?,
                    initial_temperature.map(|s| s.try_into()).transpose()?,
                )?;
                Ok(Self(s))
            }

            /// Return a list of thermodynamic state at critical conditions
            /// for each pure substance in the system.
            ///
            /// Parameters
            /// ----------
            /// eos: EquationOfState
            ///     The equation of state to use.
            /// initial_temperature: SINumber, optional
            ///     The initial temperature.
            /// max_iter : int, optional
            ///     The maximum number of iterations.
            /// tol: float, optional
            ///     The solution tolerance.
            /// verbosity : Verbosity, optional
            ///     The verbosity.
            ///
            /// Returns
            /// -------
            /// State : tate at critical conditions
            #[staticmethod]
            #[pyo3(text_signature = "(eos, initial_temperature=None, max_iter=None, tol=None, verbosity=None)")]
            fn critical_point_pure(
                eos: $py_eos,
                initial_temperature: Option<PySINumber>,
                max_iter: Option<usize>,
                tol: Option<f64>,
                verbosity: Option<Verbosity>,
            ) -> PyResult<Vec<Self>> {
                let t = initial_temperature.map(|t0| t0.try_into()).transpose()?;
                let cp = State::critical_point_pure(&eos.0, t, (max_iter, tol, verbosity).into())?;
                Ok(cp.into_iter().map(Self).collect())
            }

            /// Create a thermodynamic state at critical conditions.
            ///
            /// Parameters
            /// ----------
            /// eos: EquationOfState
            ///     The equation of state to use.
            /// moles: SIArray1, optional
            ///     Amount of substance of each component.
            ///     Only optional for a pure component.
            /// initial_temperature: SINumber, optional
            ///     The initial temperature.
            /// max_iter : int, optional
            ///     The maximum number of iterations.
            /// tol: float, optional
            ///     The solution tolerance.
            /// verbosity : Verbosity, optional
            ///     The verbosity.
            ///
            /// Returns
            /// -------
            /// State : State at critical conditions.
            #[staticmethod]
            #[pyo3(text_signature = "(eos, moles=None, initial_temperature=None, max_iter=None, tol=None, verbosity=None)")]
            fn critical_point(
                eos: $py_eos,
                moles: Option<PySIArray1>,
                initial_temperature: Option<PySINumber>,
                max_iter: Option<usize>,
                tol: Option<f64>,
                verbosity: Option<Verbosity>,
            ) -> PyResult<Self> {
                Ok(PyState(State::critical_point(
                    &eos.0,
                    moles.map(|m| m.try_into()).transpose()?.as_ref(),
                    initial_temperature.map(|t| t.try_into()).transpose()?,
                    (max_iter, tol, verbosity).into(),
                )?))
            }

            /// Create a thermodynamic state at critical conditions for a binary system.
            ///
            /// Parameters
            /// ----------
            /// eos: EquationOfState
            ///     The equation of state to use.
            /// temperature_or_pressure: SINumber
            ///     temperature_or_pressure.
            /// initial_temperature: SINumber, optional
            ///     An initial guess for the temperature.
            /// initial_molefracs: [float], optional
            ///     An initial guess for the composition.
            /// max_iter : int, optional
            ///     The maximum number of iterations.
            /// tol: float, optional
            ///     The solution tolerance.
            /// verbosity : Verbosity, optional
            ///     The verbosity.
            ///
            /// Returns
            /// -------
            /// State : State at critical conditions.
            #[staticmethod]
            #[pyo3(text_signature = "(eos, temperature_or_pressure, initial_temperature=None, initial_molefracs=None, max_iter=None, tol=None, verbosity=None)")]
            fn critical_point_binary(
                eos: $py_eos,
                temperature_or_pressure: PySINumber,
                initial_temperature: Option<PySINumber>,
                initial_molefracs: Option<[f64; 2]>,
                max_iter: Option<usize>,
                tol: Option<f64>,
                verbosity: Option<Verbosity>,
            ) -> PyResult<Self> {
                Ok(PyState(State::critical_point_binary(
                    &eos.0,
                    TPSpec::try_from(temperature_or_pressure)?,
                    initial_temperature.map(|t| t.try_into()).transpose()?,
                    initial_molefracs,
                    (max_iter, tol, verbosity).into(),
                )?))
            }

            /// Calculate spinodal states for a given temperature and composition.
            ///
            /// Parameters
            /// ----------
            /// eos: EquationOfState
            ///     The equation of state to use.
            /// temperature: SINumber
            ///     The temperature.
            /// moles: SIArray1, optional
            ///     Amount of substance of each component.
            ///     Only optional for a pure component.
            /// max_iter : int, optional
            ///     The maximum number of iterations.
            /// tol: float, optional
            ///     The solution tolerance.
            /// verbosity : Verbosity, optional
            ///     The verbosity.
            ///
            /// Returns
            /// -------
            /// State : State at critical conditions.
            #[staticmethod]
            #[pyo3(text_signature = "(eos, temperature, moles=None, max_iter=None, tol=None, verbosity=None)")]
            fn spinodal(
                eos: $py_eos,
                temperature: PySINumber,
                moles: Option<PySIArray1>,
                max_iter: Option<usize>,
                tol: Option<f64>,
                verbosity: Option<Verbosity>,
            ) -> PyResult<(Self, Self)> {
                let [state1, state2] = State::spinodal(
                    &eos.0,
                    temperature.try_into()?,
                    moles.map(|m| m.try_into()).transpose()?.as_ref(),
                    (max_iter, tol, verbosity).into(),
                )?;
                Ok((PyState(state1), PyState(state2)))
            }

            /// Performs a stability analysis and returns a list of stable
            /// candidate states.
            ///
            /// Parameters
            /// ----------
            /// max_iter : int, optional
            ///     The maximum number of iterations.
            /// tol: float, optional
            ///     The solution tolerance.
            /// verbosity : Verbosity, optional
            ///     The verbosity.
            ///
            /// Returns
            /// -------
            /// State
            #[pyo3(text_signature = "(max_iter=None, tol=None, verbosity=None)")]
            fn stability_analysis(&self,
                max_iter: Option<usize>,
                tol: Option<f64>,
                verbosity: Option<Verbosity>,
            ) -> PyResult<Vec<Self>> {
                Ok(self
                    .0
                    .stability_analysis((max_iter, tol, verbosity).into())?
                    .into_iter()
                    .map(Self)
                    .collect())
            }

            /// Performs a stability analysis and returns whether the state
            /// is stable
            ///
            /// Parameters
            /// ----------
            /// max_iter : int, optional
            ///     The maximum number of iterations.
            /// tol: float, optional
            ///     The solution tolerance.
            /// verbosity : Verbosity, optional
            ///     The verbosity.
            ///
            /// Returns
            /// -------
            /// bool
            #[pyo3(text_signature = "(max_iter=None, tol=None, verbosity=None)")]
            fn is_stable(&self,
                max_iter: Option<usize>,
                tol: Option<f64>,
                verbosity: Option<Verbosity>,
            ) -> PyResult<bool> {
                Ok(self.0.is_stable((max_iter, tol, verbosity).into())?)
            }

            /// Return pressure.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn pressure(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.pressure(contributions))
            }

            /// Return pressure contributions.
            ///
            /// Returns
            /// -------
            /// List[Tuple[str, SINumber]]
            fn pressure_contributions(&self) -> Vec<(String, PySINumber)> {
                self.0
                    .pressure_contributions()
                    .into_iter()
                    .map(|(s, q)| (s, PySINumber::from(q)))
                    .collect()
            }

            /// Return compressibility.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// float
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn compressibility(&self, contributions: Contributions) -> f64 {
                self.0.compressibility(contributions)
            }

            /// Return partial derivative of pressure w.r.t. volume.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn dp_dv(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.dp_dv(contributions))
            }

            /// Return partial derivative of pressure w.r.t. density.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn dp_drho(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.dp_drho(contributions))
            }

            /// Return partial derivative of pressure w.r.t. temperature.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn dp_dt(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.dp_dt(contributions))
            }

            /// Return partial derivative of pressure w.r.t. amount of substance.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SIArray1
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn dp_dni(&self, contributions: Contributions) -> PySIArray1 {
                PySIArray1::from(self.0.dp_dni(contributions))
            }

            /// Return second partial derivative of pressure w.r.t. volume.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn d2p_dv2(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.d2p_dv2(contributions))
            }

            /// Return second partial derivative of pressure w.r.t. density.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn d2p_drho2(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.d2p_drho2(contributions))
            }

            /// Return partial molar volume of each component.
            ///
            /// Returns
            /// -------
            /// SIArray1
            fn partial_molar_volume(&self) -> PySIArray1 {
                PySIArray1::from(self.0.partial_molar_volume())
            }

            /// Return chemical potential of each component.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SIArray1
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn chemical_potential(&self, contributions: Contributions) -> PySIArray1 {
                PySIArray1::from(self.0.chemical_potential(contributions))
            }

            /// Return residual chemical potential contributions.
            ///
            /// Parameters
            /// ----------
            /// component: int
            ///     the component for which the contributions
            ///     are calculated
            ///
            /// Returns
            /// -------
            /// List[Tuple[str, SINumber]]
            fn residual_chemical_potential_contributions(&self, component: usize) -> Vec<(String, PySINumber)> {
                self.0
                    .chemical_potential_contributions(component)
                    .into_iter()
                    .map(|(s, q)| (s, PySINumber::from(q)))
                    .collect()
            }

            /// Return derivative of chemical potential w.r.t temperature.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SIArray1
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn dmu_dt(&self, contributions: Contributions) -> PySIArray1 {
                PySIArray1::from(self.0.dmu_dt(contributions))
            }

            /// Return derivative of chemical potential w.r.t amount of substance.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SIArray2
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn dmu_dni(&self, contributions: Contributions) -> PySIArray2 {
                PySIArray2::from(self.0.dmu_dni(contributions))
            }

            /// Return logarithmic fugacity coefficient.
            ///
            /// Returns
            /// -------
            /// numpy.ndarray
            fn ln_phi<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
                self.0.ln_phi().view().to_pyarray(py)
            }

            /// Return logarithmic fugacity coefficient of all components treated as
            /// pure substance at mixture temperature and pressure.
            ///
            /// Returns
            /// -------
            /// numpy.ndarray
            fn ln_phi_pure_liquid<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
                Ok(self.0.ln_phi_pure_liquid()?.view().to_pyarray(py))
            }

            /// Return logarithmic symmetric activity coefficient.
            ///
            /// Returns
            /// -------
            /// numpy.ndarray
            fn ln_symmetric_activity_coefficient<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
                Ok(self.0.ln_symmetric_activity_coefficient()?.view().to_pyarray(py))
            }

            /// Return derivative of logarithmic fugacity coefficient w.r.t. temperature.
            ///
            /// Returns
            /// -------
            /// SIArray1
            fn dln_phi_dt(&self) -> PySIArray1 {
                PySIArray1::from(self.0.dln_phi_dt())
            }

            /// Return derivative of logarithmic fugacity coefficient w.r.t. pressure.
            ///
            /// Returns
            /// -------
            /// SIArray1
            fn dln_phi_dp(&self) -> PySIArray1 {
                PySIArray1::from(self.0.dln_phi_dp())
            }

            /// Return derivative of logarithmic fugacity coefficient w.r.t. amount of substance.
            ///
            /// Returns
            /// -------
            /// SIArray2
            fn dln_phi_dnj(&self) -> PySIArray2 {
                PySIArray2::from(self.0.dln_phi_dnj())
            }

            /// Return thermodynamic factor.
            ///
            /// Returns
            /// -------
            /// numpy.ndarray
            fn thermodynamic_factor<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
                self.0.thermodynamic_factor().view().to_pyarray(py)
            }

            /// Return molar isochoric heat capacity.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn molar_isochoric_heat_capacity(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.molar_isochoric_heat_capacity(contributions))
            }

            /// Return derivative of isochoric heat capacity w.r.t. temperature.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn dc_v_dt(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.dc_v_dt(contributions))
            }

            /// Return molar isobaric heat capacity.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn molar_isobaric_heat_capacity(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.molar_isobaric_heat_capacity(contributions))
            }

	        /// Return entropy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn entropy(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.entropy(contributions))
            }

            /// Return derivative of entropy with respect to temperature.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn ds_dt(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.ds_dt(contributions))
            }

            /// Return molar entropy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn molar_entropy(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.molar_entropy(contributions))
            }


            /// Return partial molar entropy of each component.
            ///
            /// Returns
            /// -------
            /// SIArray1
            fn partial_molar_entropy(&self) -> PySIArray1 {
                PySIArray1::from(self.0.partial_molar_entropy())
            }

            /// Return enthalpy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn enthalpy(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.enthalpy(contributions))
            }

            /// Return molar enthalpy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn molar_enthalpy(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.molar_enthalpy(contributions))
            }


            /// Return partial molar enthalpy of each component.
            ///
            /// Returns
            /// -------
            /// SIArray1
            fn partial_molar_enthalpy(&self) -> PySIArray1 {
                PySIArray1::from(self.0.partial_molar_enthalpy())
            }

            /// Return Helmholtz energy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn helmholtz_energy(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.helmholtz_energy(contributions))
            }

            /// Return molar Helmholtz energy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn molar_helmholtz_energy(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.molar_helmholtz_energy(contributions))
            }

            /// Return residual Helmholtz energy contributions.
            ///
            /// Returns
            /// -------
            /// List[Tuple[str, SINumber]]
            fn residual_helmholtz_energy_contributions(&self) -> Vec<(String, PySINumber)> {
                self.0
                    .residual_helmholtz_energy_contributions()
                    .into_iter()
                    .map(|(s, q)| (s, PySINumber::from(q)))
                    .collect()
            }

            /// Return Gibbs energy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn gibbs_energy(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.gibbs_energy(contributions))
            }

            /// Return molar Gibbs energy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn molar_gibbs_energy(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.molar_gibbs_energy(contributions))
            }


            /// Return internal energy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn internal_energy(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.internal_energy(contributions))
            }

            /// Return molar internal energy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn molar_internal_energy(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.molar_internal_energy(contributions))
            }

            /// Return Joule Thomson coefficient.
            ///
            /// Returns
            /// -------
            /// SINumber
            fn joule_thomson(&self) -> PySINumber {
                PySINumber::from(self.0.joule_thomson())
            }

            /// Return isentropy compressibility coefficient.
            ///
            /// Returns
            /// -------
            /// SINumber
            fn isentropic_compressibility(&self) -> PySINumber {
                PySINumber::from(self.0.isentropic_compressibility())
            }

            /// Return isothermal compressibility coefficient.
            ///
            /// Returns
            /// -------
            /// SINumber
            fn isothermal_compressibility(&self) -> PySINumber {
                PySINumber::from(self.0.isothermal_compressibility())
            }

            /// Return isenthalpic compressibility coefficient.
            ///
            /// Returns
            /// -------
            /// SINumber
            fn isenthalpic_compressibility(&self) -> PySINumber {
                PySINumber::from(self.0.isenthalpic_compressibility())
            }

            /// Return thermal expansivity coefficient.
            ///
            /// Returns
            /// -------
            /// SINumber
            fn thermal_expansivity(&self) -> PySINumber {
                PySINumber::from(self.0.thermal_expansivity())
            }

            /// Return Grueneisen parameter.
            ///
            /// Returns
            /// -------
            /// float
            fn grueneisen_parameter(&self) -> f64 {
                self.0.grueneisen_parameter()
            }

            /// Return structure factor.
            ///
            /// Returns
            /// -------
            /// float
            fn structure_factor(&self) -> f64 {
                self.0.structure_factor()
            }

            /// Return total molar weight.
            ///
            /// Returns
            /// -------
            /// SINumber
            fn total_molar_weight(&self) -> PySINumber {
                PySINumber::from(self.0.total_molar_weight())
            }

            /// Return speed of sound.
            ///
            /// Returns
            /// -------
            /// SINumber
            fn speed_of_sound(&self) -> PySINumber {
                PySINumber::from(self.0.speed_of_sound())
            }

            /// Returns mass of each component in the system.
            ///
            /// Returns
            /// -------
            /// SIArray1
            fn mass(&self) -> PySIArray1 {
                PySIArray1::from(self.0.mass())
            }

            /// Returns system's total mass.
            ///
            /// Returns
            /// -------
            /// SINumber
            fn total_mass(&self) -> PySINumber {
                PySINumber::from(self.0.total_mass())
            }

            /// Returns system's mass density.
            ///
            /// Returns
            /// -------
            /// SINumber
            fn mass_density(&self) -> PySINumber {
                PySINumber::from(self.0.mass_density())
            }

            /// Returns mass fractions for each component.
            ///
            /// Returns
            /// -------
            /// numpy.ndarray[Float64]
            fn massfracs<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
                self.0.massfracs().view().to_pyarray(py)
            }

            /// Return mass specific Helmholtz energy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn specific_helmholtz_energy(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.specific_helmholtz_energy(contributions))
            }

            /// Return mass specific entropy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn specific_entropy(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.specific_entropy(contributions))
            }

            /// Return mass specific internal_energy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn specific_internal_energy(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.specific_internal_energy(contributions))
            }

            /// Return mass specific gibbs_energy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn specific_gibbs_energy(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.specific_gibbs_energy(contributions))
            }

            /// Return mass specific enthalpy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn specific_enthalpy(&self, contributions: Contributions) -> PySINumber {
                PySINumber::from(self.0.specific_enthalpy(contributions))
            }

            #[getter]
            fn get_total_moles(&self) -> PySINumber {
                PySINumber::from(self.0.total_moles)
            }

            #[getter]
            fn get_temperature(&self) -> PySINumber {
                PySINumber::from(self.0.temperature)
            }

            #[getter]
            fn get_volume(&self) -> PySINumber {
                PySINumber::from(self.0.volume)
            }

            #[getter]
            fn get_density(&self) -> PySINumber {
                PySINumber::from(self.0.density)
            }

            #[getter]
            fn get_moles(&self) -> PySIArray1 {
                PySIArray1::from(self.0.moles.clone())
            }

            #[getter]
            fn get_partial_density(&self) -> PySIArray1 {
                PySIArray1::from(self.0.partial_density.clone())
            }

            #[getter]
            fn get_molefracs<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
                self.0.molefracs.view().to_pyarray(py)
            }

            fn _repr_markdown_(&self) -> String {
                if self.0.eos.components() == 1 {
                    format!(
                        "|temperature|density|\n|-|-|\n|{:.5}|{:.5}|",
                        self.0.temperature, self.0.density
                    )
                } else {
                    format!(
                        "|temperature|density|molefracs\n|-|-|-|\n|{:.5}|{:.5}|{:.5}|",
                        self.0.temperature, self.0.density, self.0.molefracs
                    )
                }
            }

            fn __repr__(&self) -> PyResult<String> {
                Ok(self.0.to_string())
            }
        }


        /// A list of states that provides convenient getters
        /// for properties of all the individual states.
        ///
        /// Parameters
        /// ----------
        /// states : [State]
        ///     A list of individual states.
        ///
        /// Returns
        /// -------
        /// StateVec
        #[pyclass(name = "StateVec")]
        pub struct PyStateVec(Vec<State<$eos>>);

        impl From<StateVec<'_, $eos>> for PyStateVec {
            fn from(vec: StateVec<$eos>) -> Self {
                Self(vec.into_iter().map(|s| s.clone()).collect())
            }
        }

        impl<'a> From<&'a PyStateVec> for StateVec<'a, $eos> {
            fn from(vec: &'a PyStateVec) -> Self {
                Self(vec.0.iter().collect())
            }
        }

        #[pymethods]
        impl PyStateVec {
            #[new]
            fn new(states: Vec<PyState>) -> Self {
                Self(states.into_iter().map(|s| s.0).collect())
            }

            fn __len__(&self) -> PyResult<usize> {
                Ok(self.0.len())
            }

            fn __getitem__(&self, idx: isize) -> PyResult<PyState> {
                let i = if idx < 0 {
                    self.0.len() as isize + idx
                } else {
                    idx
                };
                if (0..self.0.len()).contains(&(i as usize)) {
                    Ok(PyState(self.0[i as usize].clone()))
                } else {
                    Err(PyIndexError::new_err(format!("StateVec index out of range")))
                }
            }

            /// Return molar entropy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SIArray1
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn molar_entropy(&self, contributions: Contributions) -> PySIArray1 {
                StateVec::from(self).molar_entropy(contributions).into()
            }

            /// Return mass specific entropy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SIArray1
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn specific_entropy(&self, contributions: Contributions) -> PySIArray1 {
                StateVec::from(self).specific_entropy(contributions).into()
            }

            /// Return molar enthalpy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SIArray1
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn molar_enthalpy(&self, contributions: Contributions) -> PySIArray1 {
                StateVec::from(self).molar_enthalpy(contributions).into()
            }

            /// Return mass specific enthalpy.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the Helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SIArray1
            #[pyo3(signature = (contributions=Contributions::Total), text_signature = "($self, contributions)")]
            fn specific_enthalpy(&self, contributions: Contributions) -> PySIArray1 {
                StateVec::from(self).specific_enthalpy(contributions).into()
            }


            #[getter]
            fn get_temperature(&self) -> PySIArray1{
                StateVec::from(self).temperature().into()
            }

            #[getter]
            fn get_pressure(&self) -> PySIArray1 {
                StateVec::from(self).pressure().into()
            }

            #[getter]
            fn get_compressibility<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
                StateVec::from(self).compressibility().view().to_pyarray(py)
            }

            #[getter]
            fn get_density(&self) -> PySIArray1 {
                StateVec::from(self).density().into()
            }

            #[getter]
            fn get_moles<'py>(&self, py: Python<'py>) -> PySIArray2 {
                StateVec::from(self).moles().into()
            }

            #[getter]
            fn get_molefracs<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
                StateVec::from(self).molefracs().view().to_pyarray(py)
            }

            #[getter]
            fn get_mass_density(&self) -> PySIArray1 {
                StateVec::from(self).mass_density().into()
            }

            #[getter]
            fn get_massfracs<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
                StateVec::from(self).massfracs().view().to_pyarray(py)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_state_entropy_scaling {
    ($eos:ty, $py_eos:ty) => {
        #[pymethods]
        impl PyState {
            /// Return viscosity via entropy scaling.
            ///
            /// Returns
            /// -------
            /// SINumber
            fn viscosity(&self) -> PyResult<PySINumber> {
                Ok(PySINumber::from(self.0.viscosity()?))
            }

            /// Return reference viscosity for entropy scaling.
            ///
            /// Returns
            /// -------
            /// SINumber
            fn viscosity_reference(&self) -> PyResult<PySINumber> {
                Ok(PySINumber::from(self.0.viscosity_reference()?))
            }

            /// Return logarithmic reduced viscosity.
            ///
            /// This equals the viscosity correlation function
            /// as used by entropy scaling.
            ///
            /// Returns
            /// -------
            /// float
            fn ln_viscosity_reduced(&self) -> PyResult<f64> {
                Ok(self.0.ln_viscosity_reduced()?)
            }

            /// Return diffusion coefficient via entropy scaling.
            ///
            /// Returns
            /// -------
            /// SINumber
            fn diffusion(&self) -> PyResult<PySINumber> {
                Ok(PySINumber::from(self.0.diffusion()?))
            }

            /// Return reference diffusion for entropy scaling.
            ///
            /// Returns
            /// -------
            /// SINumber
            fn diffusion_reference(&self) -> PyResult<PySINumber> {
                Ok(PySINumber::from(self.0.diffusion_reference()?))
            }

            /// Return logarithmic reduced diffusion.
            ///
            /// This equals the diffusion correlation function
            /// as used by entropy scaling.
            ///
            /// Returns
            /// -------
            /// float
            fn ln_diffusion_reduced(&self) -> PyResult<f64> {
                Ok(self.0.ln_diffusion_reduced()?)
            }

            /// Return thermal conductivity via entropy scaling.
            ///
            /// Returns
            /// -------
            /// SINumber
            fn thermal_conductivity(&self) -> PyResult<PySINumber> {
                Ok(PySINumber::from(self.0.thermal_conductivity()?))
            }

            /// Return reference thermal conductivity for entropy scaling.
            ///
            /// Returns
            /// -------
            /// SINumber
            fn thermal_conductivity_reference(&self) -> PyResult<PySINumber> {
                Ok(PySINumber::from(self.0.thermal_conductivity_reference()?))
            }

            /// Return logarithmic reduced thermal conductivity.
            ///
            /// This equals the thermal conductivity correlation function
            /// as used by entropy scaling.
            ///
            /// Returns
            /// -------
            /// float
            fn ln_thermal_conductivity_reduced(&self) -> PyResult<f64> {
                Ok(self.0.ln_thermal_conductivity_reduced()?)
            }
        }
    };
}
