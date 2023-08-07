mod external_potential;
mod pore;

pub use external_potential::PyExternalPotential;

#[macro_export]
macro_rules! impl_adsorption {
    ($func:ty, $py_func:ty) => {
        /// Container structure for adsorption isotherms in 1D pores.
        #[pyclass(name = "Adsorption1D")]
        pub struct PyAdsorption1D(Adsorption1D<$func>);

        /// Container structure for adsorption isotherms in 3D pores.
        #[pyclass(name = "Adsorption3D")]
        pub struct PyAdsorption3D(Adsorption3D<$func>);

        impl_adsorption_isotherm!($func, $py_func, PyAdsorption1D, PyPore1D, PyPoreProfile1D);
        impl_adsorption_isotherm!($func, $py_func, PyAdsorption3D, PyPore3D, PyPoreProfile3D);
    };
}

#[macro_export]
macro_rules! impl_adsorption_isotherm {
    ($func:ty, $py_func:ty, $py_adsorption:ty, $py_pore:ty, $py_pore_profile:ident) => {
        #[pymethods]
        impl $py_adsorption {
            /// Calculate an adsorption isotherm for the given pressure range.
            /// The profiles are evaluated starting from the lowest pressure.
            /// The resulting density profiles can be metastable.
            ///
            /// Parameters
            /// ----------
            /// functional : HelmholtzEnergyFunctional
            ///     The Helmholtz energy functional.
            /// temperature : SINumber
            ///     The temperature.
            /// pressure : SIArray1
            ///     The pressures for which the profiles are calculated.
            /// pore : Pore
            ///     The pore parameters.
            /// molefracs: numpy.ndarray[float], optional
            ///     For a mixture, the molefracs of the bulk system.
            /// solver: DFTSolver, optional
            ///     Custom solver options.
            ///
            /// Returns
            /// -------
            /// Adsorption
            ///
            #[staticmethod]
            #[pyo3(text_signature = "(functional, temperature, pressure, pore, molefracs=None, solver=None)")]
            pub fn adsorption_isotherm(
                functional: &$py_func,
                temperature: PySINumber,
                pressure: PySIArray1,
                pore: &$py_pore,
                molefracs: Option<&PyArray1<f64>>,
                solver: Option<PyDFTSolver>,
            ) -> PyResult<Self> {
                Ok(Self(Adsorption::adsorption_isotherm(
                    &functional.0,
                    temperature.try_into()?,
                    &pressure.try_into()?,
                    &pore.0,
                    molefracs.map(|x| x.to_owned_array()).as_ref(),
                    solver.map(|s| s.0).as_ref(),
                )?))
            }

            /// Calculate a desorption isotherm for the given pressure range.
            /// The profiles are evaluated starting from the highest pressure.
            /// The resulting density profiles can be metastable.
            ///
            /// Parameters
            /// ----------
            /// functional : HelmholtzEnergyFunctional
            ///     The Helmholtz energy functional.
            /// temperature : SINumber
            ///     The temperature.
            /// pressure : SIArray1
            ///     The pressures for which the profiles are calculated.
            /// pore : Pore
            ///     The pore parameters.
            /// molefracs: numpy.ndarray[float], optional
            ///     For a mixture, the molefracs of the bulk system.
            /// solver: DFTSolver, optional
            ///     Custom solver options.
            ///
            /// Returns
            /// -------
            /// Adsorption
            ///
            #[staticmethod]
            #[pyo3(text_signature = "(functional, temperature, pressure, pore, molefracs=None, solver=None)")]
            pub fn desorption_isotherm(
                functional: &$py_func,
                temperature: PySINumber,
                pressure: PySIArray1,
                pore: &$py_pore,
                molefracs: Option<&PyArray1<f64>>,
                solver: Option<PyDFTSolver>,
            ) -> PyResult<Self> {
                Ok(Self(Adsorption::desorption_isotherm(
                    &functional.0,
                    temperature.try_into()?,
                    &pressure.try_into()?,
                    &pore.0,
                    molefracs.map(|x| x.to_owned_array()).as_ref(),
                    solver.map(|s| s.0).as_ref(),
                )?))
            }

            /// Calculate an equilibrium isotherm for the given pressure range.
            /// A phase equilibrium in the pore is calculated to determine the
            /// stable phases for every pressure. If no phase equilibrium can be
            /// calculated, the isotherm is calculated twice, one in the adsorption
            /// direction and once in the desorption direction to determine the
            /// stability of the profiles.
            ///
            /// Parameters
            /// ----------
            /// functional : HelmholtzEnergyFunctional
            ///     The Helmholtz energy functional.
            /// temperature : SINumber
            ///     The temperature.
            /// pressure : SIArray1
            ///     The pressures for which the profiles are calculated.
            /// pore : Pore
            ///     The pore parameters.
            /// molefracs: numpy.ndarray[float], optional
            ///     For a mixture, the molefracs of the bulk system.
            /// solver: DFTSolver, optional
            ///     Custom solver options.
            ///
            /// Returns
            /// -------
            /// Adsorption
            ///
            #[staticmethod]
            #[pyo3(text_signature = "(functional, temperature, pressure, pore, molefracs=None, solver=None)")]
            pub fn equilibrium_isotherm(
                functional: &$py_func,
                temperature: PySINumber,
                pressure: PySIArray1,
                pore: &$py_pore,
                molefracs: Option<&PyArray1<f64>>,
                solver: Option<PyDFTSolver>,
            ) -> PyResult<Self> {
                Ok(Self(Adsorption::equilibrium_isotherm(
                    &functional.0,
                    temperature.try_into()?,
                    &pressure.try_into()?,
                    &pore.0,
                    molefracs.map(|x| x.to_owned_array()).as_ref(),
                    solver.map(|s| s.0).as_ref(),
                )?))
            }

            /// Calculate a phase equilibrium in a pore.
            ///
            /// Parameters
            /// ----------
            /// functional : HelmholtzEnergyFunctional
            ///     The Helmholtz energy functional.
            /// temperature : SINumber
            ///     The temperature.
            /// p_min : SINumber
            ///     A suitable lower limit for the pressure.
            /// p_max : SINumber
            ///     A suitable upper limit for the pressure.
            /// pore : Pore
            ///     The pore parameters.
            /// molefracs: numpy.ndarray[float], optional
            ///     For a mixture, the molefracs of the bulk system.
            /// solver: DFTSolver, optional
            ///     Custom solver options.
            /// max_iter : int, optional
            ///     The maximum number of iterations of the phase equilibrium calculation.
            /// tol: float, optional
            ///     The tolerance of the phase equilibrium calculation.
            /// verbosity: Verbosity, optional
            ///     The verbosity of the phase equilibrium calculation.
            ///
            /// Returns
            /// -------
            /// Adsorption
            ///
            #[staticmethod]
            #[pyo3(text_signature = "(functional, temperature, p_min, p_max, pore, molefracs=None, solver=None, max_iter=None, tol=None, verbosity=None)")]
            pub fn phase_equilibrium(
                functional: &$py_func,
                temperature: PySINumber,
                p_min: PySINumber,
                p_max: PySINumber,
                pore: &$py_pore,
                molefracs: Option<&PyArray1<f64>>,
                solver: Option<PyDFTSolver>,
                max_iter: Option<usize>,
                tol: Option<f64>,
                verbosity: Option<Verbosity>,
            ) -> PyResult<Self> {
                Ok(Self(Adsorption::phase_equilibrium(
                    &functional.0,
                    temperature.try_into()?,
                    p_min.try_into()?,
                    p_max.try_into()?,
                    &pore.0,
                    molefracs.map(|x| x.to_owned_array()).as_ref(),
                    solver.map(|s| s.0).as_ref(),
                    (max_iter, tol, verbosity).into(),
                )?))
            }

            #[getter]
            fn get_profiles(&self) -> Vec<$py_pore_profile> {
                self.0
                    .profiles
                    .iter()
                    .filter_map(|p| {
                        p.as_ref()
                            .ok()
                            .map(|p| $py_pore_profile(p.clone()))
                    })
                    .collect()
            }

            #[getter]
            fn get_pressure(&self) -> PySIArray1 {
                self.0.pressure().into()
            }

            #[getter]
            fn get_adsorption(&self) -> PySIArray2 {
                self.0.adsorption().into()
            }

            #[getter]
            fn get_total_adsorption(&self) -> PySIArray1 {
                self.0.total_adsorption().into()
            }

            #[getter]
            fn get_grand_potential(&mut self) -> PySIArray1 {
                self.0.grand_potential().into()
            }

            #[getter]
            fn get_partial_molar_enthalpy_of_adsorption(&self) -> PySIArray2 {
                self.0.partial_molar_enthalpy_of_adsorption().into()
            }

            #[getter]
            fn get_enthalpy_of_adsorption(&self) -> PySIArray1 {
                self.0.enthalpy_of_adsorption().into()
            }
        }
    };
}
