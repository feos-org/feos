#[macro_export]
macro_rules! impl_micelle_profile {
    ($func:ty) => {
        /// A one-dimensional profile of a spherical or cylindrical micelle.
        #[pyclass(name = "MicelleProfile", unsendable)]
        pub struct PyMicelleProfile(MicelleProfile<SIUnit, $func>);

        impl_1d_profile!(PyMicelleProfile, [get_r]);

        #[pymethods]
        impl PyMicelleProfile {
            /// Crate an initial density profile of a spherical micelle.
            ///
            /// Parameters
            /// ----------
            /// bulk: State
            ///     The bulk state in equilibrium with the micelle.
            /// n_grid: int
            ///     The number of grid points.
            /// width: SINumber
            ///     The width of the system.
            /// initialization: {(float, float), SIArray2}
            ///     Either peak and width of an external potential used to initialize
            ///     the micelle or a density profile directly.
            /// specification: (float, SINumber), optional
            ///     Excess number of surfactant molecules and pressure. If None, the
            ///     chemical potential of the system is fixed.
            ///
            /// Returns
            /// -------
            /// MicelleProfile
            ///
            #[staticmethod]
            #[pyo3(text_signature = "(bulk, n_grid, width, initialization, specification=None)")]
            fn new_spherical(
                bulk: PyState,
                n_grid: usize,
                width: PySINumber,
                initialization: &PyAny,
                specification: Option<&PyAny>,
            ) -> PyResult<Self> {
                let profile = MicelleProfile::new_spherical(
                    &bulk.0,
                    n_grid,
                    width.into(),
                    parse_micelle_initialization(initialization)?,
                    parse_micelle_specification(specification)?,
                )?;
                Ok(PyMicelleProfile(profile))
            }

            /// Crate an initial density profile of a cylindrical micelle.
            ///
            /// Parameters
            /// ----------
            /// bulk: State
            ///     The bulk state in equilibrium with the micelle.
            /// n_grid: int
            ///     The number of grid points.
            /// width: SINumber
            ///     The width of the system.
            /// initialization: {(float, float), SIArray2}
            ///     Either peak and width of an external potential used to initialize
            ///     the micelle or a density profile directly.
            /// specification: (float, SINumber), optional
            ///     Excess number of surfactant molecules and pressure. If None, the
            ///     chemical potential of the system is fixed.
            ///
            /// Returns
            /// -------
            /// MicelleProfile
            ///
            #[staticmethod]
            #[pyo3(text_signature = "(bulk, n_grid, width, initialization, specification=None)")]
            fn new_cylindrical(
                bulk: PyState,
                n_grid: usize,
                width: PySINumber,
                initialization: &PyAny,
                specification: Option<&PyAny>,
            ) -> PyResult<Self> {
                let profile = MicelleProfile::new_cylindrical(
                    &bulk.0,
                    n_grid,
                    width.into(),
                    parse_micelle_initialization(initialization)?,
                    parse_micelle_specification(specification)?,
                )?;
                Ok(PyMicelleProfile(profile))
            }

            /// Create a new micelle profile with a given specification.
            ///
            /// Parameters
            /// ----------
            /// delta_n_surfactant: float
            ///     Excess number of surfactant molecules.
            /// pressure: SINumber
            ///     Pressure.
            ///
            #[pyo3(text_signature = "(delta_n_surfactant, pressure)")]
            fn update_specification(&self, delta_n_surfactant: f64, pressure: PySINumber) -> Self {
                Self(self.0.update_specification(MicelleSpecification::Size {
                        delta_n_surfactant,
                        pressure: pressure.into(),
                    }),
                )
            }

            /// Solve the micelle profile in-place. The first solver is used to solve
            /// the initial problem including the external potential. After the external
            /// potential is cleared, the second solver is used to calculate the result.
            ///
            /// Parameters
            /// ----------
            /// solver1 : DFTSolver, optional
            ///     The first solver used to solve the profile.
            /// solver2 : DFTSolver, optional
            ///     The second solver used to solve the profile.
            /// debug: bool, optional
            ///     If True, do not check for convergence.
            ///
            /// Returns
            /// -------
            /// MicelleProfile
            ///
            #[pyo3(text_signature = "(solver1=None, solver2=None, debug=False)")]
            #[args(solver1 = "None", solver2 = "None", debug = "false")]
            fn solve_micelle(
                slf: &PyCell<Self>,
                solver1: Option<PyDFTSolver>,
                solver2: Option<PyDFTSolver>,
                debug: bool,
            ) -> PyResult<&PyCell<Self>> {
                slf.borrow_mut().0.solve_micelle_inplace(
                    solver1.map(|s| s.0).as_ref(),
                    solver2.map(|s| s.0).as_ref(),
                    debug,
                )?;
                Ok(slf)
            }
        }

        #[pymethods]
        impl PyMicelleProfile {
            #[getter]
            fn get_delta_omega(&self) -> Option<PySINumber> {
                self.0.delta_omega.map(PySINumber::from)
            }

            #[getter]
            fn get_delta_n(&self) -> Option<PySIArray1> {
                self.0.delta_n.clone().map(PySIArray1::from)
            }

            /// Use the converged micelle to calculate the critical micelle for the given
            /// temperature and pressure.
            ///
            /// Parameters
            /// ----------
            /// solver : DFTSolver, optional
            ///     The solver used to solve the profile.
            /// max_iter : int, optional
            ///     The maximum number of iterations of the Newton solver.
            /// tol: float, optional
            ///     The tolerance of the Newton solver.
            /// verbosity: Verbosity, optional
            ///     The verbosity of the Newton solver.
            ///
            /// Returns
            /// -------
            /// MicelleProfileResult
            ///
            #[pyo3(text_signature = "(solver=None, max_iter=None, tol=None, verbosity=None)")]
            fn critical_micelle(
                &self,
                solver: Option<PyDFTSolver>,
                max_iter: Option<usize>,
                tol: Option<f64>,
                verbosity: Option<Verbosity>,
            ) -> PyResult<Self> {
                Ok(Self(self.0.clone().critical_micelle(
                        solver.map(|s| s.0).as_ref(),
                        (max_iter, tol, verbosity).into(),
                    )?,
                ))
            }
        }

        pub fn parse_micelle_initialization(
            initialization: &PyAny,
        ) -> PyResult<MicelleInitialization<SIUnit>> {
            if let Ok((peak, width)) = initialization.extract::<(f64, f64)>() {
                Ok(MicelleInitialization::ExternalPotential(peak, width))
            } else if let Ok(density) = initialization.extract::<PySIArray2>() {
                Ok(MicelleInitialization::Density(density.into()))
            } else {
                Err(PyErr::new::<PyValueError, _>(format!(
                    "`initialization` must be (peak, width) or an SIArray2 containing the initial densities."
                )))
            }
        }

        pub fn parse_micelle_specification(
            specification: Option<&PyAny>,
        ) -> PyResult<MicelleSpecification<SIUnit>> {
            match specification {
                Some(specification) => {
                    if let Ok((delta_n_surfactant, pressure)) = specification.extract::<(f64, PySINumber)>()
                    {
                        Ok(MicelleSpecification::Size {
                            delta_n_surfactant,
                            pressure: pressure.into(),
                        })
                    } else {
                        Err(PyErr::new::<PyValueError, _>(format!(
                            "`specification` must be (delta_n_surfactant, pressure) or None."
                        )))
                    }
                }
                None => Ok(MicelleSpecification::ChemicalPotential),
            }
        }
    };
}
