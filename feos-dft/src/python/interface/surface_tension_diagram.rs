#[macro_export]
macro_rules! impl_surface_tension_diagram {
    ($func:ty) => {
        /// Container structure for the efficient calculation of surface tension diagrams.
        ///
        /// Parameters
        /// ----------
        /// dia : [PhaseEquilibrium]
        ///     The underlying phase diagram given as a list of states
        ///     for which surface tensions shall be calculated.
        /// init_densities : bool, optional
        ///     None: Do not initialize densities with old results
        ///     True: Initialize and scale densities
        ///     False: Initialize without scaling
        /// n_grid : int, optional
        ///     The number of grid points (default: 2048).
        /// l_grid : SINumber, optional
        ///     The size of the calculation domain (default: 100 A)
        /// critical_temperature: SINumber, optional
        ///     An estimate for the critical temperature, used to initialize
        ///     density profile (default: 500 K)
        /// fix_equimolar_surface: bool, optional
        ///     If True use additional constraints to fix the
        ///     equimolar surface of the system.
        ///     Defaults to False.
        /// solver: DFTSolver, optional
        ///     Custom solver options
        ///
        /// Returns
        /// -------
        /// SurfaceTensionDiagram
        ///
        #[pyclass(name = "SurfaceTensionDiagram")]
        #[pyo3(text_signature = "(dia, init_densities=None, n_grid=None, l_grid=None, critical_temperature=None, fix_equimolar_surface=None, solver=None)")]
        pub struct PySurfaceTensionDiagram(SurfaceTensionDiagram<$func>);

        #[pymethods]
        impl PySurfaceTensionDiagram {
            #[new]
            pub fn isotherm(
                dia: Vec<PyPhaseEquilibrium>,
                init_densities: Option<bool>,
                n_grid: Option<usize>,
                l_grid: Option<PySINumber>,
                critical_temperature: Option<PySINumber>,
                fix_equimolar_surface: Option<bool>,
                solver: Option<PyDFTSolver>,
            ) -> PyResult<Self> {
                let x = dia.into_iter().map(|vle| vle.0).collect();
                Ok(Self(SurfaceTensionDiagram::new(
                    &x,
                    init_densities,
                    n_grid,
                    l_grid.map(|l| l.try_into()).transpose()?,
                    critical_temperature.map(|c| c.try_into()).transpose()?,
                    fix_equimolar_surface,
                    solver.map(|s| s.0).as_ref(),
                )))
            }

            #[getter]
            fn get_profiles(&self) -> Vec<PyPlanarInterface> {
                self.0
                    .profiles
                    .iter()
                    .map(|p| PyPlanarInterface(p.clone()))
                    .collect()
            }

            #[getter]
            pub fn get_vapor(&self) -> PyStateVec {
                self.0.vapor().into()
            }

            #[getter]
            pub fn get_liquid(&self) -> PyStateVec {
                self.0.liquid().into()
            }

            #[getter]
            pub fn get_surface_tension(&mut self) -> PySIArray1 {
                self.0.surface_tension().into()
            }

            #[getter]
            pub fn get_relative_adsorption(&self) -> Vec<PySIArray2> {
                self.0.relative_adsorption().iter().cloned().map(|i| i.into()).collect()
            }

            #[getter]
            pub fn get_interfacial_enrichment<'py>(&self, py: Python<'py>) -> Vec<&'py PyArray1<f64>> {
                self.0.interfacial_enrichment().iter().map(|i| i.to_pyarray(py)).collect()
            }

            #[getter]
            pub fn interfacial_thickness(&self) -> PySIArray1 {
                self.0.interfacial_thickness().into()
            }
        }
    };
}
