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
        /// solver: DFTSolver, optional
        ///     Custom solver options
        ///
        /// Returns
        /// -------
        /// SurfaceTensionDiagram
        ///
        #[pyclass(name = "SurfaceTensionDiagram", unsendable)]
        #[pyo3(text_signature = "(dia, init_densities=None, n_grid=None, l_grid=None, critical_temperature=None, solver=None)")]
        pub struct PySurfaceTensionDiagram(SurfaceTensionDiagram<SIUnit, $func>);

        #[pymethods]
        impl PySurfaceTensionDiagram {
            #[new]
            pub fn isotherm(
                dia: Vec<PyPhaseEquilibrium>,
                init_densities: Option<bool>,
                n_grid: Option<usize>,
                l_grid: Option<PySINumber>,
                critical_temperature: Option<PySINumber>,
                solver: Option<PyDFTSolver>,
            ) -> Self {
                let x = dia.into_iter().map(|vle| vle.0).collect();
                Self(SurfaceTensionDiagram::new(
                    &x,
                    init_densities,
                    n_grid,
                    l_grid.map(|l| l.into()),
                    critical_temperature.map(|c| c.into()),
                    solver.map(|s| s.0).as_ref(),
                ))
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
        }
    };
}
