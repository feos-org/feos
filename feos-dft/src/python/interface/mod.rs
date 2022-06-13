mod surface_tension_diagram;

#[macro_export]
macro_rules! impl_planar_interface {
    ($func:ty) => {
        /// A one-dimensional density profile of a vapor-liquid or liquid-liquid interface.
        #[pyclass(name = "PlanarInterface", unsendable)]
        pub struct PyPlanarInterface(PlanarInterface<SIUnit, $func>);

        impl_1d_profile!(PyPlanarInterface, [get_z]);

        #[pymethods]
        impl PyPlanarInterface {
            /// Initialize a planar interface with a hyperbolic tangent.
            ///
            /// Parameters
            /// ----------
            /// vle : PhaseEquilibrium
            ///     The bulk phase equilibrium.
            /// n_grid : int
            ///     The number of grid points.
            /// l_grid: SINumber
            ///     The width of the calculation domain.
            /// critical_temperature: SINumber
            ///     An estimate for the critical temperature of the system.
            ///     Used to guess the width of the interface.
            ///
            /// Returns
            /// -------
            /// PlanarInterface
            ///
            #[staticmethod]
            #[pyo3(text_signature = "(vle, n_grid, l_grid, critical_temperature)")]
            fn from_tanh(
                vle: &PyPhaseEquilibrium,
                n_grid: usize,
                l_grid: PySINumber,
                critical_temperature: PySINumber,
            ) -> PyResult<Self> {
                let profile = PlanarInterface::from_tanh(
                    &vle.0,
                    n_grid,
                    l_grid.into(),
                    critical_temperature.into(),
                )?;
                Ok(PyPlanarInterface(profile))
            }

            /// Initialize a planar interface with a pDGT calculation.
            ///
            /// Parameters
            /// ----------
            /// vle : PhaseEquilibrium
            ///     The bulk phase equilibrium.
            /// n_grid : int
            ///     The number of grid points.
            ///
            /// Returns
            /// -------
            /// PlanarInterface
            ///
            #[staticmethod]
            #[pyo3(text_signature = "(vle, n_grid)")]
            fn from_pdgt(vle: &PyPhaseEquilibrium, n_grid: usize) -> PyResult<Self> {
                let profile = PlanarInterface::from_pdgt(&vle.0, n_grid)?;
                Ok(PyPlanarInterface(profile))
            }

            /// Initialize a planar interface with a provided density profile.
            ///
            /// Parameters
            /// ----------
            /// vle : PhaseEquilibrium
            ///     The bulk phase equilibrium.
            /// n_grid : int
            ///     The number of grid points.
            /// l_grid: SINumber
            ///     The width of the calculation domain.
            /// density_profile: SIArray2
            ///     Initial condition for the density profile iterations
            ///
            /// Returns
            /// -------
            /// PlanarInterface
            ///
            #[staticmethod]
            #[pyo3(text_signature = "(vle, n_grid, l_grid, density_profile)")]
            fn from_density_profile(
                vle: &PyPhaseEquilibrium,
                n_grid: usize,
                l_grid: PySINumber,
                density_profile: PySIArray2,
            ) -> PyResult<Self> {
                let mut profile = PlanarInterface::new(&vle.0, n_grid, l_grid.into())?;
                profile.profile.density = density_profile.into();
                Ok(PyPlanarInterface(profile))
            }
        }

        #[pymethods]
        impl PyPlanarInterface {
            #[getter]
            fn get_surface_tension(&mut self) -> Option<PySINumber> {
                self.0.surface_tension.map(PySINumber::from)
            }

            #[getter]
            fn get_equimolar_radius(&mut self) -> Option<PySINumber> {
                self.0.equimolar_radius.map(PySINumber::from)
            }

            #[getter]
            fn get_vle(&self) -> PyPhaseEquilibrium {
                PyPhaseEquilibrium(self.0.vle.clone())
            }
        }
    };
}
