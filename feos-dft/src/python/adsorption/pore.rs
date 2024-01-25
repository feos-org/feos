#[macro_export]
macro_rules! impl_pore {
    ($func:ty, $py_func:ty) => {
        /// Parameters required to specify a 1D pore.
        ///
        /// Parameters
        /// ----------
        /// geometry : Geometry
        ///     The pore geometry.
        /// pore_size : SINumber
        ///     The width of the slit pore in cartesian coordinates,
        ///     or the pore radius in spherical and cylindrical coordinates.
        /// potential : ExternalPotential
        ///     The potential used to model wall-fluid interactions.
        /// n_grid : int, optional
        ///     The number of grid points.
        /// potential_cutoff : float, optional
        ///     Maximum value for the external potential.
        ///
        /// Returns
        /// -------
        /// Pore1D
        ///
        #[pyclass(name = "Pore1D")]
        pub struct PyPore1D(Pore1D);

        #[pyclass(name = "PoreProfile1D")]
        pub struct PyPoreProfile1D(PoreProfile1D<$func>);

        impl_1d_profile!(PyPoreProfile1D, [get_r, get_z]);
        impl_pore_profile!(PyPoreProfile1D);

        #[pymethods]
        impl PyPore1D {
            #[new]
            #[pyo3(text_signature = "(geometry, pore_size, potential, n_grid=None, potential_cutoff=None)")]
            #[pyo3(signature = (geometry, pore_size, potential, n_grid=None, potential_cutoff=None))]
            fn new(
                geometry: Geometry,
                pore_size: Length,
                potential: PyExternalPotential,
                n_grid: Option<usize>,
                potential_cutoff: Option<f64>,
            ) -> PyResult<Self> {
                Ok(Self(Pore1D::new(
                    geometry,
                    pore_size,
                    potential.0,
                    n_grid,
                    potential_cutoff,
                )))
            }

            /// Initialize the pore for the given bulk state.
            ///
            /// Parameters
            /// ----------
            /// bulk : State
            ///     The bulk state in equilibrium with the pore.
            /// density : SIArray2, optional
            ///     Initial values for the density profile.
            /// external_potential : numpy.ndarray[float], optional
            ///     The external potential in the pore. Used to
            ///     save computation time in the case of costly
            ///     evaluations of external potentials.
            ///
            /// Returns
            /// -------
            /// PoreProfile1D
            #[pyo3(text_signature = "($self, bulk, density=None, external_potential=None)")]
            #[pyo3(signature = (bulk, density=None, external_potential=None))]
            fn initialize(
                &self,
                bulk: &PyState,
                density: Option<Density<Array2<f64>>>,
                external_potential: Option<&Bound<'_ ,PyArray2<f64>>>,
            ) -> PyResult<PyPoreProfile1D> {
                Ok(PyPoreProfile1D(self.0.initialize(
                    &bulk.0,
                    density.map(|d| d.try_into()).transpose()?.as_ref(),
                    external_potential.map(|e| e.to_owned_array()).as_ref(),
                )?))
            }

            #[getter]
            fn get_geometry(&self)-> Geometry {
                self.0.geometry
            }

            #[getter]
            fn get_pore_size(&self)-> Length {
                self.0.pore_size.into()
            }

            #[getter]
            fn get_potential(&self)-> PyExternalPotential {
                PyExternalPotential(self.0.potential.clone())
            }

            #[getter]
            fn get_n_grid(&self)-> Option<usize> {
                self.0.n_grid
            }

            #[getter]
            fn get_potential_cutoff(&self)-> Option<f64> {
                self.0.potential_cutoff
            }

            /// The pore volume using Helium at 298 K as reference.
            #[getter]
            fn get_pore_volume(&self) -> PyResult<Volume> {
                Ok(self.0.pore_volume()?.into())
            }
        }

        #[pyclass(name = "Pore2D")]
        pub struct PyPore2D(Pore2D);

        #[pyclass(name = "PoreProfile2D")]
        pub struct PyPoreProfile2D(PoreProfile2D<$func>);

        impl_2d_profile!(PyPoreProfile2D, get_x, get_y);
        impl_pore_profile!(PyPoreProfile2D);

        #[pymethods]
        impl PyPore2D {
            #[new]
            #[pyo3(text_signature = "(system_size, angle, n_grid)")]
            fn new(
                system_size: [Length; 2],
                angle: Angle,
                n_grid: [usize; 2],
            ) -> PyResult<Self> {
                Ok(Self(Pore2D::new(
                    [system_size[0], system_size[1]],
                    angle,
                    n_grid,
                )))
            }

            /// Initialize the pore for the given bulk state.
            ///
            /// Parameters
            /// ----------
            /// bulk : State
            ///     The bulk state in equilibrium with the pore.
            /// density : SIArray3, optional
            ///     Initial values for the density profile.
            /// external_potential : numpy.ndarray[float], optional
            ///     The external potential in the pore. Used to
            ///     save computation time in the case of costly
            ///     evaluations of external potentials.
            ///
            /// Returns
            /// -------
            /// PoreProfile2D
            #[pyo3(text_signature = "($self, bulk, density=None, external_potential=None)")]
            #[pyo3(signature = (bulk, density=None, external_potential=None))]
            fn initialize(
                &self,
                bulk: &PyState,
                density: Option<Density<Array3<f64>>>,
                external_potential: Option<&Bound<'_, PyArray3<f64>>>,
            ) -> PyResult<PyPoreProfile2D> {
                Ok(PyPoreProfile2D(self.0.initialize(
                    &bulk.0,
                    density.map(|d| d.try_into()).transpose()?.as_ref(),
                    external_potential.map(|e| e.to_owned_array()).as_ref(),
                )?))
            }

            /// The pore volume using Helium at 298 K as reference.
            #[getter]
            fn get_pore_volume(&self) -> PyResult<Volume> {
                Ok(self.0.pore_volume()?)
            }
        }

        /// Parameters required to specify a 3D pore.
        ///
        /// Parameters
        /// ----------
        /// system_size : [SINumber; 3]
        ///     The size of the unit cell.
        /// n_grid : [int; 3]
        ///     The number of grid points in each direction.
        /// coordinates : numpy.ndarray[float]
        ///     The positions of all interaction sites in the solid.
        /// sigma_ss : numpy.ndarray[float]
        ///     The size parameters of all interaction sites.
        /// epsilon_k_ss : numpy.ndarray[float]
        ///     The energy parameter of all interaction sites.
        /// angles : [Angle; 3], optional
        ///     The angles of the unit cell or `None` if the unit cell
        ///     is orthorombic
        /// potential_cutoff: float, optional
        ///     Maximum value for the external potential.
        /// cutoff_radius: SINumber, optional
        ///     The cutoff radius for the calculation of solid-fluid interactions.
        ///
        /// Returns
        /// -------
        /// Pore3D
        ///
        #[pyclass(name = "Pore3D")]
        pub struct PyPore3D(Pore3D);

        #[pyclass(name = "PoreProfile3D")]
        pub struct PyPoreProfile3D(PoreProfile3D<$func>);

        impl_3d_profile!(PyPoreProfile3D, get_x, get_y, get_z);
        impl_pore_profile!(PyPoreProfile3D);

        #[pymethods]
        impl PyPore3D {
            #[new]
            #[pyo3(text_signature = "(system_size, n_grid, coordinates, sigma_ss, epsilon_k_ss, angles=None, potential_cutoff=None, cutoff_radius=None)")]
            #[pyo3(signature = (system_size, n_grid, coordinates, sigma_ss, epsilon_k_ss, angles=None, potential_cutoff=None, cutoff_radius=None))]
            #[expect(clippy::too_many_arguments)]
            fn new(
                system_size: [Length; 3],
                n_grid: [usize; 3],
                coordinates: Length<Array2<f64>>,
                sigma_ss: &Bound<'_, PyArray1<f64>>,
                epsilon_k_ss: &Bound<'_, PyArray1<f64>>,
                angles: Option<[Angle; 3]>,
                potential_cutoff: Option<f64>,
                cutoff_radius: Option<Length>,
            ) -> Self {
                Self(Pore3D::new(
                    system_size,
                    n_grid,
                    coordinates,
                    sigma_ss.to_owned_array(),
                    epsilon_k_ss.to_owned_array(),
                    angles,
                    potential_cutoff,
                    cutoff_radius,
                ))
            }

            /// Initialize the pore for the given bulk state.
            ///
            /// Parameters
            /// ----------
            /// bulk : State
            ///     The bulk state in equilibrium with the pore.
            /// density : SIArray4, optional
            ///     Initial values for the density profile.
            /// external_potential : numpy.ndarray[float], optional
            ///     The external potential in the pore. Used to
            ///     save computation time in the case of costly
            ///     evaluations of external potentials.
            ///
            /// Returns
            /// -------
            /// PoreProfile3D
            #[pyo3(text_signature = "($self, bulk, density=None, external_potential=None)")]
            #[pyo3(signature = (bulk, density=None, external_potential=None))]
            fn initialize(
                &self,
                bulk: &PyState,
                density: Option<Density<Array4<f64>>>,
                external_potential: Option<&Bound<'_, PyArray4<f64>>>,
            ) -> PyResult<PyPoreProfile3D> {
                Ok(PyPoreProfile3D(self.0.initialize(
                    &bulk.0,
                    density.map(|d| d.try_into()).transpose()?.as_ref(),
                    external_potential.map(|e| e.to_owned_array()).as_ref(),
                )?))
            }

            /// The pore volume using Helium at 298 K as reference.
            #[getter]
            fn get_pore_volume(&self) -> PyResult<Volume> {
                Ok(self.0.pore_volume()?.into())
            }
        }
    };
}

#[macro_export]
macro_rules! impl_pore_profile {
    ($py_profile:ty) => {
        #[pymethods]
        impl $py_profile {
            #[getter]
            fn get_grand_potential(&self) -> Option<Energy> {
                self.0.grand_potential
            }

            #[getter]
            fn get_interfacial_tension(&self) -> Option<Energy> {
                self.0.interfacial_tension
            }

            #[getter]
            fn get_partial_molar_enthalpy_of_adsorption(
                &self,
            ) -> PyResult<MolarEnergy<Array1<f64>>> {
                Ok(self.0.partial_molar_enthalpy_of_adsorption()?)
            }

            #[getter]
            fn get_enthalpy_of_adsorption(&self) -> PyResult<MolarEnergy> {
                Ok(self.0.enthalpy_of_adsorption()?)
            }

            #[getter]
            fn get_henry_coefficient(&self) -> PyResult<PySIArray1> {
                Ok(self.0.henry_coefficient()?.into())
            }

            #[getter]
            fn get_ideal_gas_enthalpy_of_adsorption(&self) -> PyResult<PySIArray1> {
                Ok(self.0.ideal_gas_enthalpy_of_adsorption()?.into())
            }
        }
    };
}
