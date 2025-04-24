macro_rules! impl_profile {
    ($struct:ident, $arr:ident, $arr2:ident, $si_arr:ident, $si_arr2:ident, $py_arr2:ident, [$([$ind:expr, $ax:ident]),+]$(, $si_arr3:ident)?) => {
        #[pymethods]
        impl $struct {
            /// Calculate the residual for the given profile.
            ///
            /// Parameters
            /// ----------
            /// log: bool, optional
            ///     calculate the logarithmic residual (default: False).
            ///
            /// Returns
            /// -------
            /// (numpy.ndarray[float], numpy.ndarray[float])
            ///
            #[pyo3(signature = (log=false), text_signature = "($self, log=False)")]
            fn residual<'py>(
                &self,
                log: bool,
                py: Python<'py>,
            ) -> PyResult<(Bound<'py, $arr2<f64>>, Bound<'py, PyArray1<f64>>, f64)> {
                let (res_rho, res_mu, res_norm) = self.0.profile.residual(log).map_err(PyFeosError::from)?;
                Ok((res_rho.view().to_pyarray(py), res_mu.view().to_pyarray(py), res_norm))
            }

            /// Solve the profile in-place. A non-default solver can be provided
            /// optionally.
            ///
            /// Parameters
            /// ----------
            /// solver : DFTSolver, optional
            ///     The solver used to solve the profile.
            /// debug: bool, optional
            ///     If True, do not check for convergence.
            ///
            /// Returns
            /// -------
            /// $struct
            ///
            #[pyo3(signature = (solver=None, debug=false), text_signature = "($self, solver=None, debug=False)")]
            fn solve(slf: Bound<'_, Self>, solver: Option<PyDFTSolver>, debug: bool) -> PyResult<Bound<'_, Self>> {
                slf.borrow_mut()
                    .0
                    .solve_inplace(solver.map(|s| s.0).as_ref(), debug).map_err(PyFeosError::from)?;
                Ok(slf)
            }

            $(
            #[getter]
            fn $ax(&self) -> Length<Array1<f64>> {
                Length::from_reduced(self.0.profile.grid.grids()[$ind].clone())
            })+

            #[getter]
            fn get_temperature(&self) -> Temperature {
                self.0.profile.temperature
            }

            #[getter]
            fn get_density(&self) -> Density<$si_arr2<f64>> {
                self.0.profile.density.clone()
            }

            #[getter]
            fn get_moles(&self) -> Moles<Array1<f64>> {
                self.0.profile.moles()
            }

            #[getter]
            fn get_total_moles(&self) -> Moles {
                self.0.profile.total_moles()
            }

            #[getter]
            fn get_external_potential<'py>(&self, py: Python<'py>) -> Bound<'py, $py_arr2<f64>> {
                self.0.profile.external_potential.view().to_pyarray(py)
            }

            #[getter]
            fn get_bulk(&self) -> PyState {
                PyState(self.0.profile.bulk.clone())
            }

            #[getter]
            fn get_solver_log(&self) -> Option<PyDFTSolverLog> {
                self.0.profile.solver_log.clone().map(PyDFTSolverLog)
            }

            #[getter]
            fn get_weighted_densities<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Vec<Bound<'py, $arr2<f64>>>> {
                let n = self.0.profile.weighted_densities().map_err(PyFeosError::from)?;
                Ok(n.into_iter().map(|n| n.view().to_pyarray(py)).collect())
            }

            #[getter]
            fn get_functional_derivative<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, $arr2<f64>>> {
                Ok(self.0.profile.functional_derivative().map_err(PyFeosError::from)?.view().to_pyarray(py))
            }

            /// Calculate the entropy density of the inhomogeneous system.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SIArray
            #[pyo3(signature = (contributions=PyContributions::Total), text_signature = "($self, contributions)")]
            fn entropy_density(
                &mut self,
                contributions: PyContributions,
            ) -> PyResult<typenum::Quot<Entropy<$si_arr<f64>>, Volume>> {
                Ok(self.0.profile.entropy_density(contributions.into()).map_err(PyFeosError::from)?)
            }

            /// Calculate the entropy of the inhomogeneous system.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=PyContributions::Total), text_signature = "($self, contributions)")]
            fn entropy(
                &mut self,
                contributions: PyContributions,
            ) -> PyResult<Entropy> {
                Ok(self.0.profile.entropy(contributions.into()).map_err(PyFeosError::from)?)
            }

            /// Calculate the internal energy of the inhomogeneous system.
            ///
            /// Parameters
            /// ----------
            /// contributions: Contributions, optional
            ///     the contributions of the helmholtz energy.
            ///     Defaults to Contributions.Total.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(signature = (contributions=PyContributions::Total), text_signature = "($self, contributions)")]
            fn internal_energy(
                &mut self,
                contributions: PyContributions,
            ) -> PyResult<Energy> {
                Ok(self.0.profile.internal_energy(contributions.into()).map_err(PyFeosError::from)?)
            }

            #[getter]
            fn get_grand_potential_density(&self) -> PyResult<Pressure<$si_arr<f64>>> {
                Ok(self.0.profile.grand_potential_density().map_err(PyFeosError::from)?)
            }
            $(
                #[getter]
                fn get_drho_dmu(&self) -> PyResult<typenum::Quot<Density<$si_arr3<f64>>, MolarEnergy>> {
                    Ok(self.0.profile.drho_dmu().map_err(PyFeosError::from)?)
                }
            )?

            #[getter]
            fn get_dn_dmu(&self) -> PyResult<typenum::Quot<Moles<Array2<f64>>, MolarEnergy>> {
                Ok(self.0.profile.dn_dmu().map_err(PyFeosError::from)?)
            }

            #[getter]
            fn get_drho_dp(&self) -> PyResult<typenum::Quot<Density<$si_arr2<f64>>, Pressure>> {
                Ok(self.0.profile.drho_dp().map_err(PyFeosError::from)?)
            }

            #[getter]
            fn get_dn_dp(&self) -> PyResult<typenum::Quot<Moles<Array1<f64>>, Pressure>> {
                Ok(self.0.profile.dn_dp().map_err(PyFeosError::from)?)
            }

            #[getter]
            fn get_drho_dt(&self) -> PyResult<typenum::Quot<Density<$si_arr2<f64>>, Temperature>> {
                Ok(self.0.profile.drho_dt().map_err(PyFeosError::from)?)
            }

            #[getter]
            fn get_dn_dt(&self) -> PyResult<typenum::Quot<Moles<Array1<f64>>, Temperature>> {
                Ok(self.0.profile.dn_dt().map_err(PyFeosError::from)?)
            }
        }
    };
}

macro_rules! impl_1d_profile {
    ($struct:ident, [$($ax:ident),+]) => {
        impl_profile!(
            $struct,
            PyArray1,
            PyArray2,
            Array1,
            Array2,
            PyArray2,
            [$([0, $ax]),+],
            Array3
        );
    };
}

macro_rules! impl_2d_profile {
    ($struct:ident, $ax1:ident, $ax2:ident) => {
        impl_profile!(
            $struct,
            PyArray2,
            PyArray3,
            Array2,
            Array3,
            PyArray3,
            [[0, $ax1], [1, $ax2]]
        );

        #[pymethods]
        impl $struct {
            #[getter]
            fn get_edges(&self) -> [Length<Array1<f64>>; 2] {
                self.0.profile.edges()
            }

            #[getter]
            fn get_meshgrid(&self) -> [Length<Array2<f64>>; 2] {
                self.0.profile.meshgrid()
            }
        }
    };
}

macro_rules! impl_3d_profile {
    ($struct:ident, $ax1:ident, $ax2:ident, $ax3:ident) => {
        impl_profile!(
            $struct,
            PyArray3,
            PyArray4,
            Array3,
            Array4,
            PyArray4,
            [[0, $ax1], [1, $ax2], [2, $ax3]]
        );

        #[pymethods]
        impl $struct {
            #[getter]
            fn get_edges(&self) -> [Length<Array1<f64>>; 3] {
                self.0.profile.edges()
            }

            #[getter]
            fn get_meshgrid(&self) -> [Length<Array3<f64>>; 3] {
                self.0.profile.meshgrid()
            }
        }
    };
}

pub(crate) use {impl_1d_profile, impl_2d_profile, impl_3d_profile, impl_profile};
