macro_rules! impl_profile {
    ($struct:ident) => {
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
            ) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArray1<f64>>, f64)> {
                let (res_rho, res_mu, res_norm) = self.0.profile.residual(log).map_err(PyFeosError::from)?;
                Ok((res_rho.view().into_dyn().to_pyarray(py), res_mu.view().to_pyarray(py), res_norm))
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

            #[getter]
            fn get_axes(&self) -> Vec<Length<Array1<f64>>>{
                self.0.profile.axes()
            }

            #[getter]
            fn get_edges(&self) -> Vec<Length<Array1<f64>>> {
                self.0.profile.edges()
            }

            #[getter]
            fn get_grid<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let mut grid = self.0.profile.grid.mesh();
                if grid.len() == 1 {
                    grid.pop().unwrap().into_pyobject(py)
                } else {
                    grid.into_pyobject(py)
                }
            }

            #[getter]
            fn get_temperature(&self) -> Temperature {
                self.0.profile.temperature
            }

            #[getter]
            fn get_density(&self) -> Density<ArrayD<f64>> {
                self.0.profile.density.clone().into_dyn()
            }

            #[getter]
            fn get_moles(&self) -> Moles<DVector<f64>> {
                self.0.profile.moles()
            }

            #[getter]
            fn get_total_moles(&self) -> Moles {
                self.0.profile.total_moles()
            }

            #[getter]
            fn get_external_potential(&self) -> Energy<ArrayD<f64>> {
                self.0.profile.external_potential().clone().into_dyn()
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
            ) -> PyResult<Vec<Bound<'py, PyArrayDyn<f64>>>> {
                let n = self.0.profile.weighted_densities().map_err(PyFeosError::from)?;
                Ok(n.into_iter().map(|n| n.view().into_dyn().to_pyarray(py)).collect())
            }

            #[getter]
            fn get_functional_derivative<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
                Ok(self.0.profile.functional_derivative().map_err(PyFeosError::from)?.view().into_dyn().to_pyarray(py))
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
            ) -> PyResult<<Entropy<ArrayD<f64>> as std::ops::Div<Volume>>::Output> {
                Ok(self.0.profile.entropy_density(contributions.into()).map_err(PyFeosError::from)?.into_dyn())
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
            fn get_grand_potential_density(&self) -> PyResult<Pressure<ArrayD<f64>>> {
                Ok(self.0.profile.grand_potential_density().map_err(PyFeosError::from)?.into_dyn())
            }

            #[getter]
            fn get_drho_dmu(&self) -> PyResult<<Density<ArrayD<f64>> as std::ops::Div<MolarEnergy>>::Output> {
                Ok(self.0.profile.drho_dmu().map_err(PyFeosError::from)?.into_dyn())
            }

            #[getter]
            fn get_dn_dmu(&self) -> PyResult<<Moles<DMatrix<f64>> as std::ops::Div<MolarEnergy>>::Output> {
                Ok(self.0.profile.dn_dmu().map_err(PyFeosError::from)?)
            }

            #[getter]
            fn get_drho_dp(&self) -> PyResult<<Density<ArrayD<f64>> as std::ops::Div<Pressure>>::Output> {
                Ok(self.0.profile.drho_dp().map_err(PyFeosError::from)?.into_dyn())
            }

            #[getter]
            fn get_dn_dp(&self) -> PyResult<<Moles<DVector<f64>> as std::ops::Div<Pressure>>::Output> {
                Ok(self.0.profile.dn_dp().map_err(PyFeosError::from)?)
            }

            #[getter]
            fn get_drho_dt(&self) -> PyResult<<Density<ArrayD<f64>> as std::ops::Div<Temperature>>::Output> {
                Ok(self.0.profile.drho_dt().map_err(PyFeosError::from)?.into_dyn())
            }

            #[getter]
            fn get_dn_dt(&self) -> PyResult<<Moles<DVector<f64>> as std::ops::Div<Temperature>>::Output> {
                Ok(self.0.profile.dn_dt().map_err(PyFeosError::from)?)
            }
        }
    };
}

pub(crate) use impl_profile;
