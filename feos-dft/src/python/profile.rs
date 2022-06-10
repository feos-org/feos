#[macro_export]
macro_rules! impl_profile {
    ($struct:ident, $arr:ident, $arr2:ident, $si_arr:ident, $si_arr2:ident, $py_arr2:ident, [$([$ind:expr, $ax:ident]),+]) => {
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
            #[args(log = "false")]
            #[pyo3(text_signature = "($self, log)")]
            fn residual<'py>(
                &self,
                log: bool,
                py: Python<'py>,
            ) -> PyResult<(&'py $arr2<f64>, &'py PyArray1<f64>)> {
                let (res_rho, res_mu) = self.0.profile.residual(log)?;
                Ok((res_rho.view().to_pyarray(py), res_mu.view().to_pyarray(py)))
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
            #[args(debug = "false")]
            #[pyo3(text_signature = "($self, solver=None, debug=False)")]
            fn solve(slf: &PyCell<Self>, solver: Option<PyDFTSolver>, debug: bool) -> PyResult<&PyCell<Self>> {
                slf.borrow_mut()
                    .0
                    .solve_inplace(solver.map(|s| s.0).as_ref(), debug)?;
                Ok(slf)
            }

            $(
            #[getter]
            fn $ax(&self) -> PySIArray1 {
                PySIArray1::from(self.0.profile.grid.grids()[$ind].clone() * SIUnit::reference_length())
            })+

            #[getter]
            fn get_temperature(&self) -> PySINumber {
                PySINumber::from(self.0.profile.temperature)
            }

            #[getter]
            fn get_density(&self) -> $si_arr2 {
                $si_arr2::from(self.0.profile.density.clone())
            }

            #[getter]
            fn get_moles(&self) -> PySIArray1 {
                PySIArray1::from(self.0.profile.moles())
            }

            #[getter]
            fn get_total_moles(&self) -> PySINumber {
                PySINumber::from(self.0.profile.total_moles())
            }

            #[getter]
            fn get_external_potential<'py>(&self, py: Python<'py>) -> &'py $py_arr2<f64> {
                self.0.profile.external_potential.view().to_pyarray(py)
            }

            #[getter]
            fn get_chemical_potential(&self) -> PySIArray1 {
                PySIArray1::from(self.0.profile.chemical_potential())
            }

            #[getter]
            fn get_bulk(&self) -> PyState {
                PyState(self.0.profile.bulk.clone())
            }

            #[getter]
            fn get_weighted_densities<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Vec<&'py $arr2<f64>>> {
                let n = self.0.profile.weighted_densities()?;
                Ok(n.into_iter().map(|n| n.view().to_pyarray(py)).collect())
            }

            #[getter]
            fn get_functional_derivative<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<&'py $arr2<f64>> {
                Ok(self.0.profile.functional_derivative()?.view().to_pyarray(py))
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
            #[args(contributions = "Contributions::Total")]
            #[pyo3(text_signature = "($self, contributions)")]
            fn entropy_density(
                &mut self,
                contributions: Contributions,
            ) -> PyResult<$si_arr> {
                Ok($si_arr::from(
                    self.0.profile.entropy_density(contributions)?,
                ))
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
            #[args(contributions = "Contributions::Total")]
            #[pyo3(text_signature = "($self, contributions)")]
            fn entropy(
                &mut self,
                contributions: Contributions,
            ) -> PyResult<PySINumber> {
                Ok(PySINumber::from(
                    self.0.profile.entropy(contributions)?,
                ))
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
            #[args(contributions = "Contributions::Total")]
            #[pyo3(text_signature = "($self, contributions)")]
            fn internal_energy(
                &mut self,
                contributions: Contributions,
            ) -> PyResult<PySINumber> {
                Ok(PySINumber::from(
                    self.0.profile.internal_energy(contributions)?,
                ))
            }

            #[getter]
            fn get_grand_potential_density(&self) -> PyResult<$si_arr> {
                Ok($si_arr::from(
                    self.0.profile.grand_potential_density()?,
                ))
            }
        }
    };
}

#[macro_export]
macro_rules! impl_1d_profile {
    ($struct:ident, [$($ax:ident),+]) => {
        impl_profile!(
            $struct,
            PyArray1,
            PyArray2,
            PySIArray1,
            PySIArray2,
            PyArray2,
            [$([0, $ax]),+]
        );
    };
}

#[macro_export]
macro_rules! impl_2d_profile {
    ($struct:ident, $ax1:ident, $ax2:ident) => {
        impl_profile!(
            $struct,
            PyArray2,
            PyArray3,
            PySIArray2,
            PySIArray3,
            PyArray3,
            [[0, $ax1], [1, $ax2]]
        );

        #[pymethods]
        impl $struct {
            #[getter]
            fn get_edges(&self) -> (PySIArray1, PySIArray1) {
                let (edge1, edge2) = self.0.profile.edges();
                (edge1.into(), edge2.into())
            }
        }
    };
}

#[macro_export]
macro_rules! impl_3d_profile {
    ($struct:ident, $ax1:ident, $ax2:ident, $ax3:ident) => {
        impl_profile!(
            $struct,
            PyArray3,
            PyArray4,
            PySIArray3,
            PySIArray4,
            PyArray4,
            [[0, $ax1], [1, $ax2], [2, $ax3]]
        );

        #[pymethods]
        impl $struct {
            #[getter]
            fn get_edges(&self) -> (PySIArray1, PySIArray1, PySIArray1) {
                let (edge1, edge2, edge3) = self.0.profile.edges();
                (edge1.into(), edge2.into(), edge3.into())
            }
        }
    };
}
