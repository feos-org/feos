#[macro_export]
macro_rules! impl_equation_of_state {
    ($py_eos:ty) => {
        #[pymethods]
        impl $py_eos {
            /// Return maximum density for given amount of substance of each component.
            ///
            /// Parameters
            /// ----------
            /// moles : SIArray1, optional
            ///     The amount of substance in mol for each component.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(text_signature = "(moles=None)")]
            fn max_density(&self, moles: Option<PySIArray1>) -> PyResult<PySINumber> {
                let m = moles.map(|m| m.try_into()).transpose()?;
                Ok(self.0.max_density(m.as_ref())?.into())
            }
        }
    };
}

#[macro_export]
macro_rules! impl_virial_coefficients {
    ($py_eos:ty) => {
        #[pymethods]
        impl $py_eos {
            /// Calculate the second Virial coefficient B(T,x).
            ///
            /// Parameters
            /// ----------
            /// temperature : SINumber
            ///     The temperature for which B should be computed.
            /// moles : SIArray1, optional
            ///     The amount of substance in mol for each component.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(text_signature = "(temperature, moles=None)")]
            fn second_virial_coefficient(
                &self,
                temperature: PySINumber,
                moles: Option<PySIArray1>,
            ) -> PyResult<PySINumber> {
                let m = moles.map(|m| m.try_into()).transpose()?;
                Ok(self
                    .0
                    .second_virial_coefficient(temperature.try_into()?, m.as_ref())?
                    .into())
            }

            /// Calculate the third Virial coefficient C(T,x).
            ///
            /// Parameters
            /// ----------
            /// temperature : SINumber
            ///     The temperature for which C should be computed.
            /// moles : SIArray1, optional
            ///     The amount of substance in mol for each component.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(text_signature = "(temperature, moles=None)")]
            fn third_virial_coefficient(
                &self,
                temperature: PySINumber,
                moles: Option<PySIArray1>,
            ) -> PyResult<PySINumber> {
                let m = moles.map(|m| m.try_into()).transpose()?;
                Ok(self
                    .0
                    .third_virial_coefficient(temperature.try_into()?, m.as_ref())?
                    .into())
            }

            /// Calculate the derivative of the second Virial coefficient B(T,x)
            /// with respect to temperature.
            ///
            /// Parameters
            /// ----------
            /// temperature : SINumber
            ///     The temperature for which B' should be computed.
            /// moles : SIArray1, optional
            ///     The amount of substance in mol for each component.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(text_signature = "(temperature, moles=None)")]
            fn second_virial_coefficient_temperature_derivative(
                &self,
                temperature: PySINumber,
                moles: Option<PySIArray1>,
            ) -> PyResult<PySINumber> {
                let m = moles.map(|m| m.try_into()).transpose()?;
                Ok(self
                    .0
                    .second_virial_coefficient_temperature_derivative(
                        temperature.try_into()?,
                        m.as_ref(),
                    )?
                    .into())
            }

            /// Calculate the derivative of the third Virial coefficient C(T,x)
            /// with respect to temperature.
            ///
            /// Parameters
            /// ----------
            /// temperature : SINumber
            ///     The temperature for which C' should be computed.
            /// moles : SIArray1, optional
            ///     The amount of substance in mol for each component.
            ///
            /// Returns
            /// -------
            /// SINumber
            #[pyo3(text_signature = "(temperature, moles=None)")]
            fn third_virial_coefficient_temperature_derivative(
                &self,
                temperature: PySINumber,
                moles: Option<PySIArray1>,
            ) -> PyResult<PySINumber> {
                let m = moles.map(|m| m.try_into()).transpose()?;
                Ok(self
                    .0
                    .third_virial_coefficient_temperature_derivative(
                        temperature.try_into()?,
                        m.as_ref(),
                    )?
                    .into())
            }
        }
    };
}
