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
            #[pyo3(text_signature = "(moles=None)", signature = (moles=None))]
            fn max_density(&self, moles: Option<Moles<Array1<f64>>>) -> PyResult<Density> {
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
            #[pyo3(text_signature = "(temperature, moles=None)", signature = (temperature, moles=None))]
            fn second_virial_coefficient(
                &self,
                temperature: Temperature,
                moles: Option<Moles<Array1<f64>>>,
            ) -> PyResult<Quot<f64, Density>> {
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
            #[pyo3(text_signature = "(temperature, moles=None)", signature = (temperature, moles=None))]
            fn third_virial_coefficient(
                &self,
                temperature: Temperature,
                moles: Option<Moles<Array1<f64>>>,
            ) -> PyResult<Quot<Quot<f64, Density>, Density>> {
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
            #[pyo3(text_signature = "(temperature, moles=None)", signature = (temperature, moles=None))]
            fn second_virial_coefficient_temperature_derivative(
                &self,
                temperature: Temperature,
                moles: Option<Moles<Array1<f64>>>,
            ) -> PyResult<Quot<Quot<f64, Density>, Temperature>> {
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
            #[pyo3(text_signature = "(temperature, moles=None)", signature = (temperature, moles=None))]
            fn third_virial_coefficient_temperature_derivative(
                &self,
                temperature: Temperature,
                moles: Option<Moles<Array1<f64>>>,
            ) -> PyResult<Quot<Quot<Quot<f64, Density>, Density>, Temperature>> {
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
