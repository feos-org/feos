#![allow(non_snake_case)]
use feos_core::{IdealGasDyn, Molarweight, ResidualDyn, StateHD, Subset};
use nalgebra::DVector;
use num_dual::*;
use numpy::{PyArray, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use quantity::MolarWeight;
use std::any::Any;

pub struct PyIdealGas(Py<PyAny>);

impl PyIdealGas {
    pub fn new(obj: Bound<'_, PyAny>) -> PyResult<Self> {
        let attr = obj.hasattr("components")?;
        if !attr {
            panic!("Python Class has to have a method 'components' with signature:\n\tdef signature(self) -> int")
        }
        let attr = obj.hasattr("subset")?;
        if !attr {
            panic!("Python Class has to have a method 'subset' with signature:\n\tdef subset(self, component_list: List[int]) -> Self")
        }
        let attr = obj.hasattr("ln_lambda3")?;
        if !attr {
            panic!("{}", "Python Class has to have a method 'ln_lambda3' with signature:\n\tdef ln_lambda3(self, temperature: HD) -> HD\nwhere 'HD' has to be any (hyper-) dual number.")
        }
        Ok(Self(obj.unbind()))
    }
}

macro_rules! impl_ideal_gas {
    ($($py_hd_id:ident, $hd_ty:ty);*) => {
        impl IdealGasDyn for PyIdealGas {
            fn ideal_gas_model(&self) -> &'static str {
                "Ideal gas (Python)"
            }

            fn ln_lambda3<D: DualNum<f64> + Copy>(&self, temperature: D) -> D {
                let mut result = D::zero();

                $(
                    if let Some(t) = (&temperature as &dyn Any).downcast_ref::<$hd_ty>() {
                        let l3_any = (&mut result as &mut dyn Any).downcast_mut::<$hd_ty>().unwrap();
                        *l3_any = Python::with_gil(|py| {
                            let py_result = self
                                .0
                                .bind(py)
                                .call_method1("ln_lambda3", (<$py_hd_id>::from(t.clone()),))
                                .unwrap();
                            <$hd_ty>::from(py_result.extract::<$py_hd_id>().unwrap())
                        });
                        return result
                    }
                )*
                panic!("ln_lambda3: input data type not understood")
            }
        }
    };
}

/// Struct containing pointer to Python Class that implements Helmholtz energy.
pub struct PyResidual(Py<PyAny>);

impl PyResidual {
    pub fn new(obj: Bound<'_, PyAny>) -> PyResult<Self> {
        let attr = obj.hasattr("components")?;
        if !attr {
            panic!("Python Class has to have a method 'components' with signature:\n\tdef signature(self) -> int")
        }
        let attr = obj.hasattr("subset")?;
        if !attr {
            panic!("Python Class has to have a method 'subset' with signature:\n\tdef subset(self, component_list: List[int]) -> Self")
        }
        let attr = obj.hasattr("molar_weight")?;
        if !attr {
            panic!("Python Class has to have a method 'molar_weight' with signature:\n\tdef molar_weight(self) -> SIArray1\nwhere the size of the returned array has to be 'components'.")
        }
        let attr = obj.hasattr("max_density")?;
        if !attr {
            panic!("Python Class has to have a method 'max_density' with signature:\n\tdef max_density(self, moles: numpy.ndarray[float]) -> float\nwhere the size of the input array has to be 'components'.")
        }
        let attr = obj.hasattr("helmholtz_energy")?;
        if !attr {
            panic!("{}", "Python Class has to have a method 'helmholtz_energy' with signature:\n\tdef helmholtz_energy(self, state: StateHD) -> HD\nwhere 'HD' has to be any of {{float, Dual64, HyperDual64, HyperDualDual64, Dual3Dual64, Dual3_64}}.")
        }
        Ok(Self(obj.unbind()))
    }
}

// impl Components for PyResidual {
//     fn components(&self) -> usize {
//         Python::with_gil(|py| {
//             let py_result = self.0.bind(py).call_method0("components").unwrap();
//             py_result.extract().unwrap()
//         })
//     }

//     fn subset(&self, component_list: &[usize]) -> Self {
//         Python::with_gil(|py| {
//             let py_result = self
//                 .0
//                 .bind(py)
//                 .call_method1("subset", (component_list.to_vec(),))
//                 .unwrap();
//             Self::new(py_result.extract().unwrap()).unwrap()
//         })
//     }
// }

macro_rules! impl_residual {
    ($($py_state_id:ident, $py_hd_id:ident, $hd_ty:ty);*) => {
        impl ResidualDyn for PyResidual {
            fn components(&self) -> usize {
                Python::with_gil(|py| {
                    let py_result = self.0.bind(py).call_method0("components").unwrap();
                    py_result.extract().unwrap()
                })
            }

            fn compute_max_density<D: DualNum<f64> + Copy>(&self, molefracs: &DVector<D>) -> D {
                let mut rho = D::zero();

                $(
                    if let Some(x) = (molefracs as &dyn Any).downcast_ref::<DVector<$hd_ty>>() {
                        let r = (&mut rho as &mut dyn Any).downcast_mut::<$hd_ty>().unwrap();
                        *r = Python::with_gil(|py| {
                            let py_result = self
                                .0
                                .bind(py)
                                .call_method1("max_density", (x.iter().copied().map(<$py_hd_id>::from).collect::<Vec<_>>(),))
                                .unwrap();
                            <$hd_ty>::from(py_result.extract::<$py_hd_id>().unwrap())
                        });
                        return rho
                    }
                )*
                panic!("compute_max_density: input data type not understood")
            }

            // fn reduced_residual_helmholtz_energy_density<D: DualNum<f64> + Copy>(&self, state: &StateHD<D>) -> D {
            //     // result to write to
            //     let mut a = D::zero();

            //     $(
            //         if let Some(s) = (state as &dyn Any).downcast_ref::<StateHD<$hd_ty>>() {
            //             let d = (&mut a as &mut dyn Any).downcast_mut::<$hd_ty>().unwrap();
            //             *d = Python::with_gil(|py| {
            //                 let py_result = self
            //                     .0
            //                     .bind(py)
            //                     .call_method1("helmholtz_energy", (<$py_state_id>::from(s.clone()),))
            //                     .unwrap();
            //                 <$hd_ty>::from(py_result.extract::<$py_hd_id>().unwrap())
            //             });
            //             return a
            //         }
            //     )*
            //     panic!("helmholtz_energy: input data type not understood")
            // }

            fn reduced_helmholtz_energy_density_contributions<D: DualNum<f64> + Copy + >(
                    &self,
                    state: &StateHD<D>,
                ) -> Vec<(String, D)> {
                // result to write to
                let mut a = D::zero();

                $(
                    if let Some(s) = (state as &dyn Any).downcast_ref::<StateHD<$hd_ty>>() {
                        let d = (&mut a as &mut dyn Any).downcast_mut::<$hd_ty>().unwrap();
                        *d = Python::with_gil(|py| {
                            let py_result = self
                                .0
                                .bind(py)
                                .call_method1("helmholtz_energy", (<$py_state_id>::from(s.clone()),))
                                .unwrap();
                            <$hd_ty>::from(py_result.extract::<$py_hd_id>().unwrap())
                        });
                        return vec![("Python".to_string(), a)]
                    }
                )*
                panic!("helmholtz_energy: input data type not understood")
            }
        }

        impl Molarweight for PyResidual {
            fn molar_weight(&self) -> MolarWeight<DVector<f64>> {
                Python::with_gil(|py| {
                    let py_result = self.0.bind(py).call_method0("molar_weight").unwrap();
                    py_result
                        .extract::<MolarWeight<DVector<f64>>>()
                        .unwrap()
                })
            }
        }

        impl Subset for PyResidual {
            fn subset(&self, component_list: &[usize]) -> Self {
                Python::with_gil(|py| {
                    let py_result = self
                        .0
                        .bind(py)
                        .call_method1("subset", (component_list.to_vec(),))
                        .unwrap();
                    Self::new(py_result.extract().unwrap()).unwrap()
                })
            }
        }
    }
}

macro_rules! state {
    ($py_state_id:ident, $py_hd_id:ident, $hd_ty:ty) => {
        #[pyclass]
        #[derive(Clone)]
        struct $py_state_id(StateHD<$hd_ty>);

        impl From<StateHD<$hd_ty>> for $py_state_id {
            fn from(s: StateHD<$hd_ty>) -> Self {
                Self(s)
            }
        }

        #[pymethods]
        impl $py_state_id {
            #[new]
            pub fn new(temperature: $py_hd_id, volume: $py_hd_id, moles: Vec<$py_hd_id>) -> Self {
                let moles = moles.into_iter().map(<$hd_ty>::from).collect();
                Self(StateHD::<$hd_ty>::new(
                    temperature.into(),
                    volume.into(),
                    &DVector::from_vec(moles),
                ))
            }

            #[getter]
            pub fn get_temperature(&self) -> $py_hd_id {
                <$py_hd_id>::from(self.0.temperature)
            }

            // #[getter]
            // pub fn get_volume(&self) -> $py_hd_id {
            //     <$py_hd_id>::from(self.0.volume)
            // }

            // #[getter]
            // pub fn get_moles(&self) -> Vec<$py_hd_id> {
            //     self.0
            //         .moles
            //         .as_ndarray1()
            //         .mapv(<$py_hd_id>::from)
            //         .into_raw_vec_and_offset()
            //         .0
            // }

            #[getter]
            pub fn get_partial_density(&self) -> Vec<$py_hd_id> {
                self.0
                    .partial_density
                    .iter()
                    .copied()
                    .map(<$py_hd_id>::from)
                    .collect()
            }

            #[getter]
            pub fn get_molefracs(&self) -> Vec<$py_hd_id> {
                self.0
                    .molefracs
                    .iter()
                    .copied()
                    .map(<$py_hd_id>::from)
                    .collect()
            }

            #[getter]
            pub fn get_density(&self) -> $py_hd_id {
                <$py_hd_id>::from(self.0.partial_density.sum())
            }
        }
    };
}

macro_rules! dual_number {
    ($py_hd_id:ident, $hd_ty:ty, $py_field_ty:ty) => {
        #[pyclass]
        #[derive(Clone)]
        struct $py_hd_id($hd_ty);
        impl_dual_num!($py_hd_id, $hd_ty, $py_field_ty);
    };
}

macro_rules! impl_dual_state_helmholtz_energy {
    ($py_state_id:ident, $py_hd_id:ident, $hd_ty:ty, $py_field_ty:ty) => {
        dual_number!($py_hd_id, $hd_ty, $py_field_ty);
        state!($py_state_id, $py_hd_id, $hd_ty);
    };
}

// No definition of dual number necessary for f64
state!(PyStateF, f64, f64);

impl_dual_state_helmholtz_energy!(PyStateD, PyDual64, Dual64, f64);

dual_number!(PyDualVec3, DualSVec64<3>, f64);
impl_dual_state_helmholtz_energy!(
    PyStateDualDualVec3,
    PyDualDualVec3,
    Dual<DualSVec64<3>, f64>,
    PyDualVec3
);
impl_dual_state_helmholtz_energy!(PyStateHD, PyHyperDual64, HyperDual64, f64);
impl_dual_state_helmholtz_energy!(PyStateD2, PyDual2_64, Dual2_64, f64);
impl_dual_state_helmholtz_energy!(PyStateD3, PyDual3_64, Dual3_64, f64);
impl_dual_state_helmholtz_energy!(PyStateHDD, PyHyperDualDual64, HyperDual<Dual64, f64>, PyDual64);
dual_number!(PyDualVec2, DualSVec64<2>, f64);
impl_dual_state_helmholtz_energy!(
    PyStateHDDVec2,
    PyHyperDualVec2,
    HyperDual<DualSVec64<2>, f64>,
    PyDualVec2
);
impl_dual_state_helmholtz_energy!(
    PyStateHDDVec3,
    PyHyperDualVec3,
    HyperDual<DualSVec64<3>, f64>,
    PyDualVec3
);
impl_dual_state_helmholtz_energy!(
    PyStateD2D,
    PyDual2Dual64,
    Dual2<Dual64, f64>,
    PyDual64
);
impl_dual_state_helmholtz_energy!(
    PyStateD3D,
    PyDual3Dual64,
    Dual3<Dual64, f64>,
    PyDual64
);
impl_dual_state_helmholtz_energy!(
    PyStateD3DVec2,
    PyDual3DualVec2,
    Dual3<DualSVec64<2>, f64>,
    PyDualVec2
);
impl_dual_state_helmholtz_energy!(
    PyStateD3DVec3,
    PyDual3DualVec3,
    Dual3<DualSVec64<3>, f64>,
    PyDualVec3
);

impl_ideal_gas!(
    f64, f64;
    PyDual64, Dual64;
    PyDualDualVec3,
    Dual<DualSVec64<3>, f64>;
    PyHyperDual64, HyperDual64;
    PyDual2_64, Dual2_64;
    PyDual3_64, Dual3_64;
    PyHyperDualDual64, HyperDual<Dual64, f64>;
    PyHyperDualVec2,
    HyperDual<DualSVec64<2>, f64>;
    PyHyperDualVec3,
    HyperDual<DualSVec64<3>, f64>;
    PyDual2Dual64,
    Dual2<Dual64, f64>;
    PyDual3Dual64,
    Dual3<Dual64, f64>;
    PyDual3DualVec2,
    Dual3<DualSVec64<2>, f64>;
    PyDual3DualVec3,
    Dual3<DualSVec64<3>, f64>
);

impl_residual!(
    PyStateF, f64, f64;
    PyStateD, PyDual64, Dual64;
    PyStateDualDualVec3,
    PyDualDualVec3,
    Dual<DualSVec64<3>, f64>;
    PyStateHD, PyHyperDual64, HyperDual64;
    PyStateD2, PyDual2_64, Dual2_64;
    PyStateD3, PyDual3_64, Dual3_64;
    PyStateHDD, PyHyperDualDual64, HyperDual<Dual64, f64>;
    PyStateHDDVec2,
    PyHyperDualVec2,
    HyperDual<DualSVec64<2>, f64>;
    PyStateHDDVec3,
    PyHyperDualVec3,
    HyperDual<DualSVec64<3>, f64>;
    PyStateD2D,
    PyDual2Dual64,
    Dual2<Dual64, f64>;
    PyStateD3D,
    PyDual3Dual64,
    Dual3<Dual64, f64>;
    PyStateD3DVec2,
    PyDual3DualVec2,
    Dual3<DualSVec64<2>, f64>;
    PyStateD3DVec3,
    PyDual3DualVec3,
    Dual3<DualSVec64<3>, f64>
);
