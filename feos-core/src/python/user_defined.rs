#![allow(non_snake_case)]
use crate::{Components, IdealGas, Molarweight, Residual, StateHD};
use ndarray::{Array1, ScalarOperand};
use num_dual::*;
use numpy::convert::IntoPyArray;
use numpy::{PyArray, PyReadonlyArray1, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use quantity::MolarWeight;
use std::any::Any;
use std::fmt;

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

impl Components for PyIdealGas {
    fn components(&self) -> usize {
        Python::with_gil(|py| {
            let py_result = self.0.bind(py).call_method0("components").unwrap();
            if py_result.get_type().name().unwrap() != "int" {
                panic!(
                    "Expected an integer for the components() method signature, got {}",
                    py_result.get_type().name().unwrap()
                );
            }
            py_result.extract().unwrap()
        })
    }

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

macro_rules! impl_ideal_gas {
    ($($py_hd_id:ident, $hd_ty:ty);*) => {
        impl IdealGas for PyIdealGas {
            fn ideal_gas_model(&self) -> String {
                "Ideal gas (Python)".to_string()
            }

            fn ln_lambda3<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D> {
                let mut result = Array1::from_elem((self.components(),), D::zero());

                $(
                    if let Some(t) = (&temperature as &dyn Any).downcast_ref::<$hd_ty>() {
                        let l3_any = (&mut result as &mut dyn Any).downcast_mut::<Array1<$hd_ty>>().unwrap();
                        *l3_any = Python::with_gil(|py| {
                            let py_result = self
                                .0
                                .bind(py)
                                .call_method1("ln_lambda3", (<$py_hd_id>::from(t.clone()),))
                                .unwrap();

                            // f64
                            if let Ok(r) = py_result.extract::<PyReadonlyArray1<f64>>() {
                                r.as_array().mapv(|ri| <$hd_ty>::from(ri))
                            // anything but f64
                            } else if let Ok(r) = py_result.extract::<PyReadonlyArray1<PyObject>>() {
                                r.as_array().map(|ri| <$hd_ty>::from(ri.extract::<$py_hd_id>(py).unwrap()))
                            } else {
                                    panic!("ln_lambda3: data type of result must be one-dimensional numpy ndarray")
                            }
                        });
                        return result
                    }
                )*
                panic!("ln_lambda3: input data type not understood")
            }
        }
    };
}

impl fmt::Display for PyIdealGas {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ideal gas (Python)")
    }
}

/// Struct containing pointer to Python Class that implements Helmholtz energy.
pub struct PyResidual(Py<PyAny>);

impl fmt::Display for PyResidual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Python residual")
    }
}

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

impl Components for PyResidual {
    fn components(&self) -> usize {
        Python::with_gil(|py| {
            let py_result = self.0.bind(py).call_method0("components").unwrap();
            if py_result.get_type().name().unwrap() != "int" {
                panic!(
                    "Expected an integer for the components() method signature, got {}",
                    py_result.get_type().name().unwrap()
                );
            }
            py_result.extract().unwrap()
        })
    }

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

macro_rules! impl_residual {
    ($($py_state_id:ident, $py_hd_id:ident, $hd_ty:ty);*) => {
        impl Residual for PyResidual {
            fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
                Python::with_gil(|py| {
                    let py_result = self
                        .0
                        .bind(py)
                        .call_method1("max_density", (moles.to_owned().into_pyarray(py),))
                        .unwrap();
                    py_result.extract().unwrap()
                })
            }

            fn residual_helmholtz_energy<D: DualNum<f64> + Copy>(&self, state: &StateHD<D>) -> D {
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
                        return a
                    }
                )*
                panic!("helmholtz_energy: input data type not understood")
            }

            fn residual_helmholtz_energy_contributions<D: DualNum<f64> + Copy + ScalarOperand>(
                    &self,
                    state: &StateHD<D>,
                ) -> Vec<(String, D)> {
                vec![("Python".to_string(), self.residual_helmholtz_energy(state))]
            }
        }

        impl Molarweight for PyResidual {
            fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
                Python::with_gil(|py| {
                    let py_result = self.0.bind(py).call_method0("molar_weight").unwrap();
                    py_result
                        .extract::<MolarWeight<Array1<f64>>>()
                        .unwrap()
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
                let m = Array1::from(moles).mapv(<$hd_ty>::from);
                Self(StateHD::<$hd_ty>::new(temperature.into(), volume.into(), m))
            }

            #[getter]
            pub fn get_temperature(&self) -> $py_hd_id {
                <$py_hd_id>::from(self.0.temperature)
            }

            #[getter]
            pub fn get_volume(&self) -> $py_hd_id {
                <$py_hd_id>::from(self.0.volume)
            }

            #[getter]
            pub fn get_moles(&self) -> Vec<$py_hd_id> {
                self.0
                    .moles
                    .mapv(<$py_hd_id>::from)
                    .into_raw_vec_and_offset()
                    .0
            }

            #[getter]
            pub fn get_partial_density(&self) -> Vec<$py_hd_id> {
                self.0
                    .partial_density
                    .mapv(<$py_hd_id>::from)
                    .into_raw_vec_and_offset()
                    .0
            }

            #[getter]
            pub fn get_molefracs(&self) -> Vec<$py_hd_id> {
                self.0
                    .molefracs
                    .mapv(<$py_hd_id>::from)
                    .into_raw_vec_and_offset()
                    .0
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
