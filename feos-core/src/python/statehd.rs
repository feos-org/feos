use crate::StateHD;
use ndarray::Array1;
use num_dual::python::{PyDual3Dual64, PyDual3_64, PyDual64, PyHyperDual64, PyHyperDualDual64};
use num_dual::{Dual, Dual3, Dual3_64, Dual64, DualVec64, HyperDual, HyperDual64};
use pyo3::prelude::*;
use std::convert::From;

#[pyclass(name = "StateF")]
#[derive(Clone)]
pub struct PyStateF(StateHD<f64>);

impl From<StateHD<f64>> for PyStateF {
    fn from(s: StateHD<f64>) -> Self {
        Self(s)
    }
}

#[pyclass(name = "StateHD")]
#[derive(Clone)]
pub struct PyStateHD(StateHD<HyperDual64>);

impl From<StateHD<HyperDual64>> for PyStateHD {
    fn from(s: StateHD<HyperDual64>) -> Self {
        Self(s)
    }
}

#[pyclass(name = "StateHDD")]
#[derive(Clone)]
pub struct PyStateHDD(StateHD<HyperDual<Dual64, f64>>);

impl From<StateHD<HyperDual<Dual64, f64>>> for PyStateHDD {
    fn from(s: StateHD<HyperDual<Dual64, f64>>) -> Self {
        Self(s)
    }
}

#[pyclass(name = "StateHDDV2")]
#[derive(Clone)]
pub struct PyStateHDDV2(StateHD<HyperDual<DualVec64<2>, f64>>);

impl From<StateHD<HyperDual<DualVec64<2>, f64>>> for PyStateHDDV2 {
    fn from(s: StateHD<HyperDual<DualVec64<2>, f64>>) -> Self {
        Self(s)
    }
}

#[pyclass(name = "StateHDDV3")]
#[derive(Clone)]
pub struct PyStateHDDV3(StateHD<HyperDual<DualVec64<3>, f64>>);

impl From<StateHD<HyperDual<DualVec64<3>, f64>>> for PyStateHDDV3 {
    fn from(s: StateHD<HyperDual<DualVec64<3>, f64>>) -> Self {
        Self(s)
    }
}

#[pyclass(name = "StateD")]
#[derive(Clone)]
pub struct PyStateD(StateHD<Dual64>);

impl From<StateHD<Dual64>> for PyStateD {
    fn from(s: StateHD<Dual64>) -> Self {
        Self(s)
    }
}

#[pyclass(name = "StateDDV3")]
#[derive(Clone)]
pub struct PyStateDDV3(StateHD<Dual<DualVec64<3>, f64>>);

impl From<StateHD<Dual<DualVec64<3>, f64>>> for PyStateDDV3 {
    fn from(s: StateHD<Dual<DualVec64<3>, f64>>) -> Self {
        Self(s)
    }
}

#[pyclass(name = "StateD3")]
#[derive(Clone)]
pub struct PyStateD3(StateHD<Dual3_64>);

impl From<StateHD<Dual3_64>> for PyStateD3 {
    fn from(s: StateHD<Dual3_64>) -> Self {
        Self(s)
    }
}

#[pyclass(name = "StateD3D")]
#[derive(Clone)]
pub struct PyStateD3D(StateHD<Dual3<Dual64, f64>>);

impl From<StateHD<Dual3<Dual64, f64>>> for PyStateD3D {
    fn from(s: StateHD<Dual3<Dual64, f64>>) -> Self {
        Self(s)
    }
}

#[pyclass(name = "StateD3DV2")]
#[derive(Clone)]
pub struct PyStateD3DV2(StateHD<Dual3<DualVec64<2>, f64>>);

impl From<StateHD<Dual3<DualVec64<2>, f64>>> for PyStateD3DV2 {
    fn from(s: StateHD<Dual3<DualVec64<2>, f64>>) -> Self {
        Self(s)
    }
}

#[pyclass(name = "StateD3DV3")]
#[derive(Clone)]
pub struct PyStateD3DV3(StateHD<Dual3<DualVec64<3>, f64>>);

impl From<StateHD<Dual3<DualVec64<3>, f64>>> for PyStateD3DV3 {
    fn from(s: StateHD<Dual3<DualVec64<3>, f64>>) -> Self {
        Self(s)
    }
}

macro_rules! impl_state_hd {
    ($pystate:ty, $pyhd:ty, $state:ty, $hd:ty) => {
        #[pymethods]
        impl $pystate {
            #[new]
            pub fn new(temperature: $pyhd, volume: $pyhd, moles: Vec<$pyhd>) -> Self {
                let m = Array1::from(moles).mapv(<$hd>::from);
                Self(<$state>::new(temperature.into(), volume.into(), m))
            }

            #[getter]
            pub fn get_temperature(&self) -> $pyhd {
                <$pyhd>::from(self.0.temperature)
            }

            #[getter]
            pub fn get_volume(&self) -> $pyhd {
                <$pyhd>::from(self.0.volume)
            }

            #[getter]
            pub fn get_moles(&self) -> Vec<$pyhd> {
                self.0.moles.mapv(<$pyhd>::from).into_raw_vec()
            }

            #[getter]
            pub fn get_partial_density(&self) -> Vec<$pyhd> {
                self.0.partial_density.mapv(<$pyhd>::from).into_raw_vec()
            }

            #[getter]
            pub fn get_molefracs(&self) -> Vec<$pyhd> {
                self.0.molefracs.mapv(<$pyhd>::from).into_raw_vec()
            }

            #[getter]
            pub fn get_density(&self) -> $pyhd {
                <$pyhd>::from(self.0.partial_density.sum())
            }
        }
    };
}

impl_state_hd!(PyStateF, f64, StateHD<f64>, f64);
impl_state_hd!(PyStateHD, PyHyperDual64, StateHD<HyperDual64>, HyperDual64);
impl_state_hd!(
    PyStateHDD,
    PyHyperDualDual64,
    StateHD<HyperDual<Dual64, f64>>,
    HyperDual<Dual64, f64>
);
impl_state_hd!(PyStateD, PyDual64, StateHD<Dual64>, Dual64);
impl_state_hd!(PyStateD3, PyDual3_64, StateHD<Dual3_64>, Dual3_64);
impl_state_hd!(
    PyStateD3D,
    PyDual3Dual64,
    StateHD<Dual3<Dual64, f64>>,
    Dual3<Dual64, f64>
);
