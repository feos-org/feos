use crate::eos::PyEquationOfState;
use crate::ideal_gas::IdealGasModel;
use crate::residual::ResidualModel;
use feos::hard_sphere::{FMTFunctional, FMTVersion};
use feos_core::{Components, EquationOfState};
use feos_dft::Geometry;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::pyclass;
use std::sync::Arc;

mod adsorption;
mod interface;
mod profile;
mod solvation;
mod solver;

pub(crate) use adsorption::{
    PyAdsorption1D, PyAdsorption3D, PyExternalPotential, PyPore1D, PyPore2D, PyPore3D,
};
pub(crate) use interface::{PyPlanarInterface, PySurfaceTensionDiagram};
pub(crate) use solvation::{PyPairCorrelation, PySolvationProfile};
pub(crate) use solver::{PyDFTSolver, PyDFTSolverLog};

/// Geometries of individual axes.
#[derive(Clone, Copy, PartialEq)]
#[pyclass(name = "Geometry", eq, eq_int)]
pub enum PyGeometry {
    Cartesian,
    Cylindrical,
    Spherical,
}

impl From<Geometry> for PyGeometry {
    fn from(geometry: Geometry) -> Self {
        match geometry {
            Geometry::Cartesian => PyGeometry::Cartesian,
            Geometry::Cylindrical => PyGeometry::Cylindrical,
            Geometry::Spherical => PyGeometry::Spherical,
        }
    }
}

impl From<PyGeometry> for Geometry {
    fn from(geometry: PyGeometry) -> Self {
        match geometry {
            PyGeometry::Cartesian => Geometry::Cartesian,
            PyGeometry::Cylindrical => Geometry::Cylindrical,
            PyGeometry::Spherical => Geometry::Spherical,
        }
    }
}

/// Different versions of fundamental measure theory.
#[derive(Clone, Copy, PartialEq)]
#[pyclass(name = "FMTVersion", eq, eq_int)]
pub enum PyFMTVersion {
    /// White Bear ([Roth et al., 2002](https://doi.org/10.1088/0953-8984/14/46/313)) or modified ([Yu and Wu, 2002](https://doi.org/10.1063/1.1520530)) fundamental measure theory
    WhiteBear,
    /// Scalar fundamental measure theory by [Kierlik and Rosinberg, 1990](https://doi.org/10.1103/PhysRevA.42.3382)
    KierlikRosinberg,
    /// Anti-symmetric White Bear fundamental measure theory ([Rosenfeld et al., 1997](https://doi.org/10.1103/PhysRevE.55.4245)) and SI of ([Kessler et al., 2021](https://doi.org/10.1016/j.micromeso.2021.111263))
    AntiSymWhiteBear,
}

impl From<FMTVersion> for PyFMTVersion {
    fn from(fmt_version: FMTVersion) -> Self {
        match fmt_version {
            FMTVersion::WhiteBear => PyFMTVersion::WhiteBear,
            FMTVersion::KierlikRosinberg => PyFMTVersion::KierlikRosinberg,
            FMTVersion::AntiSymWhiteBear => PyFMTVersion::AntiSymWhiteBear,
        }
    }
}

impl From<PyFMTVersion> for FMTVersion {
    fn from(fmt_version: PyFMTVersion) -> Self {
        match fmt_version {
            PyFMTVersion::WhiteBear => FMTVersion::WhiteBear,
            PyFMTVersion::KierlikRosinberg => FMTVersion::KierlikRosinberg,
            PyFMTVersion::AntiSymWhiteBear => FMTVersion::AntiSymWhiteBear,
        }
    }
}

#[pyclass(name = "HelmholtzEnergyFunctional")]
#[derive(Clone)]
pub struct PyHelmholtzEnergyFunctional;

#[pymethods]
impl PyHelmholtzEnergyFunctional {
    /// Helmholtz energy functional for hard sphere systems.
    ///
    /// Parameters
    /// ----------
    /// sigma : numpy.ndarray[float]
    ///     The diameters of the hard spheres in Angstrom.
    /// fmt_version : FMTVersion
    ///     The specific variant of the FMT term.
    ///
    /// Returns
    /// -------
    /// HelmholtzEnergyFunctional
    #[staticmethod]
    fn fmt(
        sigma: &Bound<'_, PyArray1<f64>>,
        fmt_version: PyFMTVersion,
    ) -> PyResult<PyEquationOfState> {
        let func = Arc::new(ResidualModel::FmtFunctional(FMTFunctional::new(
            &sigma.to_owned_array(),
            fmt_version.into(),
        )));
        let ideal_gas = Arc::new(IdealGasModel::NoModel(func.components()));
        Ok(PyEquationOfState(Arc::new(EquationOfState::new(
            ideal_gas, func,
        ))))
    }
}

// #[pymodule]
// pub fn dft(m: &Bound<'_, PyModule>) -> PyResult<()> {
//     m.add_class::<PyFMTVersion>()?;
//     m.add_class::<PyHelmholtzEnergyFunctional>()?;

//     m.add_class::<PyPlanarInterface>()?;
//     m.add_class::<PyGeometry>()?;
//     m.add_class::<PyPore1D>()?;
//     m.add_class::<PyPore2D>()?;
//     m.add_class::<PyPore3D>()?;
//     m.add_class::<PyPairCorrelation>()?;
//     m.add_class::<PyExternalPotential>()?;
//     m.add_class::<PyAdsorption1D>()?;
//     m.add_class::<PyAdsorption3D>()?;
//     m.add_class::<PySurfaceTensionDiagram>()?;
//     m.add_class::<PyDFTSolver>()?;
//     m.add_class::<PySolvationProfile>()?;

//     Ok(())
// }
