#[cfg(feature = "estimator")]
use crate::estimator::*;
#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::python::PyGcPcSaftEosParameters;
#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::{GcPcSaft, GcPcSaftOptions};
#[cfg(feature = "estimator")]
use crate::impl_estimator;
#[cfg(all(feature = "estimator", feature = "pcsaft"))]
use crate::impl_estimator_entropy_scaling;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::python::PyPcSaftParameters;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::{DQVariants, PcSaft, PcSaftOptions};
#[cfg(feature = "pets")]
use crate::pets::python::PyPetsParameters;
#[cfg(feature = "pets")]
use crate::pets::{Pets, PetsOptions};
#[cfg(feature = "uvtheory")]
use crate::uvtheory::python::PyUVParameters;
#[cfg(feature = "uvtheory")]
use crate::uvtheory::{Perturbation, UVTheory, UVTheoryOptions};
#[cfg(feature = "python")]
use feos_core::python::user_defined::PyEoSObj;
use feos_core::cubic::PengRobinson;
use feos_core::*;
use ndarray::Array1;
#[cfg(feature = "estimator")]
use pyo3::wrap_pymodule;
use quantity::si::*;

pub enum EosVariant {
    #[cfg(feature = "pcsaft")]
    PcSaft(PcSaft),
    #[cfg(feature = "gc_pcsaft")]
    GcPcSaft(GcPcSaft),
    PengRobinson(PengRobinson),
    #[cfg(feature = "python")]
    Python(PyEoSObj),
    #[cfg(feature = "pets")]
    Pets(Pets),
    #[cfg(feature = "uvtheory")]
    UVTheory(UVTheory),
}

impl EquationOfState for EosVariant {
    fn components(&self) -> usize {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.components(),
            #[cfg(feature = "gc_pcsaft")]
            EosVariant::GcPcSaft(eos) => eos.components(),
            EosVariant::PengRobinson(eos) => eos.components(),
            #[cfg(feature = "python")]
            EosVariant::Python(eos) => eos.components(),
            #[cfg(feature = "pets")]
            EosVariant::Pets(eos) => eos.components(),
            #[cfg(feature = "uvtheory")]
            EosVariant::UVTheory(eos) => eos.components(),
        }
    }

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.compute_max_density(moles),
            #[cfg(feature = "gc_pcsaft")]
            EosVariant::GcPcSaft(eos) => eos.compute_max_density(moles),
            EosVariant::PengRobinson(eos) => eos.compute_max_density(moles),
            #[cfg(feature = "python")]
            EosVariant::Python(eos) => eos.compute_max_density(moles),
            #[cfg(feature = "pets")]
            EosVariant::Pets(eos) => eos.compute_max_density(moles),
            #[cfg(feature = "uvtheory")]
            EosVariant::UVTheory(eos) => eos.compute_max_density(moles),
        }
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => Self::PcSaft(eos.subset(component_list)),
            #[cfg(feature = "gc_pcsaft")]
            EosVariant::GcPcSaft(eos) => Self::GcPcSaft(eos.subset(component_list)),
            EosVariant::PengRobinson(eos) => Self::PengRobinson(eos.subset(component_list)),
            #[cfg(feature = "python")]
            EosVariant::Python(eos) => Self::Python(eos.subset(component_list)),
            #[cfg(feature = "pets")]
            EosVariant::Pets(eos) => Self::Pets(eos.subset(component_list)),
            #[cfg(feature = "uvtheory")]
            EosVariant::UVTheory(eos) => Self::UVTheory(eos.subset(component_list)),
        }
    }

    fn residual(&self) -> &[Box<dyn HelmholtzEnergy>] {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.residual(),
            #[cfg(feature = "gc_pcsaft")]
            EosVariant::GcPcSaft(eos) => eos.residual(),
            EosVariant::PengRobinson(eos) => eos.residual(),
            #[cfg(feature = "python")]
            EosVariant::Python(eos) => eos.residual(),
            #[cfg(feature = "pets")]
            EosVariant::Pets(eos) => eos.residual(),
            #[cfg(feature = "uvtheory")]
            EosVariant::UVTheory(eos) => eos.residual(),
        }
    }

    fn ideal_gas(&self) -> &dyn IdealGasContribution {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.ideal_gas(),
            #[cfg(feature = "gc_pcsaft")]
            EosVariant::GcPcSaft(eos) => eos.ideal_gas(),
            EosVariant::PengRobinson(eos) => eos.ideal_gas(),
            #[cfg(feature = "python")]
            EosVariant::Python(eos) => eos.ideal_gas(),
            #[cfg(feature = "pets")]
            EosVariant::Pets(eos) => eos.ideal_gas(),
            #[cfg(feature = "uvtheory")]
            EosVariant::UVTheory(eos) => eos.ideal_gas(),
        }
    }
}

impl MolarWeight<SIUnit> for EosVariant {
    fn molar_weight(&self) -> SIArray1 {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.molar_weight(),
            #[cfg(feature = "gc_pcsaft")]
            EosVariant::GcPcSaft(eos) => eos.molar_weight(),
            EosVariant::PengRobinson(eos) => eos.molar_weight(),
            #[cfg(feature = "python")]
            EosVariant::Python(eos) => eos.molar_weight(),
            #[cfg(feature = "pets")]
            EosVariant::Pets(eos) => eos.molar_weight(),
            #[cfg(feature = "uvtheory")]
            EosVariant::UVTheory(_) => unimplemented!(),
        }
    }
}

#[cfg(feature = "pcsaft")]
impl EntropyScaling<SIUnit> for EosVariant {
    fn viscosity_reference(
        &self,
        temperature: SINumber,
        volume: SINumber,
        moles: &SIArray1,
    ) -> EosResult<SINumber> {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.viscosity_reference(temperature, volume, moles),
            _ => unimplemented!(),
        }
    }

    fn viscosity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.viscosity_correlation(s_res, x),
            _ => unimplemented!(),
        }
    }

    fn diffusion_reference(
        &self,
        temperature: SINumber,
        volume: SINumber,
        moles: &SIArray1,
    ) -> EosResult<SINumber> {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.diffusion_reference(temperature, volume, moles),
            _ => unimplemented!(),
        }
    }

    fn diffusion_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.diffusion_correlation(s_res, x),
            _ => unimplemented!(),
        }
    }

    fn thermal_conductivity_reference(
        &self,
        temperature: SINumber,
        volume: SINumber,
        moles: &SIArray1,
    ) -> EosResult<SINumber> {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => {
                eos.thermal_conductivity_reference(temperature, volume, moles)
            }
            _ => unimplemented!(),
        }
    }

    fn thermal_conductivity_correlation(&self, s_res: f64, x: &Array1<f64>) -> EosResult<f64> {
        match self {
            #[cfg(feature = "pcsaft")]
            EosVariant::PcSaft(eos) => eos.thermal_conductivity_correlation(s_res, x),
            _ => unimplemented!(),
        }
    }
}