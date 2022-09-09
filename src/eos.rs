#[cfg(feature = "gc_pcsaft")]
use crate::gc_pcsaft::GcPcSaft;
#[cfg(feature = "pcsaft")]
use crate::pcsaft::PcSaft;
#[cfg(feature = "pets")]
use crate::pets::Pets;
#[cfg(feature = "uvtheory")]
use crate::uvtheory::UVTheory;
use feos_core::cubic::PengRobinson;
#[cfg(feature = "python")]
use feos_core::python::user_defined::PyEoSObj;
use feos_core::*;
use feos_derive::EquationOfState;
use ndarray::Array1;
use quantity::si::*;

#[derive(EquationOfState)]
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
    #[skip_impl(molar_weight)]
    UVTheory(UVTheory),
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
