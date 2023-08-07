use crate::adsorption::ExternalPotential;
use numpy::PyArray1;
use pyo3::prelude::*;
use quantity::python::{PySIArray2, PySINumber};

/// A collection of external potentials.
#[pyclass(name = "ExternalPotential")]
#[derive(Clone)]
pub struct PyExternalPotential(pub ExternalPotential);

#[pymethods]
#[allow(non_snake_case)]
impl PyExternalPotential {
    /// Hard wall potential
    ///
    /// .. math:: V_i^\mathrm{ext}(z)=\begin{cases}\infty&z\leq\sigma_{si}\\\\0&z>\sigma_{si}\end{cases},~~~~\sigma_{si}=\frac{1}{2}\left(\sigma_{ss}+\sigma_{ii}\right)
    ///
    /// Parameters
    /// ----------
    /// sigma_ss : float
    ///     Segment diameter of the solid.
    ///
    /// Returns
    /// -------
    /// ExternalPotential
    ///
    #[staticmethod]
    #[allow(non_snake_case)]
    pub fn HardWall(sigma_ss: f64) -> Self {
        Self(ExternalPotential::HardWall { sigma_ss })
    }

    /// 9-3 Lennard-Jones potential
    ///
    /// .. math:: V_i^\mathrm{ext}(z)=\frac{2\pi}{45} m_i\varepsilon_{si}\sigma_{si}^3\rho_s\left(2\left(\frac{\sigma_{si}}{z}\right)^9-15\left(\frac{\sigma_{si}}{z}\right)^3\right),~~~~\varepsilon_{si}=\sqrt{\varepsilon_{ss}\varepsilon_{ii}},~~~~\sigma_{si}=\frac{1}{2}\left(\sigma_{ss}+\sigma_{ii}\right)
    ///
    /// Parameters
    /// ----------
    /// sigma_ss : float
    ///     Segment diameter of the solid.
    /// epsilon_k_ss : float
    ///     Energy parameter of the solid.
    /// rho_s : float
    ///     Density of the solid.
    ///
    /// Returns
    /// -------
    /// ExternalPotential
    ///
    #[staticmethod]
    pub fn LJ93(sigma_ss: f64, epsilon_k_ss: f64, rho_s: f64) -> Self {
        Self(ExternalPotential::LJ93 {
            sigma_ss,
            epsilon_k_ss,
            rho_s,
        })
    }

    /// Simple 9-3 Lennard-Jones potential
    ///
    /// .. math:: V_i^\mathrm{ext}(z)=\varepsilon_{si}\left(\left(\frac{\sigma_{si}}{z}\right)^9-\left(\frac{\sigma_{si}}{z}\right)^3\right),~~~~\varepsilon_{si}=\sqrt{\varepsilon_{ss}\varepsilon_{ii}},~~~~\sigma_{si}=\frac{1}{2}\left(\sigma_{ss}+\sigma_{ii}\right)
    ///
    /// Parameters
    /// ----------
    /// sigma_ss : float
    ///     Segment diameter of the solid.
    /// epsilon_k_ss : float
    ///     Energy parameter of the solid.
    ///
    /// Returns
    /// -------
    /// ExternalPotential
    ///
    #[staticmethod]
    pub fn SimpleLJ93(sigma_ss: f64, epsilon_k_ss: f64) -> Self {
        Self(ExternalPotential::SimpleLJ93 {
            sigma_ss,
            epsilon_k_ss,
        })
    }

    /// Custom 9-3 Lennard-Jones potential
    ///
    /// .. math:: V_i^\mathrm{ext}(z)=\varepsilon_{si}\left(\left(\frac{\sigma_{si}}{z}\right)^9-\left(\frac{\sigma_{si}}{z}\right)^3\right)
    ///
    /// Parameters
    /// ----------
    /// sigma_sf : numpy.ndarray[float]
    ///     Solid-fluid interaction diameters.
    /// epsilon_k_sf : numpy.ndarray[float]
    ///     Solid-fluid interaction energies.
    ///
    /// Returns
    /// -------
    /// ExternalPotential
    ///
    #[staticmethod]
    pub fn CustomLJ93(sigma_sf: &PyArray1<f64>, epsilon_k_sf: &PyArray1<f64>) -> Self {
        Self(ExternalPotential::CustomLJ93 {
            sigma_sf: sigma_sf.to_owned_array(),
            epsilon_k_sf: epsilon_k_sf.to_owned_array(),
        })
    }

    /// Steele potential
    ///
    /// .. math:: V_i^\mathrm{ext}(z)=2\pi m_i\xi\varepsilon_{si}\sigma_{si}^2\Delta\rho_s\left(0.4\left(\frac{\sigma_{si}}{z}\right)^{10}-\left(\frac{\sigma_{si}}{z}\right)^4-\frac{\sigma_{si}^4}{3\Delta\left(z+0.61\Delta\right)^3}\right),~~~~\varepsilon_{si}=\sqrt{\varepsilon_{ss}\varepsilon_{ii}},~~~~\sigma_{si}=\frac{1}{2}\left(\sigma_{ss}+\sigma_{ii}\right),~~~~\Delta=3.35
    ///
    /// Parameters
    /// ----------
    /// sigma_ss : float
    ///     Segment diameter of the solid.
    /// epsilon_k_ss : float
    ///     Energy parameter of the solid.
    /// rho_s : float
    ///     Density of the solid.
    /// xi : float, optional
    ///     Binary wall-fluid interaction parameter.
    ///
    /// Returns
    /// -------
    /// ExternalPotential
    ///
    #[staticmethod]
    #[pyo3(text_signature = "(sigma_ss, epsilon_k_ss, rho_s, xi=None)")]
    pub fn Steele(sigma_ss: f64, epsilon_k_ss: f64, rho_s: f64, xi: Option<f64>) -> Self {
        Self(ExternalPotential::Steele {
            sigma_ss,
            epsilon_k_ss,
            rho_s,
            xi,
        })
    }

    /// Steele potential with custom combining rules
    ///
    /// .. math:: V_i^\mathrm{ext}(z)=2\pi m_i\xi\varepsilon_{si}\sigma_{si}^2\Delta\rho_s\left(0.4\left(\frac{\sigma_{si}}{z}\right)^{10}-\left(\frac{\sigma_{si}}{z}\right)^4-\frac{\sigma_{si}^4}{3\Delta\left(z+0.61\Delta\right)^3}\right),~~~~\Delta=3.35
    ///
    /// Parameters
    /// ----------
    /// sigma_sf : numpy.ndarray[float]
    ///     Solid-fluid interaction diameters.
    /// epsilon_k_sf : numpy.ndarray[float]
    ///     Solid-fluid interaction energies.
    /// rho_s : float
    ///     Density of the solid.
    /// xi : float, optional
    ///     Binary wall-fluid interaction parameter.
    ///
    /// Returns
    /// -------
    /// ExternalPotential
    ///
    #[staticmethod]
    #[pyo3(text_signature = "(sigma_sf, epsilon_k_sf, rho_s, xi=None)")]
    pub fn CustomSteele(
        sigma_sf: &PyArray1<f64>,
        epsilon_k_sf: &PyArray1<f64>,
        rho_s: f64,
        xi: Option<f64>,
    ) -> Self {
        Self(ExternalPotential::CustomSteele {
            sigma_sf: sigma_sf.to_owned_array(),
            epsilon_k_sf: epsilon_k_sf.to_owned_array(),
            rho_s,
            xi,
        })
    }

    /// Double well potential
    ///
    /// .. math:: V_i^\mathrm{ext}(z)=\mathrm{min}\left(\frac{2\pi}{45} m_i\varepsilon_{2si}\sigma_{si}^3\rho_s\left(2\left(\frac{2\sigma_{si}}{z}\right)^9-15\left(\frac{2\sigma_{si}}{z}\right)^3\right),0\right)+\frac{2\pi}{45} m_i\varepsilon_{1si}\sigma_{si}^3\rho_s\left(2\left(\frac{\sigma_{si}}{z}\right)^9-15\left(\frac{\sigma_{si}}{z}\right)^3\right),~~~~\varepsilon_{1si}=\sqrt{\varepsilon_{1ss}\varepsilon_{ii}},~~~~\varepsilon_{2si}=\sqrt{\varepsilon_{2ss}\varepsilon_{ii}},~~~~\sigma_{si}=\frac{1}{2}\left(\sigma_{ss}+\sigma_{ii}\right)
    ///
    /// Parameters
    /// ----------
    /// sigma_ss : float
    ///     Segment diameter of the solid.
    /// epsilon1_k_ss : float
    ///     Energy parameter of the first well.
    /// epsilon2_k_ss : float
    ///     Energy parameter of the second well.
    /// rho_s : float
    ///     Density of the solid.
    ///
    /// Returns
    /// -------
    /// ExternalPotential
    ///
    #[staticmethod]
    pub fn DoubleWell(sigma_ss: f64, epsilon1_k_ss: f64, epsilon2_k_ss: f64, rho_s: f64) -> Self {
        Self(ExternalPotential::DoubleWell {
            sigma_ss,
            epsilon1_k_ss,
            epsilon2_k_ss,
            rho_s,
        })
    }

    /// Free-energy averaged potential
    ///
    /// for details see: `J. Eller, J. Gross (2021) <https://pubs.acs.org/doi/abs/10.1021/acs.langmuir.0c03287>`_
    ///
    /// Parameters
    /// ----------
    /// coordinates: SIArray2
    ///     The positions of all interaction sites in the solid.
    /// sigma_ss : numpy.ndarray[float]
    ///     The size parameters of all interaction sites.
    /// epsilon_k_ss : numpy.ndarray[float]
    ///     The energy parameter of all interaction sites.
    /// pore_center : [SINumber; 3]
    ///     The cartesian coordinates of the center of the pore
    /// system_size : [SINumber; 3]
    ///     The size of the unit cell.
    /// n_grid : [int; 2]
    ///     The number of grid points in each direction.
    /// cutoff_radius : float, optional
    ///     The cutoff used in the calculation of fluid/wall interactions.
    /// Returns
    /// -------
    /// ExternalPotential
    ///
    #[staticmethod]
    #[pyo3(
        text_signature = "(coordinates, sigma_ss, epsilon_k_ss, pore_center, system_size, n_grid, cutoff_radius=None)"
    )]
    pub fn FreeEnergyAveraged(
        coordinates: PySIArray2,
        sigma_ss: &PyArray1<f64>,
        epsilon_k_ss: &PyArray1<f64>,
        pore_center: [f64; 3],
        system_size: [PySINumber; 3],
        n_grid: [usize; 2],
        cutoff_radius: Option<f64>,
    ) -> PyResult<Self> {
        Ok(Self(ExternalPotential::FreeEnergyAveraged {
            coordinates: coordinates.try_into()?,
            sigma_ss: sigma_ss.to_owned_array(),
            epsilon_k_ss: epsilon_k_ss.to_owned_array(),
            pore_center,
            system_size: [
                system_size[0].try_into()?,
                system_size[1].try_into()?,
                system_size[2].try_into()?,
            ],
            n_grid,
            cutoff_radius,
        }))
    }
}
