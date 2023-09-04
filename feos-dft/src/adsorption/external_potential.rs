#[cfg(feature = "rayon")]
use crate::adsorption::fea_potential::calculate_fea_potential;
use crate::functional::HelmholtzEnergyFunctional;
#[cfg(feature = "rayon")]
use crate::geometry::Geometry;
#[cfg(feature = "rayon")]
use feos_core::si::Length;
use libm::tgamma;
use ndarray::{Array1, Array2, Axis as Axis_nd};
use std::f64::consts::PI;

const DELTA_STEELE: f64 = 3.35;

/// A collection of external potentials.
#[derive(Clone)]
pub enum ExternalPotential {
    /// Hard wall potential: $V_i^\mathrm{ext}(z)=\begin{cases}\infty&z\leq\sigma_{si}\\\\0&z>\sigma_{si}\end{cases},~~~~\sigma_{si}=\frac{1}{2}\left(\sigma_{ss}+\sigma_{ii}\right)$
    HardWall { sigma_ss: f64 },
    /// 9-3 Lennard-Jones potential: $V_i^\mathrm{ext}(z)=\frac{2\pi}{45} m_i\varepsilon_{si}\sigma_{si}^3\rho_s\left(2\left(\frac{\sigma_{si}}{z}\right)^9-15\left(\frac{\sigma_{si}}{z}\right)^3\right),~~~~\varepsilon_{si}=\sqrt{\varepsilon_{ss}\varepsilon_{ii}},~~~~\sigma_{si}=\frac{1}{2}\left(\sigma_{ss}+\sigma_{ii}\right)$
    LJ93 {
        sigma_ss: f64,
        epsilon_k_ss: f64,
        rho_s: f64,
    },
    /// Simple 9-3 Lennard-Jones potential: $V_i^\mathrm{ext}(z)=\varepsilon_{si}\left(\left(\frac{\sigma_{si}}{z}\right)^9-\left(\frac{\sigma_{si}}{z}\right)^3\right),~~~~\varepsilon_{si}=\sqrt{\varepsilon_{ss}\varepsilon_{ii}},~~~~\sigma_{si}=\frac{1}{2}\left(\sigma_{ss}+\sigma_{ii}\right)$
    SimpleLJ93 { sigma_ss: f64, epsilon_k_ss: f64 },
    /// Custom 9-3 Lennard-Jones potential: $V_i^\mathrm{ext}(z)=\varepsilon_{si}\left(\left(\frac{\sigma_{si}}{z}\right)^9-\left(\frac{\sigma_{si}}{z}\right)^3\right)$
    CustomLJ93 {
        sigma_sf: Array1<f64>,
        epsilon_k_sf: Array1<f64>,
    },
    /// Steele potential: $V_i^\mathrm{ext}(z)=2\pi m_i\xi\varepsilon_{si}\sigma_{si}^2\Delta\rho_s\left(0.4\left(\frac{\sigma_{si}}{z}\right)^{10}-\left(\frac{\sigma_{si}}{z}\right)^4-\frac{\sigma_{si}^4}{3\Delta\left(z+0.61\Delta\right)^3}\right),~~~~\varepsilon_{si}=\sqrt{\varepsilon_{ss}\varepsilon_{ii}},~~~~\sigma_{si}=\frac{1}{2}\left(\sigma_{ss}+\sigma_{ii}\right),~~~~\Delta=3.35$
    Steele {
        sigma_ss: f64,
        epsilon_k_ss: f64,
        rho_s: f64,
        xi: Option<f64>,
    },
    /// Steele potential with custom combining rules: $V_i^\mathrm{ext}(z)=2\pi m_i\xi\varepsilon_{si}\sigma_{si}^2\Delta\rho_s\left(0.4\left(\frac{\sigma_{si}}{z}\right)^{10}-\left(\frac{\sigma_{si}}{z}\right)^4-\frac{\sigma_{si}^4}{3\Delta\left(z+0.61\Delta\right)^3}\right),~~~~\Delta=3.35$
    CustomSteele {
        sigma_sf: Array1<f64>,
        epsilon_k_sf: Array1<f64>,
        rho_s: f64,
        xi: Option<f64>,
    },
    /// Double well potential: $V_i^\mathrm{ext}(z)=\mathrm{min}\left(\frac{2\pi}{45} m_i\varepsilon_{2si}\sigma_{si}^3\rho_s\left(2\left(\frac{2\sigma_{si}}{z}\right)^9-15\left(\frac{2\sigma_{si}}{z}\right)^3\right),0\right)+\frac{2\pi}{45} m_i\varepsilon_{1si}\sigma_{si}^3\rho_s\left(2\left(\frac{\sigma_{si}}{z}\right)^9-15\left(\frac{\sigma_{si}}{z}\right)^3\right),~~~~\varepsilon_{1si}=\sqrt{\varepsilon_{1ss}\varepsilon_{ii}},~~~~\varepsilon_{2si}=\sqrt{\varepsilon_{2ss}\varepsilon_{ii}},~~~~\sigma_{si}=\frac{1}{2}\left(\sigma_{ss}+\sigma_{ii}\right)$
    DoubleWell {
        sigma_ss: f64,
        epsilon1_k_ss: f64,
        epsilon2_k_ss: f64,
        rho_s: f64,
    },
    /// Free-energy averaged potential:
    #[cfg(feature = "rayon")]
    FreeEnergyAveraged {
        coordinates: Length<Array2<f64>>,
        sigma_ss: Array1<f64>,
        epsilon_k_ss: Array1<f64>,
        pore_center: [f64; 3],
        system_size: [Length; 3],
        n_grid: [usize; 2],
        cutoff_radius: Option<f64>,
    },

    /// Custom potential
    Custom(Array2<f64>),
}

/// Parameters of the fluid required to evaluate the external potential.
pub trait FluidParameters: HelmholtzEnergyFunctional {
    fn epsilon_k_ff(&self) -> Array1<f64>;
    fn sigma_ff(&self) -> &Array1<f64>;
}

#[allow(unused_variables)]
impl ExternalPotential {
    // Evaluate the external potential in cartesian coordinates for a given grid and fluid parameters.
    pub fn calculate_cartesian_potential<P: FluidParameters>(
        &self,
        z_grid: &Array1<f64>,
        fluid_parameters: &P,
        temperature: f64,
    ) -> Array2<f64> {
        if let ExternalPotential::Custom(potential) = self {
            return potential.clone();
        }

        // Allocate external potential
        let m = fluid_parameters.m();
        let mut ext_pot = Array2::zeros((m.len(), z_grid.len()));

        for (i, &mi) in m.iter().enumerate() {
            ext_pot.index_axis_mut(Axis_nd(0), i).assign(&match self {
                Self::HardWall { sigma_ss } => {
                    let sigma_sf = (fluid_parameters.sigma_ff()[i] + *sigma_ss) * 0.5;
                    z_grid.mapv(|z| if z < sigma_sf { f64::INFINITY } else { 0.0 })
                }
                Self::LJ93 {
                    sigma_ss,
                    epsilon_k_ss,
                    rho_s,
                } => {
                    // combining rules
                    let epsilon_k_sf =
                        (fluid_parameters.epsilon_k_ff() * *epsilon_k_ss).mapv(|e| e.sqrt());
                    let sigma_sf = (fluid_parameters.sigma_ff() + *sigma_ss) * 0.5;

                    2.0 * PI * mi * epsilon_k_sf[i] * sigma_sf[i].powi(3) * rho_s / 45.0
                        * (2.0 * (sigma_sf[i] / z_grid).mapv(|x| x.powi(9))
                            - 15.0 * (sigma_sf[i] / z_grid).mapv(|x| x.powi(3)))
                }
                Self::SimpleLJ93 {
                    sigma_ss,
                    epsilon_k_ss,
                } => {
                    // combining rules
                    let epsilon_k_sf =
                        (fluid_parameters.epsilon_k_ff() * *epsilon_k_ss).mapv(|e| e.sqrt());
                    let sigma_sf = (fluid_parameters.sigma_ff() + *sigma_ss) * 0.5;

                    epsilon_k_sf[i]
                        * ((sigma_sf[i] / z_grid).mapv(|x| x.powi(9))
                            - (sigma_sf[i] / z_grid).mapv(|x| x.powi(3)))
                }
                Self::CustomLJ93 {
                    sigma_sf,
                    epsilon_k_sf,
                } => {
                    epsilon_k_sf[i]
                        * ((sigma_sf[i] / z_grid).mapv(|x| x.powi(9))
                            - (sigma_sf[i] / z_grid).mapv(|x| x.powi(3)))
                }
                Self::Steele {
                    sigma_ss,
                    epsilon_k_ss,
                    rho_s,
                    xi,
                } => {
                    // combining rules
                    let epsilon_k_sf =
                        (fluid_parameters.epsilon_k_ff() * *epsilon_k_ss).mapv(|e| e.sqrt());
                    let sigma_sf = (fluid_parameters.sigma_ff() + *sigma_ss) * 0.5;

                    (2.0 * PI * mi * xi.unwrap_or(1.0) * epsilon_k_sf[i])
                        * (sigma_sf[i].powi(2) * DELTA_STEELE * rho_s)
                        * (0.4 * (sigma_sf[i] / z_grid).mapv(|x| x.powi(10))
                            - (sigma_sf[i] / z_grid).mapv(|x| x.powi(4))
                            - sigma_sf[i].powi(4)
                                / ((3.0 * DELTA_STEELE)
                                    * (z_grid + 0.61 * DELTA_STEELE).mapv(|x| x.powi(3))))
                }
                Self::CustomSteele {
                    sigma_sf,
                    epsilon_k_sf,
                    rho_s,
                    xi,
                } => {
                    (2.0 * PI * mi * xi.unwrap_or(1.0) * epsilon_k_sf[i])
                        * (sigma_sf[i].powi(2) * DELTA_STEELE * rho_s)
                        * (0.4 * (sigma_sf[i] / z_grid).mapv(|x| x.powi(10))
                            - (sigma_sf[i] / z_grid).mapv(|x| x.powi(4))
                            - sigma_sf[i].powi(4)
                                / ((3.0 * DELTA_STEELE)
                                    * (z_grid + 0.61 * DELTA_STEELE).mapv(|x| x.powi(3))))
                }
                Self::DoubleWell {
                    sigma_ss,
                    epsilon1_k_ss,
                    epsilon2_k_ss,
                    rho_s,
                } => {
                    // combining rules
                    let epsilon1_k_sf =
                        (fluid_parameters.epsilon_k_ff() * *epsilon1_k_ss).mapv(|e| e.sqrt());
                    let epsilon2_k_sf =
                        (fluid_parameters.epsilon_k_ff() * *epsilon2_k_ss).mapv(|e| e.sqrt());
                    let sigma_sf = (fluid_parameters.sigma_ff() + *sigma_ss) * 0.5;

                    let bh_tail = (2.0 * PI * mi * epsilon2_k_sf[i] * sigma_sf[i].powi(3) * rho_s
                        / 45.0
                        * (2.0 * (2.0 * sigma_sf[i] / z_grid).mapv(|x| x.powi(9))
                            - 15.0 * (2.0 * sigma_sf[i] / z_grid).mapv(|x| x.powi(3))))
                    .mapv(|x| x.min(0.0));

                    bh_tail
                        + &(2.0 * PI * mi * epsilon1_k_sf[i] * sigma_sf[i].powi(3) * rho_s / 45.0
                            * (2.0 * (sigma_sf[i] / z_grid).mapv(|x| x.powi(9))
                                - 15.0 * (sigma_sf[i] / z_grid).mapv(|x| x.powi(3))))
                }
                #[cfg(feature = "rayon")]
                Self::FreeEnergyAveraged {
                    coordinates,
                    sigma_ss,
                    epsilon_k_ss,
                    pore_center,
                    system_size,
                    n_grid,
                    cutoff_radius,
                } => {
                    // combining rules
                    let epsilon_k_sf =
                        (fluid_parameters.epsilon_k_ff()[i] * epsilon_k_ss).mapv(|e| e.sqrt());
                    let sigma_sf = (fluid_parameters.sigma_ff()[i] + sigma_ss) * 0.5;

                    calculate_fea_potential(
                        z_grid,
                        mi,
                        coordinates,
                        sigma_sf,
                        epsilon_k_sf,
                        pore_center,
                        system_size,
                        n_grid,
                        temperature,
                        Geometry::Cartesian,
                        *cutoff_radius,
                    )
                }
                _ => unreachable!(),
            });
        }
        ext_pot
    }

    // Evaluate the external potential in cylindrical coordinates for a given grid and fluid parameters.
    pub fn calculate_cylindrical_potential<P: FluidParameters>(
        &self,
        r_grid: &Array1<f64>,
        pore_size: f64,
        fluid_parameters: &P,
        temperature: f64,
    ) -> Array2<f64> {
        if let ExternalPotential::Custom(potential) = self {
            return potential.clone();
        }

        // Allocate external potential
        let m = fluid_parameters.m();
        let mut ext_pot = Array2::zeros((m.len(), r_grid.len()));

        for (i, &mi) in m.iter().enumerate() {
            ext_pot.index_axis_mut(Axis_nd(0), i).assign(&match self {
                Self::HardWall { sigma_ss } => {
                    let sigma_sf = (fluid_parameters.sigma_ff()[i] + *sigma_ss) * 0.5;
                    r_grid.mapv(|r| {
                        if r > pore_size - sigma_sf {
                            f64::INFINITY
                        } else {
                            0.0
                        }
                    })
                }
                Self::LJ93 {
                    sigma_ss,
                    epsilon_k_ss,
                    rho_s,
                } => {
                    // combining rules
                    let epsilon_k_sf =
                        (fluid_parameters.epsilon_k_ff() * *epsilon_k_ss).mapv(|e| e.sqrt());
                    let sigma_sf = (fluid_parameters.sigma_ff() + *sigma_ss) * 0.5;

                    (phi(6, &(r_grid / pore_size), sigma_sf[i] / pore_size)
                        - phi(3, &(r_grid / pore_size), sigma_sf[i] / pore_size))
                        * 2.0
                        * PI
                        * mi
                        * epsilon_k_sf[i]
                        * sigma_sf[i].powi(3)
                        * *rho_s
                }
                Self::SimpleLJ93 {
                    sigma_ss: _,
                    epsilon_k_ss: _,
                } => {
                    unimplemented!()
                }
                Self::CustomLJ93 {
                    sigma_sf: _,
                    epsilon_k_sf: _,
                } => {
                    unimplemented!()
                }
                Self::Steele {
                    sigma_ss,
                    epsilon_k_ss,
                    rho_s,
                    xi,
                } => {
                    // combining rules
                    let epsilon_k_sf =
                        (fluid_parameters.epsilon_k_ff() * *epsilon_k_ss).mapv(|e| e.sqrt());
                    let sigma_sf = (fluid_parameters.sigma_ff() + *sigma_ss) * 0.5;

                    (2.0 * PI * mi * xi.unwrap_or(1.0) * epsilon_k_sf[i])
                        * (sigma_sf[i].powi(2) * DELTA_STEELE * rho_s)
                        * (psi(6, &(r_grid / pore_size), sigma_sf[i] / pore_size)
                            - psi(3, &(r_grid / pore_size), sigma_sf[i] / pore_size)
                            - sigma_sf[i] / DELTA_STEELE
                                * phi(
                                    3,
                                    &(r_grid / (pore_size + DELTA_STEELE * 0.61)),
                                    sigma_sf[i] / (pore_size + DELTA_STEELE * 0.61),
                                ))
                }
                Self::CustomSteele {
                    sigma_sf,
                    epsilon_k_sf,
                    rho_s,
                    xi,
                } => {
                    (2.0 * PI * mi * xi.unwrap_or(1.0) * epsilon_k_sf[i])
                        * (sigma_sf[i].powi(2) * DELTA_STEELE * rho_s)
                        * (psi(6, &(r_grid / pore_size), sigma_sf[i] / pore_size)
                            - psi(3, &(r_grid / pore_size), sigma_sf[i] / pore_size)
                            - sigma_sf[i] / DELTA_STEELE
                                * phi(
                                    3,
                                    &(r_grid / (pore_size + DELTA_STEELE * 0.61)),
                                    sigma_sf[i] / (pore_size + DELTA_STEELE * 0.61),
                                ))
                }
                Self::DoubleWell {
                    sigma_ss,
                    epsilon1_k_ss,
                    epsilon2_k_ss,
                    rho_s,
                } => {
                    // combining rules
                    let epsilon1_k_sf =
                        (fluid_parameters.epsilon_k_ff() * *epsilon1_k_ss).mapv(|e| e.sqrt());
                    let epsilon2_k_sf =
                        (fluid_parameters.epsilon_k_ff() * *epsilon2_k_ss).mapv(|e| e.sqrt());
                    let sigma_sf = (fluid_parameters.sigma_ff() + *sigma_ss) * 0.5;

                    let bh_tail = ((phi(6, &(r_grid / pore_size), 2.0 * sigma_sf[i] / pore_size)
                        - phi(3, &(r_grid / pore_size), 2.0 * sigma_sf[i] / pore_size))
                        * 2.0
                        * PI
                        * mi
                        * epsilon2_k_sf[i]
                        * sigma_sf[i].powi(3)
                        * *rho_s)
                        .mapv(|x| x.min(0.0));

                    bh_tail
                        + &((phi(6, &(r_grid / pore_size), sigma_sf[i] / pore_size)
                            - phi(3, &(r_grid / pore_size), sigma_sf[i] / pore_size))
                            * 2.0
                            * PI
                            * mi
                            * epsilon1_k_sf[i]
                            * sigma_sf[i].powi(3)
                            * *rho_s)
                }
                #[cfg(feature = "rayon")]
                Self::FreeEnergyAveraged {
                    coordinates,
                    sigma_ss,
                    epsilon_k_ss,
                    pore_center,
                    system_size,
                    n_grid,
                    cutoff_radius,
                } => {
                    // combining rules
                    let epsilon_k_sf =
                        (fluid_parameters.epsilon_k_ff()[i] * epsilon_k_ss).mapv(|e| e.sqrt());
                    let sigma_sf = (fluid_parameters.sigma_ff()[i] + sigma_ss) * 0.5;

                    calculate_fea_potential(
                        r_grid,
                        mi,
                        coordinates,
                        sigma_sf,
                        epsilon_k_sf,
                        pore_center,
                        system_size,
                        n_grid,
                        temperature,
                        Geometry::Cylindrical,
                        *cutoff_radius,
                    )
                }
                _ => unreachable!(),
            });
        }
        ext_pot
    }

    // Evaluate the external potential in spherical coordinates for a given grid and fluid parameters.
    pub fn calculate_spherical_potential<P: FluidParameters>(
        &self,
        r_grid: &Array1<f64>,
        pore_size: f64,
        fluid_parameters: &P,
        temperature: f64,
    ) -> Array2<f64> {
        if let ExternalPotential::Custom(potential) = self {
            return potential.clone();
        }

        // Allocate external potential
        let m = fluid_parameters.m();
        let mut ext_pot = Array2::zeros((m.len(), r_grid.len()));

        for (i, &mi) in m.iter().enumerate() {
            ext_pot.index_axis_mut(Axis_nd(0), i).assign(&match self {
                Self::HardWall { sigma_ss } => {
                    let sigma_sf = (fluid_parameters.sigma_ff()[i] + *sigma_ss) * 0.5;
                    r_grid.mapv(|r| {
                        if r > pore_size - sigma_sf {
                            f64::INFINITY
                        } else {
                            0.0
                        }
                    })
                }
                Self::LJ93 {
                    sigma_ss,
                    epsilon_k_ss,
                    rho_s,
                } => {
                    // combining rules
                    let epsilon_k_sf =
                        (fluid_parameters.epsilon_k_ff() * *epsilon_k_ss).mapv(|e| e.sqrt());
                    let sigma_sf = (fluid_parameters.sigma_ff() + *sigma_ss) * 0.5;

                    PI * mi
                        * epsilon_k_sf[i]
                        * rho_s
                        * (sigma_sf[i].powi(12) / 90.
                            * ((r_grid - 9.0 * pore_size)
                                / (r_grid - pore_size).mapv(|x| x.powi(9))
                                - (r_grid + 9.0 * pore_size)
                                    / (r_grid + pore_size).mapv(|x| x.powi(9)))
                            - sigma_sf[i].powi(6) / 3.
                                * ((r_grid - 3.0 * pore_size)
                                    / (r_grid - pore_size).mapv(|x| x.powi(3))
                                    - (r_grid + 3.0 * pore_size)
                                        / (r_grid + pore_size).mapv(|x| x.powi(3))))
                        / r_grid
                }
                Self::SimpleLJ93 {
                    sigma_ss: _,
                    epsilon_k_ss: _,
                } => {
                    unimplemented!()
                }
                Self::CustomLJ93 {
                    sigma_sf: _,
                    epsilon_k_sf: _,
                } => {
                    unimplemented!()
                }
                Self::Steele {
                    sigma_ss,
                    epsilon_k_ss,
                    rho_s,
                    xi,
                } => {
                    // combining rules
                    let epsilon_k_sf =
                        (fluid_parameters.epsilon_k_ff() * *epsilon_k_ss).mapv(|e| e.sqrt());
                    let sigma_sf = (fluid_parameters.sigma_ff() + *sigma_ss) * 0.5;

                    (2.0 * PI * mi * xi.unwrap_or(1.0) * epsilon_k_sf[i])
                        * (sigma_sf[i].powi(2) * DELTA_STEELE * rho_s)
                        * (2.0 / 5.0 * sum_n(10, r_grid, sigma_sf[i], pore_size)
                            - sum_n(4, r_grid, sigma_sf[i], pore_size)
                            - sigma_sf[i] / (3.0 * DELTA_STEELE)
                                * (sigma_sf[i].powi(3)
                                    / r_grid
                                        .mapv(|r| (pore_size + 0.61 * DELTA_STEELE - r).powi(3))
                                    + sigma_sf[i].powi(3)
                                        / r_grid.mapv(|r| {
                                            (pore_size + 0.61 * DELTA_STEELE + r).powi(3)
                                        })
                                    + 1.5
                                        * sum_n(
                                            3,
                                            r_grid,
                                            sigma_sf[i],
                                            pore_size + 0.61 * DELTA_STEELE,
                                        )))
                }
                Self::CustomSteele {
                    sigma_sf,
                    epsilon_k_sf,
                    rho_s,
                    xi,
                } => {
                    (2.0 * PI * mi * xi.unwrap_or(1.0) * epsilon_k_sf[i])
                        * (sigma_sf[i].powi(2) * DELTA_STEELE * rho_s)
                        * (2.0 / 5.0 * sum_n(10, r_grid, sigma_sf[i], pore_size)
                            - sum_n(4, r_grid, sigma_sf[i], pore_size)
                            - sigma_sf[i] / (3.0 * DELTA_STEELE)
                                * (sigma_sf[i].powi(3)
                                    / r_grid
                                        .mapv(|r| (pore_size + 0.61 * DELTA_STEELE - r).powi(3))
                                    + sigma_sf[i].powi(3)
                                        / r_grid.mapv(|r| {
                                            (pore_size + 0.61 * DELTA_STEELE + r).powi(3)
                                        })
                                    + 1.5
                                        * sum_n(
                                            3,
                                            r_grid,
                                            sigma_sf[i],
                                            pore_size + 0.61 * DELTA_STEELE,
                                        )))
                }
                Self::DoubleWell {
                    sigma_ss,
                    epsilon1_k_ss,
                    epsilon2_k_ss,
                    rho_s,
                } => {
                    // combining rules
                    let epsilon1_k_sf =
                        (fluid_parameters.epsilon_k_ff() * *epsilon1_k_ss).mapv(|e| e.sqrt());
                    let epsilon2_k_sf =
                        (fluid_parameters.epsilon_k_ff() * *epsilon2_k_ss).mapv(|e| e.sqrt());
                    let sigma_sf = (fluid_parameters.sigma_ff() + *sigma_ss) * 0.5;

                    let bh_tail = (2.0
                        * PI
                        * mi
                        * epsilon2_k_sf[i]
                        * sigma_sf[i].powi(2)
                        * rho_s
                        * (2.0 / 5.0 * sum_n(10, r_grid, 2.0 * sigma_sf[i], pore_size)
                            - sum_n(4, r_grid, 2.0 * sigma_sf[i], pore_size)))
                    .mapv(|x| x.min(0.0));

                    bh_tail
                        + &(2.0
                            * PI
                            * mi
                            * epsilon1_k_sf[i]
                            * sigma_sf[i].powi(2)
                            * rho_s
                            * (2.0 / 5.0 * sum_n(10, r_grid, sigma_sf[i], pore_size)
                                - sum_n(4, r_grid, sigma_sf[i], pore_size)))
                }
                #[cfg(feature = "rayon")]
                Self::FreeEnergyAveraged {
                    coordinates,
                    sigma_ss,
                    epsilon_k_ss,
                    pore_center,
                    system_size,
                    n_grid,
                    cutoff_radius,
                } => {
                    // combining rules
                    let epsilon_k_sf =
                        (fluid_parameters.epsilon_k_ff()[i] * epsilon_k_ss).mapv(|e| e.sqrt());
                    let sigma_sf = (fluid_parameters.sigma_ff()[i] + sigma_ss) * 0.5;

                    calculate_fea_potential(
                        r_grid,
                        mi,
                        coordinates,
                        sigma_sf,
                        epsilon_k_sf,
                        pore_center,
                        system_size,
                        n_grid,
                        temperature,
                        Geometry::Spherical,
                        *cutoff_radius,
                    )
                }
                _ => unreachable!(),
            });
        }
        ext_pot
    }
}

fn phi(n: i32, r_r: &Array1<f64>, sigma_r: f64) -> Array1<f64> {
    let m3n2 = 3.0 - 2.0 * n as f64;
    let n2m3 = 2.0 * n as f64 - 3.0;

    (1.0 - &r_r.mapv(|r| r.powi(2))).mapv(|r| r.powf(m3n2)) * 4.0 * PI.sqrt() / n2m3
        * sigma_r.powf(n2m3)
        * tgamma(n as f64 - 0.5)
        / tgamma(n as f64)
        * taylor_2f1_phi(r_r, n)
}

fn psi(n: i32, r_r: &Array1<f64>, sigma_r: f64) -> Array1<f64> {
    (1.0 - &r_r.mapv(|r| r.powi(2))).mapv(|r| r.powf(2.0 - 2.0 * n as f64))
        * 4.0
        * PI.sqrt()
        * tgamma(n as f64 - 0.5)
        / tgamma(n as f64)
        * sigma_r.powf(2.0 * n as f64 - 2.0)
        * taylor_2f1_psi(r_r, n)
}

fn sum_n(n: i32, r_grid: &Array1<f64>, sigma: f64, pore_size: f64) -> Array1<f64> {
    let mut result = Array1::zeros(r_grid.len());
    for i in 0..n {
        result = result
            + sigma.powi(n) / (pore_size.powi(i) * r_grid.mapv(|r| (pore_size - r).powi(n - i)))
            + sigma.powi(n) / (pore_size.powi(i) * r_grid.mapv(|r| (pore_size + r).powi(n - i)));
    }
    result
}

fn taylor_2f1_phi(x: &Array1<f64>, n: i32) -> Array1<f64> {
    match n {
        3 => {
            1.0 + 3.0 / 4.0 * x.mapv(|x| x.powi(2)) + 3.0 / 64.0 * x.mapv(|x| x.powi(4))
                - 1.0 / 256.0 * x.mapv(|x| x.powi(6))
                - 15.0 / 16384.0 * x.mapv(|x| x.powi(8))
        }
        6 => {
            1.0 + 63.0 / 4.0 * x.mapv(|x| x.powi(2))
                + 2205.0 / 64.0 * x.mapv(|x| x.powi(4))
                + 3675.0 / 256.0 * x.mapv(|x| x.powi(6))
                + 11025.0 / 16384.0 * x.mapv(|x| x.powi(8))
        }
        _ => unreachable!(),
    }
}

fn taylor_2f1_psi(x: &Array1<f64>, n: i32) -> Array1<f64> {
    match n {
        3 => {
            1.0 + 9.0 / 4.0 * x.mapv(|x| x.powi(2))
                + 9.0 / 64.0 * x.mapv(|x| x.powi(4))
                + 1.0 / 256.0 * x.mapv(|x| x.powi(6))
                + 9.0 / 16384.0 * x.mapv(|x| x.powi(8))
        }
        6 => {
            1.0 + 81.0 / 4.0 * x.mapv(|x| x.powi(2))
                + 3969.0 / 64.0 * x.mapv(|x| x.powi(4))
                + 11025.0 / 256.0 * x.mapv(|x| x.powi(6))
                + 99125.0 / 16384.0 * x.mapv(|x| x.powi(8))
        }
        _ => unreachable!(),
    }
}
