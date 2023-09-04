//! Density profiles at planar interfaces and interfacial tensions.
use crate::convolver::ConvolverFFT;
use crate::functional::{HelmholtzEnergyFunctional, DFT};
use crate::geometry::{Axis, Grid};
use crate::profile::{DFTProfile, DFTSpecifications};
use crate::solver::DFTSolver;
use feos_core::si::{Area, Density, Length, Moles, SurfaceTension, Temperature};
use feos_core::{Contributions, EosError, EosResult, PhaseEquilibrium};
use ndarray::{s, Array1, Array2, Axis as Axis_nd, Ix1};

mod surface_tension_diagram;
pub use surface_tension_diagram::SurfaceTensionDiagram;

const RELATIVE_WIDTH: f64 = 6.0;
const MIN_WIDTH: f64 = 100.0;

/// Density profile and properties of a planar interface.
pub struct PlanarInterface<F: HelmholtzEnergyFunctional> {
    pub profile: DFTProfile<Ix1, F>,
    pub vle: PhaseEquilibrium<DFT<F>, 2>,
    pub surface_tension: Option<SurfaceTension>,
    pub equimolar_radius: Option<Length>,
}

impl<F: HelmholtzEnergyFunctional> Clone for PlanarInterface<F> {
    fn clone(&self) -> Self {
        Self {
            profile: self.profile.clone(),
            vle: self.vle.clone(),
            surface_tension: self.surface_tension,
            equimolar_radius: self.equimolar_radius,
        }
    }
}

impl<F: HelmholtzEnergyFunctional> PlanarInterface<F> {
    pub fn solve_inplace(&mut self, solver: Option<&DFTSolver>, debug: bool) -> EosResult<()> {
        // Solve the profile
        self.profile.solve(solver, debug)?;

        // postprocess
        self.surface_tension = Some(
            (self.profile.integrate(
                &(self.profile.grand_potential_density()?
                    + self.vle.vapor().pressure(Contributions::Total)),
            )) / Area::from_reduced(1.0),
        );
        let delta_rho = self.vle.liquid().density - self.vle.vapor().density;
        self.equimolar_radius = Some(
            self.profile
                .integrate(&(self.profile.density.sum_axis(Axis_nd(0)) - self.vle.vapor().density))
                / delta_rho
                / Area::from_reduced(1.0),
        );

        Ok(())
    }

    pub fn solve(mut self, solver: Option<&DFTSolver>) -> EosResult<Self> {
        self.solve_inplace(solver, false)?;
        Ok(self)
    }
}

impl<F: HelmholtzEnergyFunctional> PlanarInterface<F> {
    pub fn new(vle: &PhaseEquilibrium<DFT<F>, 2>, n_grid: usize, l_grid: Length) -> Self {
        let dft = &vle.vapor().eos;

        // generate grid
        let grid = Grid::Cartesian1(Axis::new_cartesian(n_grid, l_grid, None));

        // initialize convolver
        let t = vle.vapor().temperature.to_reduced();
        let weight_functions = dft.weight_functions(t);
        let convolver = ConvolverFFT::plan(&grid, &weight_functions, None);

        Self {
            profile: DFTProfile::new(grid, convolver, vle.vapor(), None, None),
            vle: vle.clone(),
            surface_tension: None,
            equimolar_radius: None,
        }
    }

    pub fn from_tanh(
        vle: &PhaseEquilibrium<DFT<F>, 2>,
        n_grid: usize,
        l_grid: Length,
        critical_temperature: Temperature,
        fix_equimolar_surface: bool,
    ) -> Self {
        let mut profile = Self::new(vle, n_grid, l_grid);

        // calculate segment indices
        let indices = &profile.profile.dft.component_index();

        // calculate density profile
        let z0 = 0.5 * l_grid.to_reduced();
        let (z0, sign) = (z0.abs(), -z0.signum());
        let reduced_temperature = (vle.vapor().temperature / critical_temperature).into_value();
        profile.profile.density =
            Density::from_shape_fn(profile.profile.density.raw_dim(), |(i, z)| {
                let rho_v = profile.vle.vapor().partial_density.get(indices[i]);
                let rho_l = profile.vle.liquid().partial_density.get(indices[i]);
                0.5 * (rho_l - rho_v)
                    * (sign * (profile.profile.grid.grids()[0][z] - z0) / 3.0
                        * (2.4728 - 2.3625 * reduced_temperature))
                        .tanh()
                    + 0.5 * (rho_l + rho_v)
            });

        // specify specification
        if fix_equimolar_surface {
            profile.profile.specification =
                DFTSpecifications::total_moles_from_profile(&profile.profile);
        }

        profile
    }

    pub fn from_pdgt(
        vle: &PhaseEquilibrium<DFT<F>, 2>,
        n_grid: usize,
        fix_equimolar_surface: bool,
    ) -> EosResult<Self> {
        let dft = &vle.vapor().eos;

        if dft.component_index().len() != 1 {
            panic!("Initialization from pDGT not possible for segment DFT or mixtures");
        }

        // calculate density profile from pDGT
        let n_grid_pdgt = 20;
        let mut z_pdgt = Length::zeros(n_grid_pdgt);
        let mut w_pdgt = Length::from_reduced(0.0);
        let (rho_pdgt, gamma_pdgt) =
            dft.solve_pdgt(vle, 20, 0, Some((&mut z_pdgt, &mut w_pdgt)))?;
        if !gamma_pdgt.to_reduced().is_normal() {
            return Err(EosError::InvalidState(
                String::from("DFTProfile::from_pdgt"),
                String::from("gamma_pdgt"),
                gamma_pdgt.to_reduced(),
            ));
        }

        // create PlanarInterface
        let l_grid = Length::from_reduced(MIN_WIDTH).max(w_pdgt * RELATIVE_WIDTH);
        let mut profile = Self::new(vle, n_grid, l_grid);

        // interpolate density profile from pDGT to DFT
        let r = l_grid * 0.5;
        profile.profile.density = interp_symmetric(
            vle,
            z_pdgt,
            rho_pdgt,
            &profile.vle,
            profile.profile.grid.grids()[0],
            r,
        )?;

        // specify specification
        if fix_equimolar_surface {
            profile.profile.specification =
                DFTSpecifications::total_moles_from_profile(&profile.profile);
        }

        Ok(profile)
    }
}

impl<F: HelmholtzEnergyFunctional> PlanarInterface<F> {
    pub fn shift_equimolar_inplace(&mut self) {
        let s = self.profile.density.shape();
        let m = &self.profile.dft.m();
        let mut rho_l = Density::from_reduced(0.0);
        let mut rho_v = Density::from_reduced(0.0);
        let mut rho = Density::zeros(s[1]);
        for i in 0..s[0] {
            rho_l += self.profile.density.get((i, 0)) * m[i];
            rho_v += self.profile.density.get((i, s[1] - 1)) * m[i];
            rho += &(&self.profile.density.index_axis(Axis_nd(0), i) * m[i]);
        }

        let x = (rho - rho_v) / (rho_l - rho_v);
        let ze = self.profile.grid.axes()[0].edges[0] + self.profile.integrate(&x).to_reduced();
        self.profile.grid.axes_mut()[0].grid -= ze;
    }

    pub fn shift_equimolar(mut self) -> Self {
        self.shift_equimolar_inplace();
        self
    }

    /// Relative adsorption of component `i' with respect to `j': \Gamma_i^(j)
    pub fn relative_adsorption(&self) -> Moles<Array2<f64>> {
        let s = self.profile.density.shape();
        let mut rho_l = Density::zeros(s[0]);
        let mut rho_v = Density::zeros(s[0]);

        // Calculate the partial densities in the liquid and in the vapor phase
        for i in 0..s[0] {
            rho_l.set(i, self.profile.density.get((i, 0)));
            rho_v.set(i, self.profile.density.get((i, s[1] - 1)));
        }

        // Calculate \Gamma_i^(j)
        Moles::from_shape_fn((s[0], s[0]), |(i, j)| {
            if i == j {
                Moles::from_reduced(0.0)
            } else {
                self.profile.integrate(
                    &(-(rho_l.get(i) - rho_v.get(i))
                        * ((&self.profile.density.index_axis(Axis_nd(0), j) - rho_l.get(j))
                            / (rho_l.get(j) - rho_v.get(j))
                            - (&self.profile.density.index_axis(Axis_nd(0), i) - rho_l.get(i))
                                / (rho_l.get(i) - rho_v.get(i)))),
                )
            }
        })
    }

    /// Interfacial enrichment of component `i': E_i
    pub fn interfacial_enrichment(&self) -> Array1<f64> {
        let s = self.profile.density.shape();
        let density = self.profile.density.to_reduced();
        let rho_l = density.index_axis(Axis_nd(1), 0);
        let rho_v = density.index_axis(Axis_nd(1), s[1] - 1);

        Array1::from_shape_fn(s[0], |i| {
            *(density
                .index_axis(Axis_nd(0), i)
                .iter()
                .max_by(|&a, &b| a.total_cmp(b))
                .unwrap())  // panics only of iterator is empty
                / rho_l[i].max(rho_v[i])
        })
    }

    /// Interface thickness (90-10 number density difference)
    pub fn interfacial_thickness(&self) -> EosResult<Length> {
        let s = self.profile.density.shape();
        let rho = self.profile.density.sum_axis(Axis_nd(0)).to_reduced();
        let z = self.profile.grid.grids()[0];
        let dz = z[1] - z[0];

        let limits = (0.9_f64, 0.1_f64);
        let (limit_upper, limit_lower) = if limits.0 > limits.1 {
            (limits.0, limits.1)
        } else {
            (limits.1, limits.0)
        };

        if limit_upper >= 1.0 || limit_upper.is_sign_negative() {
            return Err(EosError::IterationFailed(String::from(
                "Upper limit 'l' of interface thickness needs to satisfy 0 < l < 1.",
            )));
        }
        if limit_lower >= 1.0 || limit_lower.is_sign_negative() {
            return Err(EosError::IterationFailed(String::from(
                "Lower limit 'l' of interface thickness needs to satisfy 0 < l < 1.",
            )));
        }

        // Get the densities in the liquid and in the vapor phase
        let rho_v = rho[0].min(rho[s[1] - 1]);
        let rho_l = rho[0].max(rho[s[1] - 1]);

        if (rho_l - rho_v).abs() < 1.0e-10 {
            return Ok(Length::from_reduced(0.0));
        }

        // Density boundaries for interface definition
        let rho_upper = rho_v + limit_upper * (rho_l - rho_v);
        let rho_lower = rho_v + limit_lower * (rho_l - rho_v);

        // Get indizes right of intersection between density profile and
        // constant density boundaries
        let index_upper_plus = if rho[0] >= rho[s[1] - 1] {
            rho.iter()
                .enumerate()
                .find(|(_, &x)| (x - rho_upper).is_sign_negative())
                .expect("Could not find rho_upper value!")
                .0
        } else {
            rho.iter()
                .enumerate()
                .find(|(_, &x)| (rho_upper - x).is_sign_negative())
                .expect("Could not find rho_upper value!")
                .0
        };
        let index_lower_plus = if rho[0] >= rho[s[1] - 1] {
            rho.iter()
                .enumerate()
                .find(|(_, &x)| (x - rho_lower).is_sign_negative())
                .expect("Could not find rho_lower value!")
                .0
        } else {
            rho.iter()
                .enumerate()
                .find(|(_, &x)| (rho_lower - x).is_sign_negative())
                .expect("Could not find rho_lower value!")
                .0
        };

        // Calculate distance between two density points using a linear
        // interpolated density profiles between the two grid points where the
        // density profile crosses the limiting densities
        let z_upper = z[index_upper_plus - 1]
            + (rho_upper - rho[index_upper_plus - 1])
                / (rho[index_upper_plus] - rho[index_upper_plus - 1])
                * dz;
        let z_lower = z[index_lower_plus - 1]
            + (rho_lower - rho[index_lower_plus - 1])
                / (rho[index_lower_plus] - rho[index_lower_plus - 1])
                * dz;

        // Return
        Ok(Length::from_reduced(z_lower - z_upper))
    }

    fn set_density_scale(&mut self, init: &Density<Array2<f64>>) {
        assert_eq!(self.profile.density.shape(), init.shape());
        let n_grid = self.profile.density.shape()[1];
        let drho_init = &init.index_axis(Axis_nd(1), 0) - &init.index_axis(Axis_nd(1), n_grid - 1);
        let rho_init_0 = init.index_axis(Axis_nd(1), n_grid - 1);
        let drho = &self.profile.density.index_axis(Axis_nd(1), 0)
            - &self.profile.density.index_axis(Axis_nd(1), n_grid - 1);
        let rho_0 = self.profile.density.index_axis(Axis_nd(1), n_grid - 1);

        self.profile.density = Density::from_shape_fn(self.profile.density.raw_dim(), |(i, j)| {
            ((init.get((i, j)) - rho_init_0.get(i)) / drho_init.get(i)).into_value() * drho.get(i)
                + rho_0.get(i)
        });
    }

    pub fn set_density_inplace(&mut self, init: &Density<Array2<f64>>, scale: bool) {
        if scale {
            self.set_density_scale(init)
        } else {
            assert_eq!(self.profile.density.shape(), init.shape());
            self.profile.density = init.clone();
        }
    }

    pub fn set_density(mut self, init: &Density<Array2<f64>>, scale: bool) -> Self {
        self.set_density_inplace(init, scale);
        self
    }
}

fn interp_symmetric<F: HelmholtzEnergyFunctional>(
    vle_pdgt: &PhaseEquilibrium<DFT<F>, 2>,
    z_pdgt: Length<Array1<f64>>,
    rho_pdgt: Density<Array2<f64>>,
    vle: &PhaseEquilibrium<DFT<F>, 2>,
    z: &Array1<f64>,
    radius: Length,
) -> EosResult<Density<Array2<f64>>> {
    let reduced_density = Array2::from_shape_fn(rho_pdgt.raw_dim(), |(i, j)| {
        ((rho_pdgt.get((i, j)) - vle_pdgt.vapor().partial_density.get(i))
            / (vle_pdgt.liquid().partial_density.get(i) - vle_pdgt.vapor().partial_density.get(i)))
        .into_value()
            - 0.5
    });
    let segments = vle_pdgt.vapor().eos.component_index().len();
    let mut reduced_density = interp(
        &z_pdgt.to_reduced(),
        &reduced_density,
        &(z - radius.to_reduced()),
        &Array1::from_elem(segments, 0.5),
        &Array1::from_elem(segments, -0.5),
        false,
    ) + interp(
        &z_pdgt.to_reduced(),
        &reduced_density,
        &(z + radius.to_reduced()),
        &Array1::from_elem(segments, -0.5),
        &Array1::from_elem(segments, 0.5),
        true,
    );
    if radius.is_sign_negative() {
        reduced_density += 1.0;
    }
    Ok(Density::from_shape_fn(
        reduced_density.raw_dim(),
        |(i, j)| {
            reduced_density[(i, j)]
                * (vle.liquid().partial_density.get(i) - vle.vapor().partial_density.get(i))
                + vle.vapor().partial_density.get(i)
        },
    ))
}

fn interp(
    x_old: &Array1<f64>,
    y_old: &Array2<f64>,
    x_new: &Array1<f64>,
    y_left: &Array1<f64>,
    y_right: &Array1<f64>,
    reverse: bool,
) -> Array2<f64> {
    let n = x_old.len();

    let (x_rev, y_rev) = if reverse {
        (-&x_old.slice(s![..;-1]), y_old.slice(s![.., ..;-1]))
    } else {
        (x_old.to_owned(), y_old.view())
    };

    let mut y_new = Array2::zeros((y_rev.shape()[0], x_new.len()));
    let mut k = 0;
    for i in 0..x_new.len() {
        while k < n && x_new[i] > x_rev[k] {
            k += 1;
        }
        y_new.slice_mut(s![.., i]).assign(&if k == 0 {
            y_left
                + &((&y_rev.slice(s![.., 0]) - y_left)
                    * ((&y_rev.slice(s![.., 1]) - y_left) / (&y_rev.slice(s![.., 0]) - y_left))
                        .mapv(|x| x.powf((x_new[i] - x_rev[0]) / (x_rev[1] - x_rev[0]))))
        } else if k == n {
            y_right
                + &((&y_rev.slice(s![.., n - 2]) - y_right)
                    * ((&y_rev.slice(s![.., n - 1]) - y_right)
                        / (&y_rev.slice(s![.., n - 2]) - y_right))
                        .mapv(|x| {
                            x.powf((x_new[i] - x_rev[n - 2]) / (x_rev[n - 1] - x_rev[n - 2]))
                        }))
        } else {
            &y_rev.slice(s![.., k - 1])
                + &((x_new[i] - x_rev[k - 1]) / (x_rev[k] - x_rev[k - 1])
                    * (&y_rev.slice(s![.., k]) - &y_rev.slice(s![.., k - 1])))
        });
    }
    y_new
}
