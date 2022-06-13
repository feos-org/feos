//! Density profiles at planar interfaces and interfacial tensions.
use crate::convolver::ConvolverFFT;
use crate::functional::{HelmholtzEnergyFunctional, DFT};
use crate::geometry::{Axis, Grid};
use crate::profile::{DFTProfile, DFTSpecifications};
use crate::solver::DFTSolver;
use feos_core::{Contributions, EosError, EosResult, EosUnit, PhaseEquilibrium};
use ndarray::{s, Array, Array1, Array2, Axis as Axis_nd, Ix1};
use quantity::{QuantityArray1, QuantityArray2, QuantityScalar};

mod surface_tension_diagram;
pub use surface_tension_diagram::SurfaceTensionDiagram;

const RELATIVE_WIDTH: f64 = 6.0;
const MIN_WIDTH: f64 = 100.0;

/// Density profile and properties of a planar interface.
pub struct PlanarInterface<U: EosUnit, F: HelmholtzEnergyFunctional> {
    pub profile: DFTProfile<U, Ix1, F>,
    pub vle: PhaseEquilibrium<U, DFT<F>, 2>,
    pub surface_tension: Option<QuantityScalar<U>>,
    pub equimolar_radius: Option<QuantityScalar<U>>,
}

impl<U: EosUnit, F: HelmholtzEnergyFunctional> Clone for PlanarInterface<U, F> {
    fn clone(&self) -> Self {
        Self {
            profile: self.profile.clone(),
            vle: self.vle.clone(),
            surface_tension: self.surface_tension,
            equimolar_radius: self.equimolar_radius,
        }
    }
}

impl<U: EosUnit, F: HelmholtzEnergyFunctional> PlanarInterface<U, F> {
    pub fn solve_inplace(&mut self, solver: Option<&DFTSolver>, debug: bool) -> EosResult<()> {
        // Solve the profile
        self.profile.solve(solver, debug)?;

        // postprocess
        self.surface_tension = Some(self.profile.integrate(
            &(self.profile.grand_potential_density()?
                + self.vle.vapor().pressure(Contributions::Total)),
        ));
        let delta_rho = self.vle.liquid().density - self.vle.vapor().density;
        self.equimolar_radius = Some(
            self.profile
                .integrate(&(self.profile.density.sum_axis(Axis_nd(0)) - self.vle.vapor().density))
                / delta_rho,
        );

        Ok(())
    }

    pub fn solve(mut self, solver: Option<&DFTSolver>) -> EosResult<Self> {
        self.solve_inplace(solver, false)?;
        Ok(self)
    }
}

impl<U: EosUnit, F: HelmholtzEnergyFunctional> PlanarInterface<U, F> {
    pub fn new(
        vle: &PhaseEquilibrium<U, DFT<F>, 2>,
        n_grid: usize,
        l_grid: QuantityScalar<U>,
    ) -> EosResult<Self> {
        let dft = &vle.vapor().eos;

        // generate grid
        let grid = Grid::Cartesian1(Axis::new_cartesian(n_grid, l_grid, None)?);

        // initialize convolver
        let t = vle
            .vapor()
            .temperature
            .to_reduced(U::reference_temperature())?;
        let weight_functions = dft.weight_functions(t);
        let convolver = ConvolverFFT::plan(&grid, &weight_functions, None);

        Ok(Self {
            profile: DFTProfile::new(grid, convolver, vle.vapor(), None, None)?,
            vle: vle.clone(),
            surface_tension: None,
            equimolar_radius: None,
        })
    }

    pub fn from_tanh(
        vle: &PhaseEquilibrium<U, DFT<F>, 2>,
        n_grid: usize,
        l_grid: QuantityScalar<U>,
        critical_temperature: QuantityScalar<U>,
    ) -> EosResult<Self> {
        let mut profile = Self::new(vle, n_grid, l_grid)?;

        // calculate segment indices
        let indices = &profile.profile.dft.component_index();

        // calculate density profile
        let z0 = 0.5 * l_grid.to_reduced(U::reference_length())?;
        let (z0, sign) = (z0.abs(), -z0.signum());
        let reduced_temperature = vle.vapor().temperature.to_reduced(critical_temperature)?;
        profile.profile.density =
            QuantityArray2::from_shape_fn(profile.profile.density.raw_dim(), |(i, z)| {
                let rho_v = profile.vle.vapor().partial_density.get(indices[i]);
                let rho_l = profile.vle.liquid().partial_density.get(indices[i]);
                0.5 * (rho_l - rho_v)
                    * (sign * (profile.profile.grid.grids()[0][z] - z0) / 3.0
                        * (2.4728 - 2.3625 * reduced_temperature))
                        .tanh()
                    + 0.5 * (rho_l + rho_v)
            });

        // specify specification
        profile.profile.specification =
            DFTSpecifications::total_moles_from_profile(&profile.profile)?;

        Ok(profile)
    }

    pub fn from_pdgt(vle: &PhaseEquilibrium<U, DFT<F>, 2>, n_grid: usize) -> EosResult<Self> {
        let dft = &vle.vapor().eos;

        if dft.component_index().len() != 1 {
            panic!("Initialization from pDGT not possible for segment DFT or mixtures");
        }

        // calculate density profile from pDGT
        let n_grid_pdgt = 20;
        let mut z_pdgt = Array1::zeros(n_grid_pdgt) * U::reference_length();
        let mut w_pdgt = U::reference_length();
        let (rho_pdgt, gamma_pdgt) =
            dft.solve_pdgt(vle, 20, 0, Some((&mut z_pdgt, &mut w_pdgt)))?;
        if !gamma_pdgt
            .to_reduced(U::reference_surface_tension())?
            .is_normal()
        {
            return Err(EosError::InvalidState(
                String::from("DFTProfile::from_pdgt"),
                String::from("gamma_pdgt"),
                gamma_pdgt.to_reduced(U::reference_surface_tension())?,
            ));
        }

        // create PlanarInterface
        let l_grid = (MIN_WIDTH * U::reference_length())
            .max(w_pdgt * RELATIVE_WIDTH)
            .unwrap();
        let mut profile = Self::new(vle, n_grid, l_grid)?;

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
        profile.profile.specification =
            DFTSpecifications::total_moles_from_profile(&profile.profile)?;

        Ok(profile)
    }
}

impl<U: EosUnit, F: HelmholtzEnergyFunctional> PlanarInterface<U, F> {
    pub fn shift_equimolar_inplace(&mut self) {
        let s = self.profile.density.shape();
        let m = &self.profile.dft.m();
        let mut rho_l = 0.0 * U::reference_density();
        let mut rho_v = 0.0 * U::reference_density();
        let mut rho = Array::zeros(s[1]) * U::reference_density();
        for i in 0..s[0] {
            rho_l += self.profile.density.get((i, 0)) * m[i];
            rho_v += self.profile.density.get((i, s[1] - 1)) * m[i];
            rho += &(&self.profile.density.index_axis(Axis_nd(0), i) * m[i]);
        }

        let x = (rho - rho_v) / (rho_l - rho_v);
        let ze = self.profile.grid.axes()[0].edges[0]
            + self
                .profile
                .integrate(&x)
                .to_reduced(U::reference_length())
                .unwrap();
        self.profile.grid.axes_mut()[0].grid -= ze;
    }

    pub fn shift_equimolar(mut self) -> Self {
        self.shift_equimolar_inplace();
        self
    }

    fn set_density_scale(&mut self, init: &QuantityArray2<U>) {
        assert_eq!(self.profile.density.shape(), init.shape());
        let n_grid = self.profile.density.shape()[1];
        let drho_init = &init.index_axis(Axis_nd(1), 0) - &init.index_axis(Axis_nd(1), n_grid - 1);
        let rho_init_0 = init.index_axis(Axis_nd(1), n_grid - 1);
        let drho = &self.profile.density.index_axis(Axis_nd(1), 0)
            - &self.profile.density.index_axis(Axis_nd(1), n_grid - 1);
        let rho_0 = self.profile.density.index_axis(Axis_nd(1), n_grid - 1);

        self.profile.density =
            QuantityArray2::from_shape_fn(self.profile.density.raw_dim(), |(i, j)| {
                (init.get((i, j)) - rho_init_0.get(i))
                    .to_reduced(drho_init.get(i))
                    .unwrap()
                    * drho.get(i)
                    + rho_0.get(i)
            });
    }

    pub fn set_density_inplace(&mut self, init: &QuantityArray2<U>, scale: bool) {
        if scale {
            self.set_density_scale(init)
        } else {
            assert_eq!(self.profile.density.shape(), init.shape());
            self.profile.density = init.clone();
        }
    }

    pub fn set_density(mut self, init: &QuantityArray2<U>, scale: bool) -> Self {
        self.set_density_inplace(init, scale);
        self
    }
}

fn interp_symmetric<U: EosUnit, F: HelmholtzEnergyFunctional>(
    vle_pdgt: &PhaseEquilibrium<U, DFT<F>, 2>,
    z_pdgt: QuantityArray1<U>,
    rho_pdgt: QuantityArray2<U>,
    vle: &PhaseEquilibrium<U, DFT<F>, 2>,
    z: &Array1<f64>,
    radius: QuantityScalar<U>,
) -> EosResult<QuantityArray2<U>> {
    let reduced_density = Array2::from_shape_fn(rho_pdgt.raw_dim(), |(i, j)| {
        (rho_pdgt.get((i, j)) - vle_pdgt.vapor().partial_density.get(i))
            .to_reduced(
                vle_pdgt.liquid().partial_density.get(i) - vle_pdgt.vapor().partial_density.get(i),
            )
            .unwrap()
            - 0.5
    });
    let segments = vle_pdgt.vapor().eos.component_index().len();
    let mut reduced_density = interp(
        &z_pdgt.to_reduced(U::reference_length())?,
        &reduced_density,
        &(z - radius.to_reduced(U::reference_length())?),
        &Array1::from_elem(segments, 0.5),
        &Array1::from_elem(segments, -0.5),
        false,
    ) + interp(
        &z_pdgt.to_reduced(U::reference_length())?,
        &reduced_density,
        &(z + radius.to_reduced(U::reference_length())?),
        &Array1::from_elem(segments, -0.5),
        &Array1::from_elem(segments, 0.5),
        true,
    );
    if radius < 0.0 * U::reference_length() {
        reduced_density += 1.0;
    }
    Ok(QuantityArray2::from_shape_fn(
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
