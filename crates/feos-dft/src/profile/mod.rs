use crate::convolver::{BulkConvolver, Convolver, ConvolverFFT};
use crate::functional::HelmholtzEnergyFunctional;
use crate::geometry::Grid;
use crate::solver::{DFTSolver, DFTSolverLog};
use feos_core::{FeosError, FeosResult, ReferenceSystem, State};
use nalgebra::{DVector, Dyn, U1};
use ndarray::{
    Array, Array1, Array2, Array3, ArrayBase, Axis as Axis_nd, Data, Dimension, Ix1, Ix2, Ix3,
    RemoveAxis,
};
use num_dual::DualNum;
use quantity::{_Volume, DEGREES, Density, Length, Moles, Quantity, Temperature, Volume};
use std::ops::{Add, MulAssign};
use std::sync::Arc;
use typenum::Sum;

mod properties;

pub(crate) const MAX_POTENTIAL: f64 = 50.0;
#[cfg(feature = "rayon")]
pub(crate) const CUTOFF_RADIUS: f64 = 14.0;

/// General specifications for the chemical potential in a DFT calculation.
///
/// In the most basic case, the chemical potential is specified in a DFT calculation,
/// for more general systems, this trait provides the possibility to declare additional
/// equations for the calculation of the chemical potential during the iteration.
pub trait DFTSpecification<D: Dimension, F>: Send + Sync {
    fn calculate_bulk_density(
        &self,
        profile: &DFTProfile<D, F>,
        bulk_density: &Array1<f64>,
        z: &Array1<f64>,
    ) -> FeosResult<Array1<f64>>;
}

/// Common specifications for the grand potentials in a DFT calculation.
pub enum DFTSpecifications {
    /// DFT with specified chemical potential.
    ChemicalPotential,
    /// DFT with specified number of particles.
    ///
    /// The solution is still a grand canonical density profile, but the chemical
    /// potentials are iterated together with the density profile to obtain a result
    /// with the specified number of particles.
    Moles { moles: Array1<f64> },
    /// DFT with specified total number of moles.
    TotalMoles { total_moles: f64 },
}

impl DFTSpecifications {
    /// Calculate the number of particles from the profile.
    ///
    /// Call this after initializing the density profile to keep the number of
    /// particles constant in systems, where the number itself is difficult to obtain.
    pub fn moles_from_profile<D: Dimension, F: HelmholtzEnergyFunctional>(
        profile: &DFTProfile<D, F>,
    ) -> Arc<Self>
    where
        D::Larger: Dimension<Smaller = D>,
    {
        let rho = profile.density.to_reduced();
        Arc::new(Self::Moles {
            moles: profile.integrate_reduced_comp(&rho),
        })
    }

    /// Calculate the number of particles from the profile.
    ///
    /// Call this after initializing the density profile to keep the total number of
    /// particles constant in systems, e.g. to fix the equimolar dividing surface.
    pub fn total_moles_from_profile<D: Dimension, F: HelmholtzEnergyFunctional>(
        profile: &DFTProfile<D, F>,
    ) -> Arc<Self>
    where
        D::Larger: Dimension<Smaller = D>,
    {
        let rho = profile.density.to_reduced();
        let moles = profile.integrate_reduced_comp(&rho).sum();
        Arc::new(Self::TotalMoles { total_moles: moles })
    }
}

impl<D: Dimension, F: HelmholtzEnergyFunctional> DFTSpecification<D, F> for DFTSpecifications {
    fn calculate_bulk_density(
        &self,
        _profile: &DFTProfile<D, F>,
        bulk_density: &Array1<f64>,
        z: &Array1<f64>,
    ) -> FeosResult<Array1<f64>> {
        Ok(match self {
            Self::ChemicalPotential => bulk_density.clone(),
            Self::Moles { moles } => moles / z,
            Self::TotalMoles { total_moles } => {
                bulk_density * *total_moles / (bulk_density * z).sum()
            }
        })
    }
}

/// A one-, two-, or three-dimensional density profile.
#[derive(Clone)]
pub struct DFTProfile<D: Dimension, F> {
    pub grid: Grid,
    pub convolver: Arc<dyn Convolver<f64, D>>,
    pub temperature: Temperature,
    pub density: Density<Array<f64, D::Larger>>,
    pub specification: Arc<dyn DFTSpecification<D, F>>,
    pub external_potential: Array<f64, D::Larger>,
    pub bulk: State<F>,
    pub solver_log: Option<DFTSolverLog>,
    pub lanczos: Option<i32>,
}

impl<F> DFTProfile<Ix1, F> {
    pub fn r(&self) -> Length<Array1<f64>> {
        Length::from_reduced(self.grid.grids()[0].to_owned())
    }

    pub fn z(&self) -> Length<Array1<f64>> {
        Length::from_reduced(self.grid.grids()[0].to_owned())
    }
}

impl<F> DFTProfile<Ix2, F> {
    pub fn edges(&self) -> [Length<Array1<f64>>; 2] {
        [
            Length::from_reduced(self.grid.axes()[0].edges.to_owned()),
            Length::from_reduced(self.grid.axes()[1].edges.to_owned()),
        ]
    }

    pub fn meshgrid(&self) -> [Length<Array2<f64>>; 2] {
        let (u, v, alpha) = match &self.grid {
            Grid::Cartesian2(u, v) => (u, v, 90.0 * DEGREES),
            Grid::Periodical2(u, v, alpha) => (u, v, *alpha),
            _ => unreachable!(),
        };
        let u_grid = Array::from_shape_fn([u.grid.len(), v.grid.len()], |(i, _)| u.grid[i]);
        let v_grid = Array::from_shape_fn([u.grid.len(), v.grid.len()], |(_, j)| v.grid[j]);
        let x = Length::from_reduced(u_grid + &v_grid * alpha.cos());
        let y = Length::from_reduced(v_grid * alpha.sin());
        [x, y]
    }

    pub fn r(&self) -> Length<Array1<f64>> {
        Length::from_reduced(self.grid.grids()[0].to_owned())
    }

    pub fn z(&self) -> Length<Array1<f64>> {
        Length::from_reduced(self.grid.grids()[1].to_owned())
    }
}

impl<F> DFTProfile<Ix3, F> {
    pub fn edges(&self) -> [Length<Array1<f64>>; 3] {
        [
            Length::from_reduced(self.grid.axes()[0].edges.to_owned()),
            Length::from_reduced(self.grid.axes()[1].edges.to_owned()),
            Length::from_reduced(self.grid.axes()[2].edges.to_owned()),
        ]
    }

    pub fn meshgrid(&self) -> [Length<Array3<f64>>; 3] {
        let (u, v, w, [alpha, beta, gamma]) = match &self.grid {
            Grid::Cartesian3(u, v, w) => (u, v, w, [90.0 * DEGREES; 3]),
            Grid::Periodical3(u, v, w, angles) => (u, v, w, *angles),
            _ => unreachable!(),
        };
        let shape = [u.grid.len(), v.grid.len(), w.grid.len()];
        let u_grid = Array::from_shape_fn(shape, |(i, _, _)| u.grid[i]);
        let v_grid = Array::from_shape_fn(shape, |(_, j, _)| v.grid[j]);
        let w_grid = Array::from_shape_fn(shape, |(_, _, k)| w.grid[k]);
        let xi = (alpha.cos() - gamma.cos() * beta.cos()) / gamma.sin();
        let zeta = (1.0_f64 - beta.cos().powi(2) - xi * xi).sqrt();
        let x = Length::from_reduced(u_grid + &v_grid * gamma.cos() + &w_grid * beta.cos());
        let y = Length::from_reduced(v_grid * gamma.sin() + &w_grid * xi);
        let z = Length::from_reduced(w_grid * zeta);
        [x, y, z]
    }

    pub fn x(&self) -> Length<Array1<f64>> {
        Length::from_reduced(self.grid.grids()[0].to_owned())
    }

    pub fn y(&self) -> Length<Array1<f64>> {
        Length::from_reduced(self.grid.grids()[1].to_owned())
    }

    pub fn z(&self) -> Length<Array1<f64>> {
        Length::from_reduced(self.grid.grids()[2].to_owned())
    }
}

impl<D: Dimension + RemoveAxis + 'static, F: HelmholtzEnergyFunctional> DFTProfile<D, F>
where
    D::Larger: Dimension<Smaller = D>,
    D::Smaller: Dimension<Larger = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    /// Create a new density profile.
    ///
    /// If no external potential is specified, it is set to 0. The density is
    /// initialized based on the bulk state and the external potential. The
    /// specification is set to `ChemicalPotential` and needs to be overriden
    /// after this call if something else is required.
    pub fn new(
        grid: Grid,
        bulk: &State<F>,
        external_potential: Option<Array<f64, D::Larger>>,
        density: Option<&Density<Array<f64, D::Larger>>>,
        lanczos: Option<i32>,
    ) -> Self {
        // initialize convolver
        let t = bulk.temperature.to_reduced();
        let weight_functions = bulk.eos.weight_functions(t);
        let convolver = ConvolverFFT::plan(&grid, &weight_functions, lanczos);

        // initialize external potential
        let external_potential = external_potential.unwrap_or_else(|| {
            let mut n_grid = vec![bulk.eos.component_index().len()];
            grid.axes()
                .iter()
                .for_each(|&ax| n_grid.push(ax.grid.len()));
            Array::zeros(n_grid).into_dimensionality().unwrap()
        });

        // initialize density
        let density = if let Some(density) = density {
            density.to_owned()
        } else {
            let exp_dfdrho = (-&external_potential).mapv(f64::exp);
            let mut bonds = bulk.eos.bond_integrals(t, &exp_dfdrho, &convolver);
            bonds *= &exp_dfdrho;
            let mut density = Array::zeros(external_potential.raw_dim());
            let bulk_density = bulk.partial_density.to_reduced();
            for (s, &c) in bulk.eos.component_index().iter().enumerate() {
                density.index_axis_mut(Axis_nd(0), s).assign(
                    &(bonds.index_axis(Axis_nd(0), s).map(|is| is.min(1.0)) * bulk_density[c]),
                );
            }
            Density::from_reduced(density)
        };

        Self {
            grid,
            convolver,
            temperature: bulk.temperature,
            density,
            specification: Arc::new(DFTSpecifications::ChemicalPotential),
            external_potential,
            bulk: bulk.clone(),
            solver_log: None,
            lanczos,
        }
    }
}

impl<D: Dimension, F: HelmholtzEnergyFunctional> DFTProfile<D, F>
where
    D::Larger: Dimension<Smaller = D>,
{
    fn integrate_reduced<N: DualNum<f64> + Copy>(&self, mut profile: Array<N, D>) -> N {
        let (integration_weights, functional_determinant) = self.grid.integration_weights();

        for (i, w) in integration_weights.into_iter().enumerate() {
            for mut l in profile.lanes_mut(Axis_nd(i)) {
                l.mul_assign(&w.mapv(N::from));
            }
        }
        profile.sum() * functional_determinant
    }

    fn integrate_reduced_comp<S: Data<Elem = N>, N: DualNum<f64> + Copy>(
        &self,
        profile: &ArrayBase<S, D::Larger>,
    ) -> Array1<N> {
        Array1::from_shape_fn(profile.shape()[0], |i| {
            self.integrate_reduced(profile.index_axis(Axis_nd(0), i).to_owned())
        })
    }

    pub(crate) fn integrate_reduced_segments<S: Data<Elem = N>, N: DualNum<f64> + Copy>(
        &self,
        profile: &ArrayBase<S, D::Larger>,
    ) -> DVector<N> {
        let integral = self.integrate_reduced_comp(profile);
        let mut integral_comp = DVector::zeros(self.bulk.eos.components());
        for (i, &j) in self.bulk.eos.component_index().iter().enumerate() {
            integral_comp[j] = integral[i];
        }
        integral_comp
    }

    /// Return the volume of the profile.
    ///
    /// In periodic directions, the length is assumed to be 1 Å.
    pub fn volume(&self) -> Volume {
        let volume: f64 = self.grid.axes().iter().map(|ax| ax.volume()).product();
        Volume::from_reduced(volume * self.grid.functional_determinant())
    }

    /// Integrate a given profile over the iteration domain.
    pub fn integrate<S: Data<Elem = f64>, U>(
        &self,
        profile: &Quantity<ArrayBase<S, D>, U>,
    ) -> Quantity<f64, Sum<_Volume, U>>
    where
        _Volume: Add<U>,
    {
        let (integration_weights, functional_determinant) = self.grid.integration_weights();
        let mut value = profile.to_owned();
        for (i, &w) in integration_weights.iter().enumerate() {
            for mut l in value.lanes_mut(Axis_nd(i)) {
                l.assign(&(&l * w));
            }
        }
        Volume::from_reduced(functional_determinant) * value.sum()
    }

    /// Integrate each component individually.
    pub fn integrate_comp<S: Data<Elem = f64>, U>(
        &self,
        profile: &Quantity<ArrayBase<S, D::Larger>, U>,
    ) -> Quantity<DVector<f64>, Sum<_Volume, U>>
    where
        _Volume: Add<U>,
    {
        Quantity::from_fn_generic(Dyn(profile.shape()[0]), U1, |i, _| {
            self.integrate(&profile.index_axis(Axis_nd(0), i))
        })
    }

    /// Integrate each segment individually and aggregate to components.
    pub fn integrate_segments<S: Data<Elem = f64>, U>(
        &self,
        profile: &Quantity<ArrayBase<S, D::Larger>, U>,
    ) -> Quantity<DVector<f64>, Sum<_Volume, U>>
    where
        _Volume: Add<U>,
    {
        let integral = self.integrate_comp(profile);
        let mut integral_comp = Quantity::new(DVector::zeros(self.bulk.eos.components()));
        for (i, &j) in self.bulk.eos.component_index().iter().enumerate() {
            integral_comp.set(j, integral.get(i));
        }
        integral_comp
    }

    /// Return the number of moles of each component in the system.
    pub fn moles(&self) -> Moles<DVector<f64>> {
        self.integrate_segments(&self.density)
    }

    /// Return the total number of moles in the system.
    pub fn total_moles(&self) -> Moles {
        self.moles().sum()
    }
}

impl<D: Dimension, F> DFTProfile<D, F>
where
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
    F: HelmholtzEnergyFunctional,
{
    pub fn weighted_densities(&self) -> FeosResult<Vec<Array<f64, D::Larger>>> {
        Ok(self
            .convolver
            .weighted_densities(&self.density.to_reduced()))
    }

    #[expect(clippy::type_complexity)]
    pub fn residual(&self, log: bool) -> FeosResult<(Array<f64, D::Larger>, Array1<f64>, f64)> {
        // Read from profile
        let density = self.density.to_reduced();
        let partial_density = self.bulk.partial_density.to_reduced();
        let bulk_density = self
            .bulk
            .eos
            .component_index()
            .iter()
            .map(|&i| partial_density[i])
            .collect();

        let (res, res_bulk, res_norm, _, _) =
            self.euler_lagrange_equation(&density, &bulk_density, log)?;
        Ok((res, res_bulk, res_norm))
    }

    #[expect(clippy::type_complexity)]
    pub(crate) fn euler_lagrange_equation(
        &self,
        density: &Array<f64, D::Larger>,
        bulk_density: &Array1<f64>,
        log: bool,
    ) -> FeosResult<(
        Array<f64, D::Larger>,
        Array1<f64>,
        f64,
        Array<f64, D::Larger>,
        Array<f64, D::Larger>,
    )> {
        // calculate reduced temperature
        let temperature = self.temperature.to_reduced();

        // calculate intrinsic functional derivative
        let (_, mut dfdrho) =
            self.bulk
                .eos
                .functional_derivative(temperature, density, &self.convolver)?;

        // calculate total functional derivative
        dfdrho += &self.external_potential;

        // calculate bulk functional derivative
        let bulk_convolver = BulkConvolver::new(self.bulk.eos.weight_functions(temperature));
        let (_, dfdrho_bulk) =
            self.bulk
                .eos
                .functional_derivative(temperature, bulk_density, &bulk_convolver)?;
        dfdrho
            .outer_iter_mut()
            .zip(dfdrho_bulk)
            .zip(self.bulk.eos.m().iter())
            .for_each(|((mut df, df_b), &m)| {
                df -= df_b;
                df /= m
            });

        // calculate bond integrals
        let exp_dfdrho = dfdrho.mapv(|x| (-x).exp());
        let bonds = self
            .bulk
            .eos
            .bond_integrals(temperature, &exp_dfdrho, &self.convolver);
        let mut rho_projected = &exp_dfdrho * bonds;

        // multiply bulk density
        rho_projected
            .outer_iter_mut()
            .zip(bulk_density.iter())
            .for_each(|(mut x, &rho_b)| {
                x *= rho_b;
            });

        // calculate residual
        let mut res = if log {
            rho_projected.mapv(f64::ln) - density.mapv(f64::ln)
        } else {
            &rho_projected - density
        };

        // set residual to 0 where external potentials are overwhelming
        res.iter_mut()
            .zip(self.external_potential.iter())
            .filter(|&(_, &p)| p + f64::EPSILON >= MAX_POTENTIAL)
            .for_each(|(r, _)| *r = 0.0);

        // additional residuals for the calculation of the bulk densities
        let z = self.integrate_reduced_comp(&rho_projected);
        let res_bulk = bulk_density
            - self
                .specification
                .calculate_bulk_density(self, bulk_density, &z)?;

        // calculate the norm of the residual
        let res_norm = ((density - &rho_projected).mapv(|x| x * x).sum()
            + res_bulk.mapv(|x| x * x).sum())
        .sqrt()
            / ((res.len() + res_bulk.len()) as f64).sqrt();

        if res_norm.is_finite() {
            Ok((res, res_bulk, res_norm, exp_dfdrho, rho_projected))
        } else {
            Err(FeosError::IterationFailed("Euler-Lagrange equation".into()))
        }
    }

    pub fn solve(&mut self, solver: Option<&DFTSolver>, debug: bool) -> FeosResult<()> {
        // unwrap solver
        let solver = solver.cloned().unwrap_or_default();

        // Read from profile
        let component_index = self.bulk.eos.component_index().into_owned();
        let mut density = self.density.to_reduced();
        let partial_density = self.bulk.partial_density.to_reduced();
        let mut bulk_density = component_index
            .iter()
            .map(|&i| partial_density[i])
            .collect();

        // Call solver(s)
        self.call_solver(&mut density, &mut bulk_density, &solver, debug)?;

        // Update profile
        self.density = Density::from_reduced(density);
        let volume = Volume::from_reduced(1.0);
        let mut moles = self.bulk.moles.clone();
        bulk_density
            .into_iter()
            .enumerate()
            .for_each(|(i, r)| moles.set(component_index[i], Density::from_reduced(r) * volume));
        self.bulk = State::new_nvt(&self.bulk.eos, self.bulk.temperature, volume, &moles)?;

        Ok(())
    }
}
