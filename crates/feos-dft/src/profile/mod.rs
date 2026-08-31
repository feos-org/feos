use crate::convolver::{Convolver, ConvolverFFT, PeriodicConvolver};
use crate::functional::HelmholtzEnergyFunctional;
use crate::geometry::Grid;
use crate::solver::{DFTSolver, DFTSolverLog};
use feos_core::{FeosError, FeosResult, ReferenceSystem, State};
use nalgebra::{DVector, Dyn, U1};
use ndarray::{Array, Array1, ArrayBase, Axis as Axis_nd, Data, Dimension, Ix0, arr1};
use num_dual::DualNum;
use quantity::{_Volume, Density, Energy, Entropy, Length, Moles, Quantity, Temperature, Volume};
use std::ops::{Add, MulAssign};
use std::sync::Arc;

mod properties;

const MAX_POTENTIAL: f64 = 50.0;

/// General specifications for the chemical potential in a DFT calculation.
///
/// In the most basic case, the chemical potential is specified in a DFT calculation,
/// for more general systems, this enum provides the possibility to declare additional
/// equations for the calculation of the chemical potential during the iteration.
#[derive(Clone)]
pub enum DFTSpecification {
    /// DFT with specified chemical potential.
    ChemicalPotential(Array1<f64>),
    /// DFT with specified number of particles.
    ///
    /// The solution is still a grand canonical density profile, but the chemical
    /// potentials are iterated together with the density profile to obtain a result
    /// with the specified number of particles.
    Moles(Array1<f64>),
    /// DFT with specified total number of moles.
    TotalMoles(f64, Array1<f64>),
}

impl DFTSpecification {
    fn calculate_fugacity(&self, z: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::ChemicalPotential(fugacity) => fugacity.clone(),
            Self::Moles(moles) => moles / z,
            Self::TotalMoles(total_moles, fugacity) => {
                fugacity * *total_moles / (fugacity * z).sum()
            }
        }
    }

    pub(crate) fn delta_fugacity(&self, z: &Array1<f64>, delta_z: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::ChemicalPotential(fugacity) => Array1::zeros(fugacity.len()),
            Self::Moles(_) => -delta_z / z,
            Self::TotalMoles(_, fugacity) => {
                -(fugacity * delta_z).sum() / (fugacity * z).sum() * Array1::ones(fugacity.len())
            }
        }
    }

    pub fn from_state<F: HelmholtzEnergyFunctional>(state: &State<F>) -> Self {
        let component_index = state.eos.component_index().into_owned();
        let m = arr1(&state.eos.m());
        let partial_density = state.partial_density().into_reduced();
        let temperature = state.temperature.into_reduced();
        let bulk_density = component_index
            .iter()
            .map(|&i| partial_density[i])
            .collect();
        let bulk_convolver =
            PeriodicConvolver::<_, Ix0>::new_0d(&state.eos.weight_functions(temperature));
        let (_, dfdrho_bulk) = state
            .eos
            .functional_derivative(temperature, &bulk_density, bulk_convolver.as_ref())
            .unwrap();
        let exp_dfdrho = (dfdrho_bulk / m).mapv(f64::exp);
        let bonds = state
            .eos
            .bond_integrals(temperature, &exp_dfdrho, bulk_convolver.as_ref());
        let fugacity = bulk_density * exp_dfdrho * bonds;
        Self::ChemicalPotential(fugacity)
    }
}

#[derive(Clone)]
/// A one-, two-, or three-dimensional density profile.
pub struct DFTProfile<D: Dimension, F> {
    pub grid: Grid,
    pub convolver: Arc<dyn Convolver<f64, D>>,
    pub temperature: Temperature,
    pub density: Density<Array<f64, D::Larger>>,
    pub specification: DFTSpecification,
    pub external_potential: Array<f64, D::Larger>,
    pub bulk: State<F>,
    pub solver_log: Option<DFTSolverLog>,
    pub lanczos: Option<i32>,
}

impl<D: Dimension, F> DFTProfile<D, F> {
    pub fn axes(&self) -> Vec<Length<Array1<f64>>> {
        self.grid
            .grids()
            .into_iter()
            .cloned()
            .map(Length::from_reduced)
            .collect()
    }

    pub fn edges(&self) -> Vec<Length<Array1<f64>>> {
        self.grid
            .axes()
            .into_iter()
            .map(|a| Length::from_reduced(a.edges.clone()))
            .collect()
    }
}

impl<D: Dimension + 'static, F: HelmholtzEnergyFunctional> DFTProfile<D, F>
where
    D::Larger: Dimension<Smaller = D>,
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
        external_potential: Option<&Energy<Array<f64, D::Larger>>>,
        density: Option<&Density<Array<f64, D::Larger>>>,
        lanczos: Option<i32>,
    ) -> Self {
        // initialize convolver
        let t = bulk.temperature.to_reduced();
        let weight_functions = bulk.eos.weight_functions(t);
        let convolver = ConvolverFFT::plan(&grid, &weight_functions, lanczos);

        // initialize external potential
        let external_potential = external_potential.map_or_else(
            || {
                let mut n_grid = vec![bulk.eos.component_index().len()];
                grid.axes()
                    .iter()
                    .for_each(|&ax| n_grid.push(ax.grid.len()));
                Array::zeros(n_grid).into_dimensionality().unwrap()
            },
            |e| {
                let mut external_potential = e.to_reduced() / t;
                external_potential.map_inplace(|x| {
                    if *x > MAX_POTENTIAL {
                        *x = MAX_POTENTIAL
                    }
                });
                external_potential
            },
        );

        // initialize density
        let density = density.map_or_else(
            || {
                let exp_dfdrho = (-&external_potential).mapv(f64::exp);
                let mut bonds = bulk.eos.bond_integrals(t, &exp_dfdrho, convolver.as_ref());
                bonds *= &exp_dfdrho;
                let mut density = Array::zeros(external_potential.raw_dim());
                let bulk_density = bulk.partial_density().into_reduced();
                for (s, &c) in bulk.eos.component_index().iter().enumerate() {
                    density.index_axis_mut(Axis_nd(0), s).assign(
                        &(bonds.index_axis(Axis_nd(0), s).map(|is| is.min(1.0)) * bulk_density[c]),
                    );
                }
                Density::from_reduced(density)
            },
            Clone::clone,
        );

        Self {
            grid,
            convolver,
            temperature: bulk.temperature,
            density,
            specification: DFTSpecification::from_state(bulk),
            external_potential,
            bulk: bulk.clone(),
            solver_log: None,
            lanczos,
        }
    }

    /// Set a constraint to fix the number of particles of every component based on
    /// the current density profile.
    pub fn fix_moles(&mut self) {
        let moles = self.integrate_reduced_comp(&self.density.to_reduced());
        self.specification = DFTSpecification::Moles(moles);
    }

    /// Set a constraint to fix the total number of particles based on
    /// the current density profile.
    pub fn fix_total_moles(&mut self) {
        let rho = self.density.to_reduced();
        let moles = self.integrate_reduced_comp(&rho).sum();
        let DFTSpecification::ChemicalPotential(fugacity) =
            DFTSpecification::from_state(&self.bulk)
        else {
            unreachable!()
        };
        self.specification = DFTSpecification::TotalMoles(moles, fugacity);
    }

    /// Return the external potential in SI units.
    pub fn external_potential(&self) -> Energy<Array<f64, D::Larger>> {
        Entropy::from_reduced(self.external_potential.clone()) * self.temperature
    }
}

impl<D: Dimension, F: HelmholtzEnergyFunctional> DFTProfile<D, F>
where
    D::Larger: Dimension<Smaller = D>,
{
    fn integrate_reduced<N: DualNum<Primitive = f64> + Copy>(&self, mut profile: Array<N, D>) -> N {
        let (integration_weights, functional_determinant) = self.grid.integration_weights();

        for (i, w) in integration_weights.into_iter().enumerate() {
            for mut l in profile.lanes_mut(Axis_nd(i)) {
                l.mul_assign(&w.mapv(N::from));
            }
        }
        profile.sum() * functional_determinant
    }

    pub(crate) fn integrate_reduced_comp<S: Data<Elem = N>, N: DualNum<Primitive = f64> + Copy>(
        &self,
        profile: &ArrayBase<S, D::Larger>,
    ) -> Array1<N> {
        Array1::from_shape_fn(profile.shape()[0], |i| {
            self.integrate_reduced(profile.index_axis(Axis_nd(0), i).to_owned())
        })
    }

    pub(crate) fn integrate_reduced_segments<
        S: Data<Elem = N>,
        N: DualNum<Primitive = f64> + Copy,
    >(
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
    ) -> Quantity<f64, <_Volume as Add<U>>::Output>
    where
        _Volume: Add<U>,
    {
        let (integration_weights, functional_determinant) = self.grid.integration_weights();
        let mut value = profile.to_owned();
        for (i, &w) in integration_weights.iter().enumerate() {
            for mut l in value.lanes_mut(Axis_nd(i)) {
                l.mul_assign(w);
            }
        }
        Volume::from_reduced(functional_determinant) * value.sum()
    }

    /// Integrate each component individually.
    pub fn integrate_comp<S: Data<Elem = f64>, U>(
        &self,
        profile: &Quantity<ArrayBase<S, D::Larger>, U>,
    ) -> Quantity<DVector<f64>, <_Volume as Add<U>>::Output>
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
    ) -> Quantity<DVector<f64>, <_Volume as Add<U>>::Output>
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

    pub fn residual(&self, log: bool) -> FeosResult<(Array<f64, D::Larger>, f64)> {
        let density = self.density.to_reduced();
        let (res, res_norm, _, _, _) = self.euler_lagrange_equation(&density, log)?;
        Ok((res, res_norm))
    }

    #[expect(clippy::type_complexity)]
    fn fugacity(
        &self,
        density: &Array<f64, D::Larger>,
    ) -> FeosResult<(
        Array<f64, D::Larger>,
        Array1<f64>,
        Array<f64, D::Larger>,
        Array1<f64>,
    )> {
        // calculate reduced temperature
        let temperature = self.temperature.to_reduced();

        // calculate intrinsic functional derivative
        let (_, mut dfdrho) =
            self.bulk
                .eos
                .functional_derivative(temperature, density, self.convolver.as_ref())?;

        // calculate total functional derivative
        dfdrho += &self.external_potential;

        dfdrho
            .outer_iter_mut()
            .zip(self.bulk.eos.m().iter())
            .for_each(|(mut df, &m)| df /= m);

        // calculate bond integrals
        let exp_dfdrho = dfdrho.mapv(|x| (-x).exp());
        let bonds = self
            .bulk
            .eos
            .bond_integrals(temperature, &exp_dfdrho, self.convolver.as_ref());
        let mut rho_projected = &exp_dfdrho * bonds;
        let z = self.integrate_reduced_comp(&rho_projected);

        // calculate fugacity based on the given specification
        let fugacity = self.specification.calculate_fugacity(&z);

        // multiply fugacity
        rho_projected
            .outer_iter_mut()
            .zip(fugacity.iter())
            .for_each(|(mut x, &f)| {
                x *= f;
            });

        Ok((exp_dfdrho, z, rho_projected, fugacity))
    }

    #[expect(clippy::type_complexity)]
    pub(crate) fn euler_lagrange_equation(
        &self,
        density: &Array<f64, D::Larger>,
        log: bool,
    ) -> FeosResult<(
        Array<f64, D::Larger>,
        f64,
        Array<f64, D::Larger>,
        Array1<f64>,
        Array<f64, D::Larger>,
    )> {
        // calculate functional derivatives and fugacity
        let (exp_dfdrho, z, rho_projected, _) = self.fugacity(density)?;

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

        // calculate the norm of the residual
        let res_norm =
            (density - &rho_projected).mapv(|x| x * x).sum().sqrt() / (res.len() as f64).sqrt();

        if res_norm.is_finite() {
            Ok((res, res_norm, exp_dfdrho, z, rho_projected))
        } else {
            Err(FeosError::IterationFailed("Euler-Lagrange equation".into()))
        }
    }

    pub fn solve(&mut self, solver: Option<&DFTSolver>, debug: bool) -> FeosResult<()> {
        // unwrap solver
        let solver = solver.cloned().unwrap_or_default();

        // Read from profile
        let mut density = self.density.to_reduced();

        // Call solver(s)
        self.call_solver(&mut density, &solver, debug)?;

        // Update bulk state
        if !matches!(self.specification, DFTSpecification::ChemicalPotential(_)) {
            // solve a bulk profile with the Newton solver
            let mut bulk_profile =
                DFTProfile::<Ix0, _>::new(Grid::Bulk, &self.bulk, None, None, None);
            let (_, _, _, fugacity) = self.fugacity(&density)?;
            bulk_profile.specification = DFTSpecification::ChemicalPotential(fugacity);
            let solver = DFTSolver::new(None).newton(None, None, None, None);
            bulk_profile.solve(Some(&solver), false)?;

            // create the state based on the results from the bulk profile
            let component_index = self.bulk.eos.component_index();
            let mut partial_density = self.bulk.partial_density();
            bulk_profile
                .density
                .into_iter()
                .enumerate()
                .for_each(|(i, r)| partial_density.set(component_index[i], r));
            self.bulk = State::new_density(&self.bulk.eos, self.bulk.temperature, partial_density)?;
        }

        // Update profile
        self.density = Density::from_reduced(density);

        Ok(())
    }
}
