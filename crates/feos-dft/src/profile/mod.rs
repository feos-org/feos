use crate::association::MassAction;
use crate::convolver::Convolver;
use crate::functional::HelmholtzEnergyFunctional;
use crate::geometry::Grid;
use crate::solver::{DFTSolver, DFTSolverLog};
use feos_core::{FeosResult, ReferenceSystem, State};
use nalgebra::{DVector, Dyn, U1};
use ndarray::{Array, Array1, ArrayBase, Axis as Axis_nd, Data, Dimension, Ix0, IxDyn};
use num_dual::DualNum;
use quantity::{_Volume, Density, Energy, Entropy, Length, Moles, Quantity, Temperature, Volume};
use std::ops::{Add, MulAssign};
use std::sync::Arc;

mod properties;

pub(crate) const MAX_POTENTIAL: f64 = 50.0;

/// General specifications for the chemical potential in a DFT calculation.
///
/// In the most basic case, the chemical potential is specified in a DFT calculation,
/// for more general systems, this enum provides the possibility to declare additional
/// equations for the calculation of the chemical potential during the iteration.
#[derive(Clone)]
pub enum DFTSpecification {
    /// DFT with specified chemical potential, given as the logarithms of the fugacities of the
    /// segments.
    ChemicalPotential(Array1<f64>),
    /// DFT with specified number of particles.
    ///
    /// The solution is still a grand canonical density profile, but the chemical
    /// potentials are iterated together with the density profile to obtain a result
    /// with the specified number of particles.
    Moles(Array1<f64>),
    /// DFT with specified total number of moles. The logarithms of the fugacities of the
    /// segments are shifted by a common value, so that their differences are fixed.
    TotalMoles(f64, Array1<f64>),
}

impl DFTSpecification {
    pub fn from_state<F: HelmholtzEnergyFunctional>(state: &State<F>) -> Self {
        let density = state
            .eos
            .segment_densities(&state.partial_density().into_reduced());
        let profile = DFTProfile::bulk(state, Self::Moles(density.clone()));
        let ln_fugacity = profile
            .ln_fugacity(&density, &profile.non_bonded_fractions)
            .unwrap();
        Self::ChemicalPotential(ln_fugacity)
    }
}

/// Initial condition for the density profile of a [DFTProfile].
pub enum InitialDensity<D: Dimension> {
    /// Use the provided density profile.
    Density(Density<Array<f64, D::Larger>>),
    /// Initialize the density profile from the given bulk partial densities,
    /// accounting for the external potential and bond integrals.
    Bulk(Density<DVector<f64>>),
}

#[derive(Clone)]
/// A one-, two-, or three-dimensional density profile.
pub struct DFTProfile<D: Dimension, F> {
    pub grid: Grid,
    pub convolver: Arc<dyn Convolver<f64, D>>,
    pub functional: F,
    pub temperature: Temperature,
    pub density: Density<Array<f64, D::Larger>>,
    pub specification: DFTSpecification,
    pub external_potential: Array<f64, D::Larger>,
    pub non_bonded_fractions: Array<f64, D::Larger>,
    pub association_convolver: Option<Arc<dyn Convolver<f64, D>>>,
    pub association_kernel_convolver: Option<Arc<dyn Convolver<f64, D>>>,
    pub solver_log: DFTSolverLog,
}

impl<D: Dimension, F> DFTProfile<D, F> {
    /// Return the grid points of every axis.
    pub fn axes(&self) -> Vec<Length<Array1<f64>>> {
        self.grid
            .grids()
            .into_iter()
            .cloned()
            .map(Length::from_reduced)
            .collect()
    }

    /// Return the cell edges of every axis.
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
    /// If no external potential is specified, it is set to 0.
    pub fn new(
        grid: Grid,
        functional: &F,
        temperature: Temperature,
        external_potential: Option<&Energy<Array<f64, D::Larger>>>,
        density: InitialDensity<D>,
        specification: DFTSpecification,
    ) -> Self {
        // initialize convolvers
        let t = temperature.to_reduced();
        let (convolver, association_convolver, association_kernel_convolver) =
            functional.convolvers(&grid, t);

        // initialize external potential
        let external_potential = external_potential.map_or_else(
            || {
                let mut n_grid = vec![functional.component_index().len()];
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
        let density = match density {
            InitialDensity::Density(density) => density,
            InitialDensity::Bulk(partial_density) => {
                let exp_dfdrho = (-&external_potential).mapv(f64::exp);
                let mut bonds = functional.bond_integrals(t, &exp_dfdrho, convolver.as_ref());
                bonds *= &exp_dfdrho;
                let mut density = Array::zeros(external_potential.raw_dim());
                let bulk_density = functional.segment_densities(&partial_density.into_reduced());
                for (s, &rho) in bulk_density.iter().enumerate() {
                    density
                        .index_axis_mut(Axis_nd(0), s)
                        .assign(&(bonds.index_axis(Axis_nd(0), s).map(|is| is.min(1.0)) * rho));
                }
                Density::from_reduced(density)
            }
        };

        // initialize fractions of non-bonded association sites
        let non_bonded_fractions = match (functional.association(), &association_convolver) {
            (Some(a), Some(convolver)) => {
                let wd = convolver
                    .weighted_densities(&density.to_reduced())
                    .remove(0);
                MassAction::new(
                    &a,
                    temperature.into_reduced(),
                    &wd,
                    association_kernel_convolver.as_deref(),
                )
                .solve()
            }
            _ => {
                let mut dim = external_potential.raw_dim();
                dim[0] = 0;
                Array::zeros(dim)
            }
        };

        Self {
            grid,
            convolver,
            functional: functional.clone(),
            temperature,
            density,
            specification,
            external_potential,
            non_bonded_fractions,
            association_convolver,
            association_kernel_convolver,
            solver_log: DFTSolverLog::new(),
        }
    }

    /// Create a new density profile in equilibrium with a bulk state.
    ///
    /// The density is initialized based on the bulk state and the external
    /// potential. The specification is set to `ChemicalPotential` and needs
    /// to be overriden after this call if something else is required.
    pub fn from_bulk(
        grid: Grid,
        bulk: &State<F>,
        external_potential: Option<&Energy<Array<f64, D::Larger>>>,
    ) -> Self {
        Self::new(
            grid,
            &bulk.eos,
            bulk.temperature,
            external_potential,
            InitialDensity::Bulk(bulk.partial_density()),
            DFTSpecification::from_state(bulk),
        )
    }

    /// Convert the profile into a profile with dynamic dimension (the convolvers are planned again).
    pub fn into_dyn(self) -> DFTProfile<IxDyn, F> {
        let t = self.temperature.to_reduced();
        let (convolver, association_convolver, association_kernel_convolver) =
            self.functional.convolvers(&self.grid, t);
        DFTProfile {
            grid: self.grid,
            convolver,
            functional: self.functional,
            temperature: self.temperature,
            density: self.density.into_dyn(),
            specification: self.specification,
            external_potential: self.external_potential.into_dyn(),
            non_bonded_fractions: self.non_bonded_fractions.into_dyn(),
            association_convolver,
            association_kernel_convolver,
            solver_log: self.solver_log,
        }
    }

    /// Set a constraint to fix the number of particles of every component based on
    /// the current density profile.
    pub fn fix_moles(&mut self) {
        let moles = self.grid.integrate_reduced_comp(&self.density.to_reduced());
        self.specification = DFTSpecification::Moles(moles);
    }

    /// Set a constraint to fix the total number of particles based on
    /// the current density profile.
    ///
    /// The ratios of the fugacities are fixed to those of the bulk state.
    pub fn fix_total_moles(&mut self, bulk: &State<F>) {
        let rho = self.density.to_reduced();
        let moles = self.grid.integrate_reduced_comp(&rho).sum();
        let DFTSpecification::ChemicalPotential(ln_fugacity) = DFTSpecification::from_state(bulk)
        else {
            unreachable!()
        };
        self.specification = DFTSpecification::TotalMoles(moles, ln_fugacity);
    }

    /// Return the external potential in SI units.
    pub fn external_potential(&self) -> Energy<Array<f64, D::Larger>> {
        Entropy::from_reduced(self.external_potential.clone()) * self.temperature
    }
}

impl<F: HelmholtzEnergyFunctional> DFTProfile<Ix0, F> {
    /// 0D profile of the bulk `state` with given `specification`.
    fn bulk(state: &State<F>, specification: DFTSpecification) -> Self {
        Self::new(
            Grid::Bulk,
            &state.eos,
            state.temperature,
            None,
            InitialDensity::Bulk(state.partial_density()),
            specification,
        )
    }
}

impl<D: Dimension, F: HelmholtzEnergyFunctional> DFTProfile<D, F>
where
    D::Larger: Dimension<Smaller = D>,
{
    pub(crate) fn integrate_reduced_segments<
        S: Data<Elem = N>,
        N: DualNum<Primitive = f64> + Copy,
    >(
        &self,
        profile: &ArrayBase<S, D::Larger>,
    ) -> DVector<N> {
        let integral = self.grid.integrate_reduced_comp(profile);
        let mut integral_comp = DVector::zeros(self.functional.components());
        for (i, &j) in self.functional.component_index().iter().enumerate() {
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

    /// Integrate each segment individually.
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

    /// Integrate each segment individually and assign the result to the
    /// corresponding component.
    ///
    /// Only valid for profiles for which all segments of a molecule have the
    /// same integral (e.g., density profiles).
    pub fn integrate_segments<S: Data<Elem = f64>, U>(
        &self,
        profile: &Quantity<ArrayBase<S, D::Larger>, U>,
    ) -> Quantity<DVector<f64>, <_Volume as Add<U>>::Output>
    where
        _Volume: Add<U>,
    {
        let integral = self.integrate_comp(profile);
        let mut integral_comp = Quantity::new(DVector::zeros(self.functional.components()));
        for (i, &j) in self.functional.component_index().iter().enumerate() {
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

impl<D: Dimension + 'static, F> DFTProfile<D, F>
where
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
    F: HelmholtzEnergyFunctional,
{
    /// Calculate the (reduced) weighted densities of every functional contribution.
    pub fn weighted_densities(&self) -> FeosResult<Vec<Array<f64, D::Larger>>> {
        Ok(self
            .convolver
            .weighted_densities(&self.density.to_reduced()))
    }

    /// Calculate the residual of the Euler-Lagrange equation and its norm.
    pub fn residual(&self, log: bool) -> FeosResult<(Array<f64, D::Larger>, f64)> {
        let residual = self.euler_lagrange_equation(
            &self.density.to_reduced(),
            &self.non_bonded_fractions,
            log,
            &mut DFTSolverLog::new(),
        )?;
        Ok((residual.density, residual.norm))
    }

    /// Solve the Euler-Lagrange equation.
    ///
    /// For specifications other than `ChemicalPotential`, the provided bulk
    /// states are updated to be in equilibrium with the converged profile.
    /// If `debug` is true, no error is returned if the solver does not converge.
    pub fn solve<const B: usize>(
        &mut self,
        bulk_states: [&mut State<F>; B],
        solver: Option<&DFTSolver>,
        debug: bool,
    ) -> FeosResult<()> {
        // unwrap solver
        let solver = solver.cloned().unwrap_or_default();

        // Read from profile
        let mut density = self.density.to_reduced();
        let mut non_bonded_fractions = self.non_bonded_fractions.clone();

        // Call solver(s)
        let mut solver_log = DFTSolverLog::new();
        let result = self.call_solver(
            &mut density,
            &mut non_bonded_fractions,
            &solver,
            &mut solver_log,
            debug,
        );
        // the log of the last solve is kept, also if the solver failed
        self.solver_log = solver_log;
        result?;

        // Update bulk state
        if !matches!(self.specification, DFTSpecification::ChemicalPotential(_)) {
            let ln_fugacity = self.ln_fugacity(&density, &non_bonded_fractions)?;
            for bulk in bulk_states {
                // solve a bulk profile
                let specification = DFTSpecification::ChemicalPotential(ln_fugacity.clone());
                let mut bulk_profile = DFTProfile::bulk(bulk, specification);
                bulk_profile.solve([], None, false)?;

                // create the state based on the results from the bulk profile
                let component_index = self.functional.component_index();
                let mut partial_density = bulk.partial_density();
                bulk_profile
                    .density
                    .into_iter()
                    .enumerate()
                    .for_each(|(i, r)| partial_density.set(component_index[i], r));
                *bulk = State::new_density(&self.functional, self.temperature, partial_density)?;
            }
        }

        // Update profile
        self.density = Density::from_reduced(density);
        self.non_bonded_fractions = non_bonded_fractions;

        Ok(())
    }
}
