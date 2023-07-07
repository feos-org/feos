use crate::convolver::{BulkConvolver, Convolver};
use crate::functional::{HelmholtzEnergyFunctional, DFT};
use crate::geometry::Grid;
use crate::solver::{DFTSolver, DFTSolverLog};
use feos_core::{Components, EosError, EosResult, EosUnit, State};
use ndarray::{Array, Array1, ArrayBase, Axis as Axis_nd, Data, Dimension, Ix1, Ix2, Ix3};
use quantity::si::{SIArray, SIArray1, SINumber, SIUnit};
use quantity::Quantity;
use std::ops::MulAssign;
use std::sync::Arc;

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
    ) -> EosResult<Array1<f64>>;
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
    ) -> EosResult<Arc<Self>>
    where
        <D as Dimension>::Larger: Dimension<Smaller = D>,
    {
        let rho = profile.density.to_reduced(SIUnit::reference_density())?;
        Ok(Arc::new(Self::Moles {
            moles: profile.integrate_reduced_comp(&rho),
        }))
    }

    /// Calculate the number of particles from the profile.
    ///
    /// Call this after initializing the density profile to keep the total number of
    /// particles constant in systems, e.g. to fix the equimolar dividing surface.
    pub fn total_moles_from_profile<D: Dimension, F: HelmholtzEnergyFunctional>(
        profile: &DFTProfile<D, F>,
    ) -> EosResult<Arc<Self>>
    where
        <D as Dimension>::Larger: Dimension<Smaller = D>,
    {
        let rho = profile.density.to_reduced(SIUnit::reference_density())?;
        let moles = profile.integrate_reduced_comp(&rho).sum();
        Ok(Arc::new(Self::TotalMoles { total_moles: moles }))
    }
}

impl<D: Dimension, F: HelmholtzEnergyFunctional> DFTSpecification<D, F> for DFTSpecifications {
    fn calculate_bulk_density(
        &self,
        _profile: &DFTProfile<D, F>,
        bulk_density: &Array1<f64>,
        z: &Array1<f64>,
    ) -> EosResult<Array1<f64>> {
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
pub struct DFTProfile<D: Dimension, F> {
    pub grid: Grid,
    pub convolver: Arc<dyn Convolver<f64, D>>,
    pub dft: Arc<DFT<F>>,
    pub temperature: SINumber,
    pub density: SIArray<D::Larger>,
    pub specification: Arc<dyn DFTSpecification<D, F>>,
    pub external_potential: Array<f64, D::Larger>,
    pub bulk: State<DFT<F>>,
    pub solver_log: Option<DFTSolverLog>,
}

impl<F> DFTProfile<Ix1, F> {
    pub fn r(&self) -> SIArray1 {
        self.grid.grids()[0] * SIUnit::reference_length()
    }

    pub fn z(&self) -> SIArray1 {
        self.grid.grids()[0] * SIUnit::reference_length()
    }
}

impl<F> DFTProfile<Ix2, F> {
    pub fn edges(&self) -> (SIArray1, SIArray1) {
        (
            &self.grid.axes()[0].edges * SIUnit::reference_length(),
            &self.grid.axes()[1].edges * SIUnit::reference_length(),
        )
    }

    pub fn r(&self) -> SIArray1 {
        self.grid.grids()[0] * SIUnit::reference_length()
    }

    pub fn z(&self) -> SIArray1 {
        self.grid.grids()[1] * SIUnit::reference_length()
    }
}

impl<F> DFTProfile<Ix3, F> {
    pub fn edges(&self) -> (SIArray1, SIArray1, SIArray1) {
        (
            &self.grid.axes()[0].edges * SIUnit::reference_length(),
            &self.grid.axes()[1].edges * SIUnit::reference_length(),
            &self.grid.axes()[2].edges * SIUnit::reference_length(),
        )
    }

    pub fn x(&self) -> SIArray1 {
        self.grid.grids()[0] * SIUnit::reference_length()
    }

    pub fn y(&self) -> SIArray1 {
        self.grid.grids()[1] * SIUnit::reference_length()
    }

    pub fn z(&self) -> SIArray1 {
        self.grid.grids()[2] * SIUnit::reference_length()
    }
}

impl<D: Dimension, F: HelmholtzEnergyFunctional> DFTProfile<D, F>
where
    <D as Dimension>::Larger: Dimension<Smaller = D>,
{
    /// Create a new density profile.
    ///
    /// If no external potential is specified, it is set to 0. The density is
    /// initialized based on the bulk state and the external potential. The
    /// specification is set to `ChemicalPotential` and needs to be overriden
    /// after this call if something else is required.
    pub fn new(
        grid: Grid,
        convolver: Arc<dyn Convolver<f64, D>>,
        bulk: &State<DFT<F>>,
        external_potential: Option<Array<f64, D::Larger>>,
        density: Option<&SIArray<D::Larger>>,
    ) -> EosResult<Self> {
        let dft = bulk.eos.clone();

        // initialize external potential
        let external_potential = external_potential.unwrap_or_else(|| {
            let mut n_grid = vec![dft.component_index().len()];
            grid.axes()
                .iter()
                .for_each(|&ax| n_grid.push(ax.grid.len()));
            Array::zeros(n_grid).into_dimensionality().unwrap()
        });

        // initialize density
        let density = if let Some(density) = density {
            density.clone()
        } else {
            let t = bulk
                .temperature
                .to_reduced(SIUnit::reference_temperature())?;
            let exp_dfdrho = (-&external_potential).mapv(f64::exp);
            let mut bonds = dft.bond_integrals(t, &exp_dfdrho, &convolver);
            bonds *= &exp_dfdrho;
            let mut density = Array::zeros(external_potential.raw_dim());
            let bulk_density = bulk
                .partial_density
                .to_reduced(SIUnit::reference_density())?;
            for (s, &c) in dft.component_index().iter().enumerate() {
                density.index_axis_mut(Axis_nd(0), s).assign(
                    &(bonds.index_axis(Axis_nd(0), s).map(|is| is.min(1.0)) * bulk_density[c]),
                );
            }
            density * SIUnit::reference_density()
        };

        Ok(Self {
            grid,
            convolver,
            dft: bulk.eos.clone(),
            temperature: bulk.temperature,
            density,
            specification: Arc::new(DFTSpecifications::ChemicalPotential),
            external_potential,
            bulk: bulk.clone(),
            solver_log: None,
        })
    }

    fn integrate_reduced(&self, mut profile: Array<f64, D>) -> f64 {
        let integration_weights = self.grid.integration_weights();

        for (i, w) in integration_weights.into_iter().enumerate() {
            for mut l in profile.lanes_mut(Axis_nd(i)) {
                l.mul_assign(w);
            }
        }
        profile.sum()
    }

    fn integrate_reduced_comp(&self, profile: &Array<f64, D::Larger>) -> Array1<f64> {
        Array1::from_shape_fn(profile.shape()[0], |i| {
            self.integrate_reduced(profile.index_axis(Axis_nd(0), i).to_owned())
        })
    }

    /// Return the volume of the profile.
    ///
    /// Depending on the geometry, the result is in m, m² or m³.
    pub fn volume(&self) -> SINumber {
        self.grid
            .axes()
            .iter()
            .fold(None, |acc, &ax| {
                Some(acc.map_or(ax.volume(), |acc| acc * ax.volume()))
            })
            .unwrap()
    }

    /// Integrate a given profile over the iteration domain.
    pub fn integrate<S: Data<Elem = f64>>(
        &self,
        profile: &Quantity<ArrayBase<S, D>, SIUnit>,
    ) -> SINumber {
        profile.integrate(&self.grid.integration_weights_unit())
    }

    /// Integrate each component individually.
    pub fn integrate_comp<S: Data<Elem = f64>>(
        &self,
        profile: &Quantity<ArrayBase<S, D::Larger>, SIUnit>,
    ) -> SIArray1 {
        SIArray1::from_shape_fn(profile.shape()[0], |i| {
            self.integrate(&profile.index_axis(Axis_nd(0), i))
        })
    }

    /// Integrate each segment individually and aggregate to components.
    pub fn integrate_segments<S: Data<Elem = f64>>(
        &self,
        profile: &Quantity<ArrayBase<S, D::Larger>, SIUnit>,
    ) -> SIArray1 {
        let integral = self.integrate_comp(profile);
        let mut integral_comp = Array1::zeros(self.dft.components()) * integral.get(0);
        for (i, &j) in self.dft.component_index().iter().enumerate() {
            integral_comp.try_set(j, integral.get(i)).unwrap();
        }
        integral_comp
    }

    /// Return the number of moles of each component in the system.
    pub fn moles(&self) -> SIArray1 {
        self.integrate_segments(&self.density)
    }

    /// Return the total number of moles in the system.
    pub fn total_moles(&self) -> SINumber {
        self.moles().sum()
    }
}

impl<D: Dimension, F> Clone for DFTProfile<D, F> {
    fn clone(&self) -> Self {
        Self {
            grid: self.grid.clone(),
            convolver: self.convolver.clone(),
            dft: self.dft.clone(),
            temperature: self.temperature,
            density: self.density.clone(),
            specification: self.specification.clone(),
            external_potential: self.external_potential.clone(),
            bulk: self.bulk.clone(),
            solver_log: self.solver_log.clone(),
        }
    }
}

impl<D, F> DFTProfile<D, F>
where
    D: Dimension,
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
    F: HelmholtzEnergyFunctional,
{
    pub fn weighted_densities(&self) -> EosResult<Vec<Array<f64, D::Larger>>> {
        Ok(self
            .convolver
            .weighted_densities(&self.density.to_reduced(SIUnit::reference_density())?))
    }

    #[allow(clippy::type_complexity)]
    pub fn residual(&self, log: bool) -> EosResult<(Array<f64, D::Larger>, Array1<f64>, f64)> {
        // Read from profile
        let density = self.density.to_reduced(SIUnit::reference_density())?;
        let partial_density = self
            .bulk
            .partial_density
            .to_reduced(SIUnit::reference_density())?;
        let bulk_density = self.dft.component_index().mapv(|i| partial_density[i]);

        let (res, res_bulk, res_norm, _, _) =
            self.euler_lagrange_equation(&density, &bulk_density, log)?;
        Ok((res, res_bulk, res_norm))
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn euler_lagrange_equation(
        &self,
        density: &Array<f64, D::Larger>,
        bulk_density: &Array1<f64>,
        log: bool,
    ) -> EosResult<(
        Array<f64, D::Larger>,
        Array1<f64>,
        f64,
        Array<f64, D::Larger>,
        Array<f64, D::Larger>,
    )> {
        // calculate reduced temperature
        let temperature = self
            .temperature
            .to_reduced(SIUnit::reference_temperature())?;

        // calculate intrinsic functional derivative
        let (_, mut dfdrho) =
            self.dft
                .functional_derivative(temperature, density, &self.convolver)?;

        // calculate total functional derivative
        dfdrho += &self.external_potential;

        // calculate bulk functional derivative
        let bulk_convolver = BulkConvolver::new(self.dft.weight_functions(temperature));
        let (_, dfdrho_bulk) =
            self.dft
                .functional_derivative(temperature, bulk_density, &bulk_convolver)?;
        dfdrho
            .outer_iter_mut()
            .zip(dfdrho_bulk.into_iter())
            .zip(self.dft.m().iter())
            .for_each(|((mut df, df_b), &m)| {
                df -= df_b;
                df /= m
            });

        // calculate bond integrals
        let exp_dfdrho = dfdrho.mapv(|x| (-x).exp());
        let bonds = self
            .dft
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
            .filter(|(_, &p)| p + f64::EPSILON >= MAX_POTENTIAL)
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
            Err(EosError::IterationFailed("Euler-Lagrange equation".into()))
        }
    }

    pub fn solve(&mut self, solver: Option<&DFTSolver>, debug: bool) -> EosResult<()> {
        // unwrap solver
        let solver = solver.cloned().unwrap_or_default();

        // Read from profile
        let component_index = self.dft.component_index().into_owned();
        let mut density = self.density.to_reduced(SIUnit::reference_density())?;
        let partial_density = self
            .bulk
            .partial_density
            .to_reduced(SIUnit::reference_density())?;
        let mut bulk_density = component_index.mapv(|i| partial_density[i]);

        // Call solver(s)
        self.call_solver(&mut density, &mut bulk_density, &solver, debug)?;

        // Update profile
        self.density = density * SIUnit::reference_density();
        let volume = SIUnit::reference_volume();
        let mut moles = self.bulk.moles.clone();
        bulk_density
            .into_iter()
            .enumerate()
            .try_for_each(|(i, r)| {
                moles.try_set(component_index[i], r * SIUnit::reference_density() * volume)
            })?;
        self.bulk = State::new_nvt(&self.bulk.eos, self.bulk.temperature, volume, &moles)?;

        Ok(())
    }
}
