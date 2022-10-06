use crate::convolver::{BulkConvolver, Convolver, ConvolverFFT};
use crate::functional::{HelmholtzEnergyFunctional, DFT};
use crate::geometry::Grid;
use crate::solver::DFTSolver;
use crate::weight_functions::WeightFunctionInfo;
use feos_core::{
    log_result, Contributions, EosError, EosResult, EosUnit, EquationOfState, State, Verbosity,
};
use ndarray::{
    s, Array, Array1, ArrayBase, ArrayViewMut, ArrayViewMut1, Axis as Axis_nd, Data, Dimension,
    Ix1, Ix2, Ix3, RemoveAxis,
};
use num_dual::Dual64;
use num_traits::Zero;
use quantity::{Quantity, QuantityArray, QuantityArray1, QuantityScalar};
use std::ops::MulAssign;
use std::sync::Arc;

pub(crate) const MAX_POTENTIAL: f64 = 50.0;
#[cfg(feature = "rayon")]
pub(crate) const CUTOFF_RADIUS: f64 = 14.0;

/// General specifications for the chemical potential in a DFT calculation.
///
/// In the most basic case, the chemical potential is specified in a DFT calculation,
/// for more general systems, this trait provides the possibility to declare additional
/// equations for the calculation of the chemical potential during the iteration.
pub trait DFTSpecification<U, D: Dimension, F>: Send + Sync {
    fn calculate_bulk_density(
        &self,
        profile: &DFTProfile<U, D, F>,
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
    pub fn moles_from_profile<U: EosUnit, D: Dimension, F: HelmholtzEnergyFunctional>(
        profile: &DFTProfile<U, D, F>,
    ) -> EosResult<Arc<Self>>
    where
        <D as Dimension>::Larger: Dimension<Smaller = D>,
    {
        let rho = profile.density.to_reduced(U::reference_density())?;
        Ok(Arc::new(Self::Moles {
            moles: profile.integrate_reduced_comp(&rho),
        }))
    }

    /// Calculate the number of particles from the profile.
    ///
    /// Call this after initializing the density profile to keep the total number of
    /// particles constant in systems, e.g. to fix the equimolar dividing surface.
    pub fn total_moles_from_profile<U: EosUnit, D: Dimension, F: HelmholtzEnergyFunctional>(
        profile: &DFTProfile<U, D, F>,
    ) -> EosResult<Arc<Self>>
    where
        <D as Dimension>::Larger: Dimension<Smaller = D>,
    {
        let rho = profile.density.to_reduced(U::reference_density())?;
        let moles = profile.integrate_reduced_comp(&rho).sum();
        Ok(Arc::new(Self::TotalMoles { total_moles: moles }))
    }
}

impl<U: EosUnit, D: Dimension, F: HelmholtzEnergyFunctional> DFTSpecification<U, D, F>
    for DFTSpecifications
{
    fn calculate_bulk_density(
        &self,
        _profile: &DFTProfile<U, D, F>,
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
pub struct DFTProfile<U, D: Dimension, F> {
    pub grid: Grid,
    pub convolver: Arc<dyn Convolver<f64, D>>,
    pub dft: Arc<DFT<F>>,
    pub temperature: QuantityScalar<U>,
    pub density: QuantityArray<U, D::Larger>,
    pub specification: Arc<dyn DFTSpecification<U, D, F>>,
    pub external_potential: Array<f64, D::Larger>,
    pub bulk: State<U, DFT<F>>,
}

impl<U: EosUnit, F> DFTProfile<U, Ix1, F> {
    pub fn r(&self) -> QuantityArray1<U> {
        self.grid.grids()[0] * U::reference_length()
    }

    pub fn z(&self) -> QuantityArray1<U> {
        self.grid.grids()[0] * U::reference_length()
    }
}

impl<U: EosUnit, F> DFTProfile<U, Ix2, F> {
    pub fn edges(&self) -> (QuantityArray1<U>, QuantityArray1<U>) {
        (
            &self.grid.axes()[0].edges * U::reference_length(),
            &self.grid.axes()[1].edges * U::reference_length(),
        )
    }

    pub fn r(&self) -> QuantityArray1<U> {
        self.grid.grids()[0] * U::reference_length()
    }

    pub fn z(&self) -> QuantityArray1<U> {
        self.grid.grids()[1] * U::reference_length()
    }
}

impl<U: EosUnit, F> DFTProfile<U, Ix3, F> {
    pub fn edges(&self) -> (QuantityArray1<U>, QuantityArray1<U>, QuantityArray1<U>) {
        (
            &self.grid.axes()[0].edges * U::reference_length(),
            &self.grid.axes()[1].edges * U::reference_length(),
            &self.grid.axes()[2].edges * U::reference_length(),
        )
    }

    pub fn x(&self) -> QuantityArray1<U> {
        self.grid.grids()[0] * U::reference_length()
    }

    pub fn y(&self) -> QuantityArray1<U> {
        self.grid.grids()[1] * U::reference_length()
    }

    pub fn z(&self) -> QuantityArray1<U> {
        self.grid.grids()[2] * U::reference_length()
    }
}

impl<U: EosUnit, D: Dimension, F: HelmholtzEnergyFunctional> DFTProfile<U, D, F>
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
        bulk: &State<U, DFT<F>>,
        external_potential: Option<Array<f64, D::Larger>>,
        density: Option<&QuantityArray<U, D::Larger>>,
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
            let t = bulk.temperature.to_reduced(U::reference_temperature())?;
            let bonds = dft
                .bond_integrals(t, &external_potential, &convolver)
                .mapv(f64::abs)
                * (-&external_potential).mapv(f64::exp);
            let mut density = Array::zeros(external_potential.raw_dim());
            let bulk_density = bulk.partial_density.to_reduced(U::reference_density())?;
            for (s, &c) in dft.component_index().iter().enumerate() {
                density.index_axis_mut(Axis_nd(0), s).assign(
                    &(bonds.index_axis(Axis_nd(0), s).map(|is| is.min(1.0)) * bulk_density[c]),
                );
            }
            density * U::reference_density()
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
    pub fn volume(&self) -> QuantityScalar<U> {
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
        profile: &Quantity<ArrayBase<S, D>, U>,
    ) -> QuantityScalar<U> {
        profile.integrate(&self.grid.integration_weights_unit())
    }

    /// Integrate each component individually.
    pub fn integrate_comp<S: Data<Elem = f64>>(
        &self,
        profile: &Quantity<ArrayBase<S, D::Larger>, U>,
    ) -> QuantityArray1<U> {
        QuantityArray1::from_shape_fn(profile.shape()[0], |i| {
            self.integrate(&profile.index_axis(Axis_nd(0), i))
        })
    }

    /// Return the number of moles of each component in the system.
    pub fn moles(&self) -> QuantityArray1<U> {
        let rho = self.density.to_reduced(U::reference_density()).unwrap();
        let mut d = rho.raw_dim();
        d[0] = self.dft.components();
        let mut density_comps = Array::zeros(d);
        for (i, &j) in self.dft.component_index().iter().enumerate() {
            density_comps
                .index_axis_mut(Axis_nd(0), j)
                .assign(&rho.index_axis(Axis_nd(0), i));
        }
        self.integrate_comp(&(density_comps * U::reference_density()))
    }

    /// Return the total number of moles in the system.
    pub fn total_moles(&self) -> QuantityScalar<U> {
        self.moles().sum()
    }

    /// Return the chemical potential of the system
    pub fn chemical_potential(&self) -> QuantityArray1<U> {
        self.bulk.chemical_potential(Contributions::Total)
    }
}

impl<U: Clone, D: Dimension, F> Clone for DFTProfile<U, D, F> {
    fn clone(&self) -> Self {
        Self {
            grid: self.grid.clone(),
            convolver: self.convolver.clone(),
            dft: self.dft.clone(),
            temperature: self.temperature.clone(),
            density: self.density.clone(),
            specification: self.specification.clone(),
            external_potential: self.external_potential.clone(),
            bulk: self.bulk.clone(),
        }
    }
}

impl<U, D, F> DFTProfile<U, D, F>
where
    U: EosUnit,
    D: Dimension,
    D::Larger: Dimension<Smaller = D>,
    F: HelmholtzEnergyFunctional,
{
    pub fn weighted_densities(&self) -> EosResult<Vec<Array<f64, D::Larger>>> {
        Ok(self
            .convolver
            .weighted_densities(&self.density.to_reduced(U::reference_density())?))
    }

    pub fn functional_derivative(&self) -> EosResult<Array<f64, D::Larger>> {
        let (_, dfdrho) = self.dft.functional_derivative(
            self.temperature.to_reduced(U::reference_temperature())?,
            &self.density.to_reduced(U::reference_density())?,
            &self.convolver,
        )?;
        Ok(dfdrho)
    }

    #[allow(clippy::type_complexity)]
    pub fn residual(&self, log: bool) -> EosResult<(Array<f64, D::Larger>, Array1<f64>)> {
        // Read from profile
        let temperature = self.temperature.to_reduced(U::reference_temperature())?;
        let density = self.density.to_reduced(U::reference_density())?;
        let partial_density = self
            .bulk
            .partial_density
            .to_reduced(U::reference_density())?;
        let bulk_density = self.dft.component_index().mapv(|i| partial_density[i]);

        // Allocate residual vectors
        let mut res_rho = Array::zeros(density.raw_dim());
        let mut res_bulk = Array1::zeros(bulk_density.len());

        self.calculate_residual(
            temperature,
            &density,
            &bulk_density,
            res_rho.view_mut(),
            res_bulk.view_mut(),
            log,
        )?;

        Ok((res_rho, res_bulk))
    }

    fn calculate_residual(
        &self,
        temperature: f64,
        density: &Array<f64, D::Larger>,
        bulk_density: &Array1<f64>,
        mut res_rho: ArrayViewMut<f64, D::Larger>,
        mut res_bulk: ArrayViewMut1<f64>,
        log: bool,
    ) -> EosResult<()> {
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
        let mut exp_dfdrho = dfdrho.mapv(|x| (-x).exp());
        let bonds = self
            .dft
            .bond_integrals(temperature, &exp_dfdrho, &self.convolver);
        exp_dfdrho *= &bonds;

        // Euler-Lagrange equation
        res_rho
            .outer_iter_mut()
            .zip(exp_dfdrho.outer_iter())
            .zip(bulk_density.iter())
            .zip(density.outer_iter())
            .for_each(|(((mut res, df), &rho_b), rho)| {
                res.assign(
                    &(if log {
                        if rho_b.is_zero() {
                            Array::zeros(res.raw_dim())
                        } else {
                            rho.mapv(f64::ln) - rho_b.ln() - df.mapv(f64::ln)
                        }
                    } else {
                        &rho - rho_b * &df
                    }),
                );
            });

        // set residual to 0 where external potentials are overwhelming
        res_rho
            .iter_mut()
            .zip(self.external_potential.iter())
            .for_each(|(r, &p)| {
                if p + f64::EPSILON >= MAX_POTENTIAL {
                    *r = 0.0;
                }
            });

        // Additional residuals for the calculation of the bulk densitiess
        let z = self.integrate_reduced_comp(&exp_dfdrho);
        let bulk_spec = self
            .specification
            .calculate_bulk_density(self, bulk_density, &z)?;

        res_bulk.assign(
            &(if log {
                println!("{bulk_density} {bulk_spec}");
                (bulk_density.mapv(f64::ln) - bulk_spec.mapv(f64::ln)).mapv(|r| {
                    if r.is_normal() {
                        r
                    } else {
                        0.0
                    }
                })
            } else {
                bulk_density - &bulk_spec
            }),
        );

        Ok(())
    }

    pub fn solve(&mut self, solver: Option<&DFTSolver>, debug: bool) -> EosResult<()> {
        // unwrap solver
        let solver = solver.cloned().unwrap_or_default();

        // Read from profile
        let component_index = self.dft.component_index();
        let temperature = self.temperature.to_reduced(U::reference_temperature())?;
        let mut density = self.density.to_reduced(U::reference_density())?;
        let partial_density = self
            .bulk
            .partial_density
            .to_reduced(U::reference_density())?;
        let mut bulk_density = component_index.mapv(|i| partial_density[i]);

        // initialize x-vector
        let n_rho = density.len();
        let mut x = Array1::zeros(n_rho + bulk_density.len());
        x.slice_mut(s![..n_rho])
            .assign(&density.view().into_shape(n_rho).unwrap());
        x.slice_mut(s![n_rho..]).assign(&bulk_density);

        // Residual function
        let mut residual =
            |x: &Array1<f64>, mut res: ArrayViewMut1<f64>, log: bool| -> EosResult<()> {
                // Read density and chemical potential from solution vector
                density.assign(&x.slice(s![..n_rho]).into_shape(density.shape()).unwrap());
                bulk_density.assign(&x.slice(s![n_rho..]));

                // Create views for different residuals
                let (res_rho, res_mu) = res.multi_slice_mut((s![..n_rho], s![n_rho..]));
                let res_rho = res_rho.into_shape(density.raw_dim()).unwrap();

                // Calculate residual
                self.calculate_residual(temperature, &density, &bulk_density, res_rho, res_mu, log)
            };

        // Call solver(s)
        let (converged, iterations) = solver.solve(&mut x, &mut residual)?;
        if converged {
            log_result!(solver.verbosity, "DFT solved in {} iterations", iterations);
        } else if debug {
            log_result!(
                solver.verbosity,
                "DFT not converged in {} iterations",
                iterations
            );
        } else {
            return Err(EosError::NotConverged(String::from("DFT")));
        }

        // Update profile
        self.density = density * U::reference_density();
        let volume = U::reference_volume();
        let mut moles = self.bulk.moles.clone();
        bulk_density
            .into_iter()
            .enumerate()
            .try_for_each(|(i, r)| {
                moles.try_set(component_index[i], r * U::reference_density() * volume)
            })?;
        self.bulk = State::new_nvt(&self.bulk.eos, self.bulk.temperature, volume, &moles)?;

        Ok(())
    }
}

impl<U: EosUnit, D: Dimension + RemoveAxis + 'static, F: HelmholtzEnergyFunctional>
    DFTProfile<U, D, F>
where
    D::Larger: Dimension<Smaller = D>,
    D::Smaller: Dimension<Larger = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    pub fn entropy_density(&self, contributions: Contributions) -> EosResult<QuantityArray<U, D>> {
        // initialize convolver
        let t = self.temperature.to_reduced(U::reference_temperature())?;
        let functional_contributions = self.dft.contributions();
        let weight_functions: Vec<WeightFunctionInfo<Dual64>> = functional_contributions
            .iter()
            .map(|c| c.weight_functions(Dual64::from(t).derive()))
            .collect();
        let convolver = ConvolverFFT::plan(&self.grid, &weight_functions, None);

        Ok(self.dft.entropy_density(
            t,
            &self.density.to_reduced(U::reference_density())?,
            &convolver,
            contributions,
        )? * (U::reference_entropy() / U::reference_volume()))
    }

    pub fn entropy(&self, contributions: Contributions) -> EosResult<QuantityScalar<U>> {
        Ok(self.integrate(&self.entropy_density(contributions)?))
    }

    pub fn grand_potential_density(&self) -> EosResult<QuantityArray<U, D>> {
        self.dft
            .grand_potential_density(self.temperature, &self.density, &self.convolver)
    }

    pub fn grand_potential(&self) -> EosResult<QuantityScalar<U>> {
        Ok(self.integrate(&self.grand_potential_density()?))
    }

    pub fn internal_energy(&self, contributions: Contributions) -> EosResult<QuantityScalar<U>> {
        // initialize convolver
        let t = self.temperature.to_reduced(U::reference_temperature())?;
        let functional_contributions = self.dft.contributions();
        let weight_functions: Vec<WeightFunctionInfo<Dual64>> = functional_contributions
            .iter()
            .map(|c| c.weight_functions(Dual64::from(t).derive()))
            .collect();
        let convolver = ConvolverFFT::plan(&self.grid, &weight_functions, None);

        let internal_energy_density = self.dft.internal_energy_density(
            t,
            &self.density.to_reduced(U::reference_density())?,
            &self.external_potential,
            &convolver,
            contributions,
        )? * U::reference_pressure();
        Ok(self.integrate(&internal_energy_density))
    }
}
