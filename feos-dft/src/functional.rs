use crate::convolver::Convolver;
use crate::functional_contribution::*;
use crate::ideal_chain_contribution::IdealChainContribution;
use crate::weight_functions::{WeightFunction, WeightFunctionInfo, WeightFunctionShape};
use feos_core::{
    Contributions, EosResult, EosUnit, EquationOfState, HelmholtzEnergy, HelmholtzEnergyDual,
    IdealGasContribution, IdealGasContributionDual, MolarWeight, StateHD,
};
use ndarray::*;
use num_dual::*;
use petgraph::graph::{Graph, UnGraph};
use petgraph::visit::EdgeRef;
use petgraph::Directed;
use quantity::{QuantityArray, QuantityArray1, QuantityScalar};
use std::borrow::Cow;
use std::fmt;
use std::ops::{AddAssign, Deref, MulAssign};
use std::rc::Rc;

/// Wrapper struct for the [HelmholtzEnergyFunctional] trait.
///
/// Needed (for now) to generically implement the `EquationOfState`
/// trait for Helmholtz energy functionals.
#[derive(Clone)]
pub struct DFT<F>(F);

impl<F> From<F> for DFT<F> {
    fn from(functional: F) -> Self {
        Self(functional)
    }
}

impl<F> DFT<F> {
    pub fn into<F2: From<F>>(self) -> DFT<F2> {
        DFT(self.0.into())
    }
}

impl<F> Deref for DFT<F> {
    type Target = F;
    fn deref(&self) -> &F {
        &self.0
    }
}

impl<T: MolarWeight<U>, U: EosUnit> MolarWeight<U> for DFT<T> {
    fn molar_weight(&self) -> QuantityArray1<U> {
        (self as &T).molar_weight()
    }
}

struct DefaultIdealGasContribution();
impl<D: DualNum<f64>> IdealGasContributionDual<D> for DefaultIdealGasContribution {
    fn de_broglie_wavelength(&self, _: D, components: usize) -> Array1<D> {
        Array1::zeros(components)
    }
}

impl fmt::Display for DefaultIdealGasContribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ideal gas (default)")
    }
}

impl<T: HelmholtzEnergyFunctional> EquationOfState for DFT<T> {
    fn components(&self) -> usize {
        self.component_index()[self.component_index().len() - 1] + 1
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        (self as &T).subset(component_list)
    }

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        (self as &T).compute_max_density(moles)
    }

    fn residual(&self) -> &[Box<dyn HelmholtzEnergy>] {
        unreachable!()
    }

    fn evaluate_residual<D: DualNum<f64>>(&self, state: &StateHD<D>) -> D
    where
        dyn HelmholtzEnergy: HelmholtzEnergyDual<D>,
    {
        self.contributions()
            .iter()
            .map(|c| (c as &dyn HelmholtzEnergy).helmholtz_energy(state))
            .sum::<D>()
            + self.ideal_chain_contribution().helmholtz_energy(state)
    }

    fn evaluate_residual_contributions<D: DualNum<f64>>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)>
    where
        dyn HelmholtzEnergy: HelmholtzEnergyDual<D>,
    {
        let mut res: Vec<(String, D)> = self
            .contributions()
            .iter()
            .map(|c| {
                (
                    c.to_string(),
                    (c as &dyn HelmholtzEnergy).helmholtz_energy(state),
                )
            })
            .collect();
        res.push((
            self.ideal_chain_contribution().to_string(),
            self.ideal_chain_contribution().helmholtz_energy(state),
        ));
        res
    }

    fn ideal_gas(&self) -> &dyn IdealGasContribution {
        (self as &T).ideal_gas()
    }
}

/// Different representations for molecules within DFT.
pub enum MoleculeShape<'a> {
    /// For spherical molecules, the number of components.
    Spherical(usize),
    /// For non-spherical molecules in a homosegmented approach, the
    /// chain length parameter $m$.
    NonSpherical(&'a Array1<f64>),
    /// For non-spherical molecules in a heterosegmented approach,
    /// the component index for every segment.
    Heterosegmented(&'a Array1<usize>),
}

/// A general Helmholtz energy functional.
pub trait HelmholtzEnergyFunctional: Sized {
    /// Return a slice of [FunctionalContribution]s.
    fn contributions(&self) -> &[Box<dyn FunctionalContribution>];

    /// Return the shape of the molecules and the necessary specifications.
    fn molecule_shape(&self) -> MoleculeShape;

    /// Return a functional for the specified subset of components.
    fn subset(&self, component_list: &[usize]) -> DFT<Self>;

    /// Return the maximum density in Angstrom^-3.
    ///
    /// This value is used as an estimate for a liquid phase for phase
    /// equilibria and other iterations. It is not explicitly meant to
    /// be a mathematical limit for the density (if those exist in the
    /// equation of state anyways).
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64;

    /// Return the ideal gas contribution.
    ///
    /// Per default this function returns an ideal gas contribution
    /// in which the de Broglie wavelength is 1 for every component.
    /// Therefore, the correct ideal gas pressure is obtained even
    /// with no explicit ideal gas term. If a more detailed model is
    /// required (e.g. for the calculation of internal energies) this
    /// function has to be overwritten.
    fn ideal_gas(&self) -> &dyn IdealGasContribution {
        &DefaultIdealGasContribution()
    }

    /// Overwrite this, if the functional consists of heterosegmented chains.
    fn bond_lengths(&self, _temperature: f64) -> UnGraph<(), f64> {
        Graph::with_capacity(0, 0)
    }

    fn weight_functions(&self, temperature: f64) -> Vec<WeightFunctionInfo<f64>> {
        self.contributions()
            .iter()
            .map(|c| c.weight_functions(temperature))
            .collect()
    }

    fn m(&self) -> Cow<Array1<f64>> {
        match self.molecule_shape() {
            MoleculeShape::Spherical(n) => Cow::Owned(Array1::ones(n)),
            MoleculeShape::NonSpherical(m) => Cow::Borrowed(m),
            MoleculeShape::Heterosegmented(component_index) => {
                Cow::Owned(Array1::ones(component_index.len()))
            }
        }
    }

    fn component_index(&self) -> Cow<Array1<usize>> {
        match self.molecule_shape() {
            MoleculeShape::Spherical(n) => Cow::Owned(Array1::from_shape_fn(n, |i| i)),
            MoleculeShape::NonSpherical(m) => Cow::Owned(Array1::from_shape_fn(m.len(), |i| i)),
            MoleculeShape::Heterosegmented(component_index) => Cow::Borrowed(component_index),
        }
    }

    fn ideal_chain_contribution(&self) -> IdealChainContribution {
        IdealChainContribution::new(&self.component_index(), &self.m())
    }
}

impl<T: HelmholtzEnergyFunctional> DFT<T> {
    /// Calculate the grand potential density $\omega$.
    pub fn grand_potential_density<U, D>(
        &self,
        temperature: QuantityScalar<U>,
        density: &QuantityArray<U, D::Larger>,
        convolver: &Rc<dyn Convolver<f64, D>>,
    ) -> EosResult<QuantityArray<U, D>>
    where
        U: EosUnit,
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
    {
        // Calculate residual Helmholtz energy density and functional derivative
        let t = temperature.to_reduced(U::reference_temperature())?;
        let rho = density.to_reduced(U::reference_density())?;
        let (mut f, dfdrho) = self.functional_derivative(t, &rho, convolver)?;

        // Calculate the grand potential density
        for ((rho, dfdrho), &m) in rho
            .outer_iter()
            .zip(dfdrho.outer_iter())
            .zip(self.m().iter())
        {
            f -= &((&dfdrho + m) * &rho);
        }

        let bond_lengths = self.bond_lengths(t);
        for segment in bond_lengths.node_indices() {
            let n = bond_lengths.neighbors(segment).count();
            f += &(&rho.index_axis(Axis(0), segment.index()) * (0.5 * n as f64));
        }

        Ok(f * t * U::reference_pressure())
    }

    pub(crate) fn ideal_gas_contribution<D>(
        &self,
        temperature: f64,
        density: &Array<f64, D::Larger>,
    ) -> Array<f64, D>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
    {
        let n = self.components();
        let ig = self.ideal_gas();
        let lambda = ig.de_broglie_wavelength(temperature, n);
        let mut phi = Array::zeros(density.raw_dim().remove_axis(Axis(0)));
        for (i, rhoi) in density.outer_iter().enumerate() {
            phi += &rhoi.mapv(|rhoi| (rhoi.ln() + lambda[i] - 1.0) * rhoi);
        }
        phi * temperature
    }

    fn ideal_gas_contribution_dual<D>(
        &self,
        temperature: Dual64,
        density: &Array<f64, D::Larger>,
    ) -> Array<Dual64, D>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
    {
        let n = self.components();
        let ig = self.ideal_gas();
        let lambda = ig.de_broglie_wavelength(temperature, n);
        let mut phi = Array::zeros(density.raw_dim().remove_axis(Axis(0)));
        for (i, rhoi) in density.outer_iter().enumerate() {
            phi += &rhoi.mapv(|rhoi| (lambda[i] + rhoi.ln() - 1.0) * rhoi);
        }
        phi * temperature
    }

    fn intrinsic_helmholtz_energy_density<D, N>(
        &self,
        temperature: N,
        density: &Array<f64, D::Larger>,
        convolver: &Rc<dyn Convolver<N, D>>,
    ) -> EosResult<Array<N, D>>
    where
        N: DualNum<f64> + ScalarOperand,
        dyn FunctionalContribution: FunctionalContributionDual<N>,
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
    {
        let density_dual = density.mapv(N::from);
        let weighted_densities = convolver.weighted_densities(&density_dual);
        let functional_contributions = self.contributions();
        let mut helmholtz_energy_density: Array<N, D> = self
            .ideal_chain_contribution()
            .calculate_helmholtz_energy_density(&density.mapv(N::from))?;
        for (c, wd) in functional_contributions.iter().zip(weighted_densities) {
            let nwd = wd.shape()[0];
            let ngrid = wd.len() / nwd;
            helmholtz_energy_density
                .view_mut()
                .into_shape(ngrid)
                .unwrap()
                .add_assign(&c.calculate_helmholtz_energy_density(
                    temperature,
                    wd.into_shape((nwd, ngrid)).unwrap().view(),
                )?);
        }
        Ok(helmholtz_energy_density * temperature)
    }

    /// Calculate the entropy density $s$.
    ///
    /// Untested with heterosegmented functionals.
    pub fn entropy_density<D>(
        &self,
        temperature: f64,
        density: &Array<f64, D::Larger>,
        convolver: &Rc<dyn Convolver<Dual64, D>>,
        contributions: Contributions,
    ) -> EosResult<Array<f64, D>>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
    {
        let temperature_dual = Dual64::from(temperature).derive();
        let mut helmholtz_energy_density =
            self.intrinsic_helmholtz_energy_density(temperature_dual, density, convolver)?;
        match contributions {
            Contributions::Total => {
                helmholtz_energy_density += &self.ideal_gas_contribution_dual::<D>(temperature_dual, density);
            },
            Contributions::ResidualNpt|Contributions::IdealGas => panic!("Entropy density can only be calculated for Contributions::Residual or Contributions::Total"),
            Contributions::ResidualNvt => (),
        }
        Ok(helmholtz_energy_density.mapv(|f| -f.eps[0]))
    }

    /// Calculate the individual contributions to the entropy density.
    ///
    /// Untested with heterosegmented functionals.
    pub fn entropy_density_contributions<D>(
        &self,
        temperature: f64,
        density: &Array<f64, D::Larger>,
        convolver: &Rc<dyn Convolver<Dual64, D>>,
    ) -> EosResult<Vec<Array<f64, D>>>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
        <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
    {
        let density_dual = density.mapv(Dual64::from);
        let temperature_dual = Dual64::from(temperature).derive();
        let weighted_densities = convolver.weighted_densities(&density_dual);
        let functional_contributions = self.contributions();
        let mut helmholtz_energy_density: Vec<Array<Dual64, D>> =
            Vec::with_capacity(functional_contributions.len() + 1);
        helmholtz_energy_density.push(
            self.ideal_chain_contribution()
                .calculate_helmholtz_energy_density(&density.mapv(Dual64::from))?,
        );

        for (c, wd) in functional_contributions.iter().zip(weighted_densities) {
            let nwd = wd.shape()[0];
            let ngrid = wd.len() / nwd;
            helmholtz_energy_density.push(
                c.calculate_helmholtz_energy_density(
                    temperature_dual,
                    wd.into_shape((nwd, ngrid)).unwrap().view(),
                )?
                .into_shape(density.raw_dim().remove_axis(Axis(0)))
                .unwrap(),
            );
        }
        Ok(helmholtz_energy_density
            .iter()
            .map(|v| v.mapv(|f| -(f * temperature_dual).eps[0]))
            .collect())
    }

    /// Calculate the internal energy density $u$.
    ///
    /// Untested with heterosegmented functionals.
    pub fn internal_energy_density<D>(
        &self,
        temperature: f64,
        density: &Array<f64, D::Larger>,
        external_potential: &Array<f64, D::Larger>,
        convolver: &Rc<dyn Convolver<Dual64, D>>,
        contributions: Contributions,
    ) -> EosResult<Array<f64, D>>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
    {
        let temperature_dual = Dual64::from(temperature).derive();
        let mut helmholtz_energy_density_dual =
            self.intrinsic_helmholtz_energy_density(temperature_dual, density, convolver)?;
        match contributions {
                Contributions::Total => {
                    helmholtz_energy_density_dual += &self.ideal_gas_contribution_dual::<D>(temperature_dual, density);
                },
                Contributions::ResidualNpt|Contributions::IdealGas => panic!("Internal energy density can only be calculated for Contributions::Residual or Contributions::Total"),
                Contributions::ResidualNvt => (),
            }
        let helmholtz_energy_density = helmholtz_energy_density_dual
            .mapv(|f| f.re - f.eps[0] * temperature)
            + (external_potential * density).sum_axis(Axis(0)) * temperature;
        Ok(helmholtz_energy_density)
    }

    /// Calculate the (residual) functional derivative $\frac{\delta\mathcal{F}}{\delta\rho_i(\mathbf{r})}$.
    #[allow(clippy::type_complexity)]
    pub fn functional_derivative<D>(
        &self,
        temperature: f64,
        density: &Array<f64, D::Larger>,
        convolver: &Rc<dyn Convolver<f64, D>>,
    ) -> EosResult<(Array<f64, D>, Array<f64, D::Larger>)>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
    {
        let weighted_densities = convolver.weighted_densities(density);
        let contributions = self.contributions();
        let mut partial_derivatives = Vec::with_capacity(contributions.len());
        let mut helmholtz_energy_density = Array::zeros(density.raw_dim().remove_axis(Axis(0)));
        for (c, wd) in contributions.iter().zip(weighted_densities) {
            let nwd = wd.shape()[0];
            let ngrid = wd.len() / nwd;
            let mut phi = Array::zeros(density.raw_dim().remove_axis(Axis(0)));
            let mut pd = Array::zeros(wd.raw_dim());
            c.first_partial_derivatives(
                temperature,
                wd.into_shape((nwd, ngrid)).unwrap(),
                phi.view_mut().into_shape(ngrid).unwrap(),
                pd.view_mut().into_shape((nwd, ngrid)).unwrap(),
            )?;
            partial_derivatives.push(pd);
            helmholtz_energy_density += &phi;
        }
        Ok((
            helmholtz_energy_density,
            convolver.functional_derivative(&partial_derivatives),
        ))
    }

    /// Calculate the bond integrals $I_{\alpha\alpha'}(\mathbf{r})$
    pub fn bond_integrals<D>(
        &self,
        temperature: f64,
        functional_derivative: &Array<f64, D::Larger>,
        convolver: &Rc<dyn Convolver<f64, D>>,
    ) -> Array<f64, D::Larger>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
    {
        // calculate weight functions
        let bond_lengths = self.bond_lengths(temperature).into_edge_type();
        let mut bond_weight_functions = bond_lengths.map(
            |_, _| (),
            |_, &l| WeightFunction::new_scaled(arr1(&[l]), WeightFunctionShape::Delta),
        );
        for n in bond_lengths.node_indices() {
            for e in bond_lengths.edges(n) {
                bond_weight_functions.add_edge(
                    e.target(),
                    e.source(),
                    WeightFunction::new_scaled(arr1(&[*e.weight()]), WeightFunctionShape::Delta),
                );
            }
        }

        let expdfdrho = functional_derivative.mapv(|x| (-x).exp());
        let mut i_graph: Graph<_, Option<Array<f64, D>>, Directed> =
            bond_weight_functions.map(|_, _| (), |_, _| None);

        let bonds = i_graph.edge_count();
        let mut calc = 0;

        // go through the whole graph until every bond has been calculated
        while calc < bonds {
            let mut edge_id = None;
            let mut i1 = None;

            // find the first bond that can be calculated
            'nodes: for node in i_graph.node_indices() {
                for edge in i_graph.edges(node) {
                    // skip already calculated bonds
                    if edge.weight().is_some() {
                        continue;
                    }

                    // if all bonds from the neighboring segment are calculated calculate the bond
                    let edges = i_graph
                        .edges(edge.target())
                        .filter(|e| e.target().index() != node.index());
                    if edges.clone().all(|e| e.weight().is_some()) {
                        edge_id = Some(edge.id());
                        let i0 = edges.fold(
                            expdfdrho
                                .index_axis(Axis(0), edge.target().index())
                                .to_owned(),
                            |acc: Array<f64, D>, e| acc * e.weight().as_ref().unwrap(),
                        );
                        i1 =
                            Some(convolver.convolve(i0.clone(), &bond_weight_functions[edge.id()]));
                        break 'nodes;
                    }
                }
            }
            if let Some(edge_id) = edge_id {
                i_graph[edge_id] = i1;
                calc += 1;
            } else {
                panic!("Cycle in molecular structure detected!")
            }
        }

        let mut i = Array::ones(functional_derivative.raw_dim());
        for node in i_graph.node_indices() {
            for edge in i_graph.edges(node) {
                i.index_axis_mut(Axis(0), node.index())
                    .mul_assign(edge.weight().as_ref().unwrap());
            }
        }

        i
    }
}
