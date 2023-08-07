use crate::adsorption::FluidParameters;
use crate::convolver::Convolver;
use crate::functional_contribution::*;
use crate::ideal_chain_contribution::IdealChainContribution;
use crate::solvation::PairPotential;
use crate::weight_functions::{WeightFunction, WeightFunctionInfo, WeightFunctionShape};
use feos_core::si::MolarWeight;
use feos_core::{
    Components, DeBroglieWavelength, EosResult, EquationOfState, HelmholtzEnergy,
    HelmholtzEnergyDual, IdealGas, Residual, StateHD,
};
use ndarray::*;
use num_dual::*;
use petgraph::graph::{Graph, UnGraph};
use petgraph::visit::EdgeRef;
use petgraph::Directed;
use std::borrow::Cow;
use std::ops::{Deref, MulAssign};
use std::sync::Arc;

impl<I: Components + Send + Sync, F: HelmholtzEnergyFunctional> HelmholtzEnergyFunctional
    for EquationOfState<I, F>
{
    fn contributions(&self) -> &[Box<dyn FunctionalContribution>] {
        self.residual.contributions()
    }

    fn molecule_shape(&self) -> MoleculeShape {
        self.residual.molecule_shape()
    }

    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.residual.molar_weight()
    }

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.residual.compute_max_density(moles)
    }
}

impl<I, F: PairPotential> PairPotential for EquationOfState<I, F> {
    fn pair_potential(&self, i: usize, r: &Array1<f64>, temperature: f64) -> Array2<f64> {
        self.residual.pair_potential(i, r, temperature)
    }
}

impl<I: Components + Send + Sync, F: FluidParameters> FluidParameters for EquationOfState<I, F> {
    fn epsilon_k_ff(&self) -> Array1<f64> {
        self.residual.epsilon_k_ff()
    }

    fn sigma_ff(&self) -> &Array1<f64> {
        self.residual.sigma_ff()
    }
}

/// Wrapper struct for the [HelmholtzEnergyFunctional] trait.
///
/// Needed (for now) to generically implement the `Residual`
/// trait for Helmholtz energy functionals.
#[derive(Clone)]
pub struct DFT<F>(pub F);

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

impl<F> DFT<F> {
    pub fn ideal_gas<I>(self, ideal_gas: I) -> DFT<EquationOfState<I, F>> {
        DFT(EquationOfState::new(Arc::new(ideal_gas), Arc::new(self.0)))
    }
}

impl<F: HelmholtzEnergyFunctional> Components for DFT<F> {
    fn components(&self) -> usize {
        self.0.components()
    }

    fn subset(&self, component_list: &[usize]) -> Self {
        Self(self.0.subset(component_list))
    }
}

impl<F: HelmholtzEnergyFunctional> Residual for DFT<F> {
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        self.0.compute_max_density(moles)
    }

    fn contributions(&self) -> &[Box<dyn HelmholtzEnergy>] {
        unreachable!()
    }

    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        self.0.molar_weight()
    }

    fn evaluate_residual<D: DualNum<f64> + Copy>(&self, state: &StateHD<D>) -> D
    where
        dyn HelmholtzEnergy: HelmholtzEnergyDual<D>,
    {
        self.0
            .contributions()
            .iter()
            .map(|c| (c as &dyn HelmholtzEnergy).helmholtz_energy(state))
            .sum::<D>()
            + self.ideal_chain_contribution().helmholtz_energy(state)
    }

    fn evaluate_residual_contributions<D: DualNum<f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(String, D)>
    where
        dyn HelmholtzEnergy: HelmholtzEnergyDual<D>,
    {
        let mut res: Vec<(String, D)> = self
            .0
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
}

impl<F: HelmholtzEnergyFunctional + IdealGas> IdealGas for DFT<F> {
    fn ideal_gas_model(&self) -> &dyn DeBroglieWavelength {
        self.0.ideal_gas_model()
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
pub trait HelmholtzEnergyFunctional: Components + Sized + Send + Sync {
    /// Return a slice of [FunctionalContribution]s.
    fn contributions(&self) -> &[Box<dyn FunctionalContribution>];

    /// Return the shape of the molecules and the necessary specifications.
    fn molecule_shape(&self) -> MoleculeShape;

    /// Molar weight of all components.
    ///
    /// Enables calculation of (mass) specific properties.
    fn molar_weight(&self) -> MolarWeight<Array1<f64>>;

    /// Return the maximum density in Angstrom^-3.
    ///
    /// This value is used as an estimate for a liquid phase for phase
    /// equilibria and other iterations. It is not explicitly meant to
    /// be a mathematical limit for the density (if those exist in the
    /// equation of state anyways).
    fn compute_max_density(&self, moles: &Array1<f64>) -> f64;

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

    /// Calculate the (residual) intrinsic functional derivative $\frac{\delta\mathcal{F}}{\delta\rho_i(\mathbf{r})}$.
    #[allow(clippy::type_complexity)]
    fn functional_derivative<D>(
        &self,
        temperature: f64,
        density: &Array<f64, D::Larger>,
        convolver: &Arc<dyn Convolver<f64, D>>,
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

    #[allow(clippy::type_complexity)]
    fn functional_derivative_dual<D>(
        &self,
        temperature: f64,
        density: &Array<f64, D::Larger>,
        convolver: &Arc<dyn Convolver<Dual64, D>>,
    ) -> EosResult<(Array<Dual64, D>, Array<Dual64, D::Larger>)>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
    {
        let temperature_dual = Dual64::from(temperature).derivative();
        let density_dual = density.mapv(Dual64::from);
        let weighted_densities = convolver.weighted_densities(&density_dual);
        let contributions = self.contributions();
        let mut partial_derivatives = Vec::with_capacity(contributions.len());
        let mut helmholtz_energy_density = Array::zeros(density.raw_dim().remove_axis(Axis(0)));
        for (c, wd) in contributions.iter().zip(weighted_densities) {
            let nwd = wd.shape()[0];
            let ngrid = wd.len() / nwd;
            let mut phi = Array::zeros(density.raw_dim().remove_axis(Axis(0)));
            let mut pd = Array::zeros(wd.raw_dim());
            c.first_partial_derivatives_dual(
                temperature_dual,
                wd.into_shape((nwd, ngrid)).unwrap(),
                phi.view_mut().into_shape(ngrid).unwrap(),
                pd.view_mut().into_shape((nwd, ngrid)).unwrap(),
            )?;
            partial_derivatives.push(pd);
            helmholtz_energy_density += &phi;
        }
        Ok((
            helmholtz_energy_density,
            convolver.functional_derivative(&partial_derivatives) * temperature_dual,
        ))
    }

    /// Calculate the bond integrals $I_{\alpha\alpha'}(\mathbf{r})$
    fn bond_integrals<D>(
        &self,
        temperature: f64,
        exponential: &Array<f64, D::Larger>,
        convolver: &Arc<dyn Convolver<f64, D>>,
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
                            exponential
                                .index_axis(Axis(0), edge.target().index())
                                .to_owned(),
                            |acc: Array<f64, D>, e| acc * e.weight().as_ref().unwrap(),
                        );
                        i1 = Some(convolver.convolve(i0, &bond_weight_functions[edge.id()]));
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

        let mut i = Array::ones(exponential.raw_dim());
        for node in i_graph.node_indices() {
            for edge in i_graph.edges(node) {
                i.index_axis_mut(Axis(0), node.index())
                    .mul_assign(edge.weight().as_ref().unwrap());
            }
        }

        i
    }
}
