use crate::adsorption::FluidParameters;
use crate::convolver::Convolver;
use crate::functional_contribution::*;
use crate::ideal_chain_contribution::IdealChainContribution;
use crate::solvation::PairPotential;
use crate::weight_functions::{WeightFunction, WeightFunctionInfo, WeightFunctionShape};
use feos_core::{EquationOfState, FeosResult, Residual, ResidualDyn, StateHD};
use nalgebra::{DVector, dvector};
use ndarray::*;
use num_dual::*;
use petgraph::Directed;
use petgraph::graph::{Graph, UnGraph};
use petgraph::visit::EdgeRef;
use std::borrow::Cow;
use std::ops::{Deref, MulAssign};
use std::sync::Arc;

impl<I: Clone, F: HelmholtzEnergyFunctionalDyn> HelmholtzEnergyFunctionalDyn
    for EquationOfState<Vec<I>, F>
{
    type Contribution<'a>
        = F::Contribution<'a>
    where
        Self: 'a;

    fn contributions<'a>(&'a self) -> impl Iterator<Item = Self::Contribution<'a>> {
        self.residual.contributions()
    }

    fn molecule_shape(&self) -> MoleculeShape<'_> {
        self.residual.molecule_shape()
    }

    fn bond_lengths<N: DualNum<f64> + Copy>(&self, temperature: N) -> UnGraph<(), N> {
        self.residual.bond_lengths(temperature)
    }
}

impl<I, F: PairPotential> PairPotential for EquationOfState<I, F> {
    fn pair_potential(&self, i: usize, r: &Array1<f64>, temperature: f64) -> Array2<f64> {
        self.residual.pair_potential(i, r, temperature)
    }
}

impl<I, F: FluidParameters> FluidParameters for EquationOfState<I, F> {
    fn epsilon_k_ff(&self) -> DVector<f64> {
        self.residual.epsilon_k_ff()
    }

    fn sigma_ff(&self) -> DVector<f64> {
        self.residual.sigma_ff()
    }
}

/// Different representations for molecules within DFT.
pub enum MoleculeShape<'a> {
    /// For spherical molecules, the number of components.
    Spherical(usize),
    /// For non-spherical molecules in a homosegmented approach, the
    /// chain length parameter $m$.
    NonSpherical(&'a DVector<f64>),
    /// For non-spherical molecules in a heterosegmented approach,
    /// the component index for every segment.
    Heterosegmented(&'a DVector<usize>),
}

/// A general Helmholtz energy functional.
pub trait HelmholtzEnergyFunctionalDyn: ResidualDyn {
    type Contribution<'a>: FunctionalContribution
    where
        Self: 'a;

    /// Return a slice of [FunctionalContribution]s.
    fn contributions<'a>(&'a self) -> impl Iterator<Item = Self::Contribution<'a>>;

    /// Return the shape of the molecules and the necessary specifications.
    fn molecule_shape(&self) -> MoleculeShape<'_>;

    /// Overwrite this, if the functional consists of heterosegmented chains.
    fn bond_lengths<N: DualNum<f64> + Copy>(&self, _temperature: N) -> UnGraph<(), N> {
        Graph::with_capacity(0, 0)
    }
}

/// A general Helmholtz energy functional.
pub trait HelmholtzEnergyFunctional: Residual {
    type Contribution<'a>: FunctionalContribution
    where
        Self: 'a;

    /// Return a slice of [FunctionalContribution]s.
    fn contributions<'a>(&'a self) -> impl Iterator<Item = Self::Contribution<'a>>;

    /// Return the shape of the molecules and the necessary specifications.
    fn molecule_shape(&self) -> MoleculeShape<'_>;

    /// Overwrite this, if the functional consists of heterosegmented chains.
    fn bond_lengths<N: DualNum<f64> + Copy>(&self, _temperature: N) -> UnGraph<(), N> {
        Graph::with_capacity(0, 0)
    }

    fn weight_functions(&self, temperature: f64) -> Vec<WeightFunctionInfo<f64>> {
        self.contributions()
            .map(|c| c.weight_functions(temperature))
            .collect()
    }

    fn m(&self) -> Cow<'_, [f64]> {
        match self.molecule_shape() {
            MoleculeShape::Spherical(n) => Cow::Owned(vec![1.0; n]),
            MoleculeShape::NonSpherical(m) => Cow::Borrowed(m.as_slice()),
            MoleculeShape::Heterosegmented(component_index) => {
                Cow::Owned(vec![1.0; component_index.len()])
            }
        }
    }

    fn component_index(&self) -> Cow<'_, [usize]> {
        match self.molecule_shape() {
            MoleculeShape::Spherical(n) => Cow::Owned((0..n).collect()),
            MoleculeShape::NonSpherical(m) => Cow::Owned((0..m.len()).collect()),
            MoleculeShape::Heterosegmented(component_index) => {
                Cow::Borrowed(component_index.as_slice())
            }
        }
    }

    fn ideal_chain_contribution(&self) -> IdealChainContribution {
        IdealChainContribution::new(&self.component_index(), &self.m())
    }

    /// Calculate the (residual) intrinsic functional derivative $\frac{\delta\mathcal{\beta F}}{\delta\rho_i(\mathbf{r})}$.
    #[expect(clippy::type_complexity)]
    fn functional_derivative<D, N: DualNum<f64> + Copy>(
        &self,
        temperature: N,
        density: &Array<N, D::Larger>,
        convolver: &Arc<dyn Convolver<N, D>>,
    ) -> FeosResult<(Array<N, D>, Array<N, D::Larger>)>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
    {
        let weighted_densities = convolver.weighted_densities(density);
        let contributions = self.contributions();
        let mut partial_derivatives = Vec::new();
        let mut helmholtz_energy_density = Array::zeros(density.raw_dim().remove_axis(Axis(0)));
        for (c, wd) in contributions.into_iter().zip(weighted_densities) {
            let nwd = wd.shape()[0];
            let ngrid = wd.len() / nwd;
            let mut phi = Array::zeros(density.raw_dim().remove_axis(Axis(0)));
            let mut pd = Array::zeros(wd.raw_dim());
            c.first_partial_derivatives(
                temperature,
                wd.into_shape_with_order((nwd, ngrid)).unwrap(),
                phi.view_mut().into_shape_with_order(ngrid).unwrap(),
                pd.view_mut().into_shape_with_order((nwd, ngrid)).unwrap(),
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
    fn bond_integrals<D, N: DualNum<f64> + Copy>(
        &self,
        temperature: N,
        exponential: &Array<N, D::Larger>,
        convolver: &Arc<dyn Convolver<N, D>>,
    ) -> Array<N, D::Larger>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
    {
        // calculate weight functions
        let bond_lengths = self.bond_lengths(temperature).into_edge_type();
        let mut bond_weight_functions = bond_lengths.map(
            |_, _| (),
            |_, &l| WeightFunction::new_scaled(dvector![l], WeightFunctionShape::Delta),
        );
        for n in bond_lengths.node_indices() {
            for e in bond_lengths.edges(n) {
                bond_weight_functions.add_edge(
                    e.target(),
                    e.source(),
                    WeightFunction::new_scaled(dvector![*e.weight()], WeightFunctionShape::Delta),
                );
            }
        }

        let mut i_graph: Graph<_, Option<Array<N, D>>, Directed> =
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
                            |acc: Array<N, D>, e| acc * e.weight().as_ref().unwrap(),
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

    fn evaluate_bulk<D: DualNum<f64> + Copy>(&self, state: &StateHD<D>) -> Vec<(&'static str, D)> {
        let mut res: Vec<_> = self
            .contributions()
            .map(|c| (c.name(), c.bulk_helmholtz_energy_density(state)))
            .collect();
        res.push((
            "Ideal chain",
            self.ideal_chain_contribution()
                .bulk_helmholtz_energy_density(&state.partial_density),
        ));
        res
    }
}

impl<C: Deref<Target = F> + Clone, F: HelmholtzEnergyFunctionalDyn + ResidualDyn + 'static>
    HelmholtzEnergyFunctional for C
{
    type Contribution<'a>
        = F::Contribution<'a>
    where
        Self: 'a;

    fn contributions<'a>(&'a self) -> impl Iterator<Item = Self::Contribution<'a>> {
        F::contributions(self.deref())
    }

    fn molecule_shape(&self) -> MoleculeShape<'_> {
        F::molecule_shape(self.deref())
    }

    fn bond_lengths<N: DualNum<f64> + Copy>(&self, temperature: N) -> UnGraph<(), N> {
        F::bond_lengths(self.deref(), temperature)
    }
}
