use crate::association::AssociationFunctional;
use crate::convolver::{Convolver, ConvolverFFT};
use crate::functional_contribution::*;
use crate::geometry::Grid;
use crate::ideal_chain_contribution::IdealChainContribution;
use crate::solver::{Evaluate, Tape};
use crate::weight_functions::{WeightFunction, WeightFunctionInfo, WeightFunctionShape};
use feos_core::{EquationOfState, Residual, ResidualDyn, StateHD};
use nalgebra::{DVector, dvector};
use ndarray::*;
use num_dual::*;
use petgraph::Directed;
use petgraph::graph::{Graph, UnGraph};
use petgraph::visit::EdgeRef;
use rustdct::DctNum;
use std::borrow::Cow;
use std::ops::Deref;
use std::sync::Arc;

impl<I: Clone, F: HelmholtzEnergyFunctionalDyn> HelmholtzEnergyFunctionalDyn
    for EquationOfState<Vec<I>, F>
{
    type Contribution<'a>
        = F::Contribution<'a>
    where
        Self: 'a;
    type Association<'a>
        = F::Association<'a>
    where
        Self: 'a;

    fn contributions<'a>(&'a self) -> impl Iterator<Item = Self::Contribution<'a>> {
        self.residual.contributions()
    }

    fn molecule_shape(&self) -> MoleculeShape<'_> {
        self.residual.molecule_shape()
    }

    fn bond_lengths<N: DualNum<Primitive = f64> + Copy>(&self, temperature: N) -> UnGraph<(), N> {
        self.residual.bond_lengths(temperature)
    }

    fn association(&self) -> Option<Self::Association<'_>> {
        self.residual.association()
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

    /// The association functional ([NoAssociation](crate::NoAssociation) for functionals without
    /// association).
    type Association<'a>: AssociationFunctional
    where
        Self: 'a;

    /// Return a slice of [FunctionalContribution]s.
    fn contributions<'a>(&'a self) -> impl Iterator<Item = Self::Contribution<'a>>;

    /// Return the shape of the molecules and the necessary specifications.
    fn molecule_shape(&self) -> MoleculeShape<'_>;

    /// Overwrite this, if the functional consists of heterosegmented chains.
    fn bond_lengths<N: DualNum<Primitive = f64> + Copy>(&self, _temperature: N) -> UnGraph<(), N> {
        Graph::with_capacity(0, 0)
    }

    /// Overwrite this, if the functional contains association.
    fn association(&self) -> Option<Self::Association<'_>> {
        None
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
    fn bond_lengths<N: DualNum<Primitive = f64> + Copy>(&self, _temperature: N) -> UnGraph<(), N> {
        Graph::with_capacity(0, 0)
    }

    /// Return the association functional, if the functional contains association.
    fn association(&self) -> Option<impl AssociationFunctional + '_>;

    fn weight_functions<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
    ) -> Vec<WeightFunctionInfo<N>> {
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

    /// Return the densities of all segments given the partial densities of the components.
    fn segment_densities<D: Copy>(&self, partial_density: &DVector<D>) -> Array1<D> {
        self.component_index()
            .iter()
            .map(|&c| partial_density[c])
            .collect()
    }

    fn ideal_chain_contribution(&self) -> IdealChainContribution {
        IdealChainContribution::new(&self.component_index(), &self.m())
    }

    /// Convolvers of the functional contributions, of the association functional, and of the
    /// association kernel on the grid `grid` for the temperature `temperature` (any number type,
    /// e.g., dual numbers for temperature derivatives).
    #[expect(clippy::type_complexity)]
    fn convolvers<D, N>(
        &self,
        grid: &Grid,
        temperature: N,
    ) -> (
        Arc<dyn Convolver<N, D>>,
        Option<Arc<dyn Convolver<N, D>>>,
        Option<Arc<dyn Convolver<N, D>>>,
    )
    where
        D: Dimension + 'static,
        D::Larger: Dimension<Smaller = D>,
        <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
        N: DctNum + DualNum<Primitive = f64>,
    {
        let convolver = ConvolverFFT::plan(grid, &self.weight_functions(temperature));
        let association_convolver = self
            .association()
            .map(|a| ConvolverFFT::plan(grid, &[a.weight_functions(temperature)]));
        let association_kernel_convolver = self
            .association()
            .and_then(|a| a.kernel_weight_functions(temperature))
            .map(|kernel| ConvolverFFT::plan(grid, &[kernel]));
        (
            convolver,
            association_convolver,
            association_kernel_convolver,
        )
    }

    /// Calculate the bond integrals $I_{\alpha\alpha'}(\mathbf{r})$
    fn bond_integrals<D, N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
        exponential: &Array<N, D::Larger>,
        convolver: &dyn Convolver<N, D>,
    ) -> Array<N, D::Larger>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
    {
        self.bond_integrals_on_tape(&mut Evaluate, temperature, exponential, convolver)
    }

    /// Calculate the bond integrals $I_{\alpha\alpha'}(\mathbf{r})$ of a heterosegmented functional
    /// (the product of all bond integrals of every segment) on a tape.
    fn bond_integrals_on_tape<D, N, T>(
        &self,
        tape: &mut T,
        temperature: N,
        exponential: &Array<N, D::Larger>,
        convolver: &dyn Convolver<N, D>,
    ) -> Array<N, D::Larger>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
        N: DualNum<Primitive = f64> + Copy,
        T: Tape<N>,
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
                            |acc, e| tape.mul(&acc, e.weight().as_ref().unwrap()),
                        );
                        i1 = Some(convolver.convolve(i0, &bond_weight_functions[edge.id()]));
                        break 'nodes;
                    }
                }
            }
            let edge_id = edge_id.expect("Cycle in molecular structure detected!");
            i_graph[edge_id] = i1;
            calc += 1;
        }

        // product of all bond integrals of a segment
        let mut i = tape.constant(Array::ones(exponential.raw_dim()));
        for node in i_graph.node_indices() {
            for edge in i_graph.edges(node) {
                let row = tape.mul(
                    &i.index_axis(Axis(0), node.index()).to_owned(),
                    edge.weight().as_ref().unwrap(),
                );
                i.index_axis_mut(Axis(0), node.index()).assign(&row);
            }
        }
        i
    }

    fn evaluate_bulk<D: DualNum<Primitive = f64> + Copy>(
        &self,
        state: &StateHD<D>,
        mut f: impl FnMut(&'static str, D),
    ) {
        let density = self.segment_densities(&state.partial_density);
        for c in self.contributions() {
            f(
                c.name(),
                c.bulk_helmholtz_energy_density(state.temperature, &density),
            )
        }
        f(
            "Ideal chain",
            self.ideal_chain_contribution()
                .bulk_helmholtz_energy_density(&state.partial_density),
        );
        if let Some(a) = self.association() {
            f(
                "Association",
                a.bulk_helmholtz_energy_density(state.temperature, &density),
            )
        }
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

    fn bond_lengths<N: DualNum<Primitive = f64> + Copy>(&self, temperature: N) -> UnGraph<(), N> {
        F::bond_lengths(self.deref(), temperature)
    }

    fn association(&self) -> Option<impl AssociationFunctional + '_> {
        F::association(self.deref())
    }
}
