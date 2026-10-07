use crate::convolver::{Convolver, ConvolverFFT};
use crate::geometry::Grid;
use crate::ideal_chain_contribution::IdealChainContribution;
use crate::profile::MAX_POTENTIAL;
use crate::weight_functions::{WeightFunction, WeightFunctionInfo, WeightFunctionShape};
use crate::{DFTSolverLog, functional_contribution::*};
use feos_core::{
    EquationOfState, FeosError, FeosResult, ReferenceSystem, Residual, ResidualDyn, StateHD,
};
use nalgebra::{DVector, dvector};
use ndarray::*;
use num_dual::*;
use petgraph::Directed;
use petgraph::graph::{Graph, UnGraph};
use petgraph::visit::EdgeRef;
use quantity::{Energy, Temperature};
use std::borrow::Cow;
use std::ops::{AddAssign, Deref, MulAssign};

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

    fn bond_lengths_homo<N: DualNum<Primitive = f64> + Copy>(&self, temperature: N) -> DVector<N> {
        self.residual.bond_lengths_homo(temperature)
    }

    fn bond_lengths_hetero<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
    ) -> UnGraph<(), N> {
        self.residual.bond_lengths_hetero(temperature)
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

    /// Overwrite this, if the functional consists of homosegmented chains.
    fn bond_lengths_homo<N: DualNum<Primitive = f64> + Copy>(&self, _temperature: N) -> DVector<N> {
        DVector::zeros(self.components())
    }

    /// Overwrite this, if the functional consists of heterosegmented chains.
    fn bond_lengths_hetero<N: DualNum<Primitive = f64> + Copy>(
        &self,
        _temperature: N,
    ) -> UnGraph<(), N> {
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

    /// Overwrite this, if the functional consists of homosegmented chains.
    fn bond_lengths_homo<N: DualNum<Primitive = f64> + Copy>(&self, _temperature: N) -> DVector<N> {
        DVector::zeros(self.components())
    }

    /// Overwrite this, if the functional consists of heterosegmented chains.
    fn bond_lengths_hetero<N: DualNum<Primitive = f64> + Copy>(
        &self,
        _temperature: N,
    ) -> UnGraph<(), N> {
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

    /// Calculate the (residual) intrinsic functional derivative $\frac{\delta\mathcal{\beta F}}{\delta\rho_i(\mathbf{r})}$.
    #[expect(clippy::type_complexity)]
    fn functional_derivative<D, N: DualNumCopy<Primitive = f64>>(
        &self,
        temperature: N,
        density: &Array<N, D::Larger>,
        convolver: &dyn Convolver<N, D>,
        solver_log: &mut DFTSolverLog,
    ) -> FeosResult<(Array<N, D>, Array<N, D::Larger>)>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
    {
        // calculate weighted densities
        let weighted_densities = solver_log.time_function("weighted densities", || {
            convolver.weighted_densities(density)
        });

        // calculate partial derivatives
        let contributions = self.contributions();
        let mut partial_derivatives = Vec::new();
        let mut helmholtz_energy_density = Array::zeros(density.raw_dim().remove_axis(Axis(0)));
        solver_log.time_function("partial derivatives", || {
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
            Ok::<_, FeosError>(())
        })?;

        // calculate functional derivative
        let functional_derivative = solver_log.time_function("functional derivative", || {
            convolver.functional_derivative(&partial_derivatives)
        });

        Ok((helmholtz_energy_density, functional_derivative))
    }

    fn second_partial_derivatives<D>(
        &self,
        temperature: f64,
        density: &Array<f64, D::Larger>,
        convolver: &dyn Convolver<f64, D>,
    ) -> FeosResult<Vec<Array<f64, <D::Larger as Dimension>::Larger>>>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
    {
        let contributions = self.contributions();

        let weighted_densities = convolver.weighted_densities(density);

        let mut second_partial_derivatives = Vec::new();
        for (c, wd) in contributions.into_iter().zip(&weighted_densities) {
            let nwd = wd.shape()[0];
            let ngrid = wd.len() / nwd;
            let mut phi = Array::zeros(density.raw_dim().remove_axis(Axis(0)));
            let mut pd = Array::zeros(wd.raw_dim());
            let dim = wd.shape();
            let dim: Vec<_> = std::iter::once(&nwd).chain(dim).cloned().collect();
            let mut pd2 = Array::zeros(dim).into_dimensionality().unwrap();
            c.second_partial_derivatives(
                temperature,
                wd.view().into_shape_with_order((nwd, ngrid)).unwrap(),
                phi.view_mut().into_shape_with_order(ngrid).unwrap(),
                pd.view_mut().into_shape_with_order((nwd, ngrid)).unwrap(),
                pd2.view_mut()
                    .into_shape_with_order((nwd, nwd, ngrid))
                    .unwrap(),
            )?;
            second_partial_derivatives.push(pd2);
        }
        Ok(second_partial_derivatives)
    }

    /// Return the external potential, corrected such that an ideal gas of homosegmented
    /// chains reproduces the density profile $\rho_\alpha(\mathbf{r})\propto e^{-\beta V_\alpha^\mathrm{ext}(\mathbf{r})}$
    /// of the original external potential.
    ///
    /// The corrected potential is temperature dependent and has to be used in place of
    /// the original potential when creating the density profile.
    fn chain_corrected_external_potential<D>(
        &self,
        grid: &Grid,
        temperature: Temperature,
        external_potential: &Energy<Array<f64, D::Larger>>,
    ) -> Energy<Array<f64, D::Larger>>
    where
        D: Dimension + 'static,
        D::Larger: Dimension<Smaller = D>,
        <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
    {
        let t = temperature.to_reduced();
        let convolver = ConvolverFFT::<f64, D>::plan(grid, &self.weight_functions(t));
        let cap = |x: &mut f64| *x = x.min(MAX_POTENTIAL);

        let mut potential = external_potential.to_reduced() / t;
        potential.map_inplace(cap);

        let bonds = self.bond_lengths_homo(t);
        let m = self.m();
        for ((mut v, &l), &m) in potential.outer_iter_mut().zip(bonds.iter()).zip(m.iter()) {
            let w = WeightFunction::new_scaled(dvector![l], WeightFunctionShape::Delta);
            let rho = v.mapv(|x| (-x).exp());
            let lambda = convolver
                .convolve(rho.clone(), &w)
                .mapv(|l| l.abs() + f64::EPSILON);
            let rho_lambda = &rho / &lambda;
            v += &((-rho_lambda.mapv(f64::ln) + convolver.convolve(rho_lambda, &w) - 1.0)
                * (m - 1.0));
        }

        potential.map_inplace(cap);
        Energy::from_reduced(potential * t)
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
        // calculate weight functions
        let bond_lengths = self.bond_lengths_hetero(temperature).into_edge_type();
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

    fn delta_bond_integrals<D>(
        &self,
        temperature: f64,
        exponential: &Array<f64, D::Larger>,
        delta_functional_derivative: &Array<f64, D::Larger>,
        convolver: &dyn Convolver<f64, D>,
    ) -> Array<f64, D::Larger>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
    {
        // calculate weight functions
        let bond_lengths = self.bond_lengths_hetero(temperature).into_edge_type();
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

        let mut i_graph: Graph<_, Option<Array<f64, D>>, Directed> =
            bond_weight_functions.map(|_, _| (), |_, _| None);
        let mut delta_i_graph: Graph<_, Option<Array<f64, D>>, Directed> =
            bond_weight_functions.map(|_, _| (), |_, _| None);

        let bonds = i_graph.edge_count();
        let mut calc = 0;

        // go through the whole graph until every bond has been calculated
        while calc < bonds {
            let mut edge_id = None;
            let mut i1 = None;
            let mut delta_i1 = None;

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
                    let delta_edges = delta_i_graph
                        .edges(edge.target())
                        .filter(|e| e.target().index() != node.index());
                    if edges.clone().all(|e| e.weight().is_some()) {
                        edge_id = Some(edge.id());
                        let i0 = edges.fold(
                            exponential
                                .index_axis(Axis(0), edge.target().index())
                                .to_owned(),
                            |acc: Array<f64, _>, e| acc * e.weight().as_ref().unwrap(),
                        );
                        let delta_i0 = delta_edges.fold(
                            -&delta_functional_derivative
                                .index_axis(Axis(0), edge.target().index()),
                            |acc: Array<f64, _>, delta_e| acc + delta_e.weight().as_ref().unwrap(),
                        ) * &i0;
                        i1 = Some(convolver.convolve(i0, &bond_weight_functions[edge.id()]));
                        delta_i1 = Some(
                            (convolver.convolve(delta_i0, &bond_weight_functions[edge.id()])
                                / i1.as_ref().unwrap())
                            .mapv(|x| if x.is_finite() { x } else { 0.0 }),
                        );
                        break 'nodes;
                    }
                }
            }
            if let Some(edge_id) = edge_id {
                i_graph[edge_id] = i1;
                delta_i_graph[edge_id] = delta_i1;
                calc += 1;
            } else {
                panic!("Cycle in molecular structure detected!")
            }
        }

        let mut delta_i = Array::zeros(exponential.raw_dim());
        for node in delta_i_graph.node_indices() {
            for edge in delta_i_graph.edges(node) {
                delta_i
                    .index_axis_mut(Axis(0), node.index())
                    .add_assign(edge.weight().as_ref().unwrap());
            }
        }

        delta_i
    }

    fn evaluate_bulk<D: DualNum<Primitive = f64> + Copy>(
        &self,
        state: &StateHD<D>,
    ) -> Vec<(&'static str, D)> {
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

    fn bond_lengths_homo<N: DualNum<Primitive = f64> + Copy>(&self, temperature: N) -> DVector<N> {
        F::bond_lengths_homo(self.deref(), temperature)
    }

    fn bond_lengths_hetero<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
    ) -> UnGraph<(), N> {
        F::bond_lengths_hetero(self.deref(), temperature)
    }
}
