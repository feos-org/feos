//! Association (Wertheim TPT1) in classical DFT, formulated with Michelsen's Q function.
//!
//! The fractions of non-bonded sites $X_\alpha(\mathbf{r})$ are treated as independent
//! variables. The Helmholtz energy density is
//! $$q=\sum_\alpha\tilde\rho_\alpha\left(\ln X_\alpha-X_\alpha+1\right)-\frac{1}{2}\sum_\alpha\tilde\rho_\alpha X_\alpha S_\alpha$$
//! with
//! $$S_\alpha=\sum_{k\ni\alpha}c_k\sqrt{y_k}\left(K_k*\sqrt{y_k}\tilde\rho_\beta X_\beta\right)$$
//! where the sum runs over all binary association records $k$ that contain site $\alpha$, and
//! $\beta$ is the association partner of $\alpha$ in record $k$. The association strength $c_k$,
//! the contact value $y_k$, and the association kernel $K_k$ are symmetric and therefore given per
//! record. The kernel is either the identity (Yu-Wu, WDA) or a normalized contact shell (iSAFT).
//! It is specified via [WeightFunctionInfo]s, so a local kernel corresponds to the local density.
//!
//! All functions work on the [AssociationSites] (sites and binary association records without
//! the model records).
//! Sites are ordered as A sites, B sites, C sites. Quantities per record are ordered as
//! `binary_ab` followed by `binary_cc`. The fields that are convolved with the association kernel
//! are ordered in three blocks: for every AB record (A, B) the field of B (contributes to $S_A$),
//! for every AB record the field of A (contributes to $S_B$), and for every CC record (C, C') the
//! field of C' (contributes to $S_C$). CC records contain both orderings (C, C') and (C', C), as
//! in the bulk implementation.
use crate::convolver::Convolver;
use crate::solver::{
    DFTSolverLog, Evaluate, LocalFunction, NonlinearSystem, Tape, flatten, newton_krylov, unflatten,
};
use crate::weight_functions::{WeightFunction, WeightFunctionInfo, WeightFunctionShape};
use feos_core::parameter::{AssociationParameters, AssociationSite, BinaryParameters};
use feos_core::{FeosError, Verbosity};
use nalgebra::{DMatrix, DVector};
use ndarray::{
    Array, Array1, Array2, ArrayView, ArrayView2, Axis, Dimension, Ix0, RemoveAxis, Slice,
    concatenate, stack,
};
use num_dual::{Dual64, DualNum, DualNumCopy};
use std::convert::Infallible;
use std::ops::AddAssign;

/// Rows of `a` multiplied by the factors `c`.
fn scale_rows<D: RemoveAxis, N: DualNum<Primitive = f64> + Copy>(
    mut a: Array<N, D>,
    c: &[N],
) -> Array<N, D> {
    for (mut a, &c) in a.outer_iter_mut().zip(c) {
        a.mapv_inplace(|a| a * c);
    }
    a
}

/// Sum of the rows of `a` into the rows `rows` of an array with `n` rows.
fn scatter_rows<D: RemoveAxis, N: DualNum<Primitive = f64> + Copy>(
    a: &Array<N, D>,
    rows: &[usize],
    n: usize,
) -> Array<N, D> {
    let mut dim = a.raw_dim();
    dim[0] = n;
    let mut s = Array::zeros(dim);
    for (a, &r) in a.outer_iter().zip(rows) {
        s.index_axis_mut(Axis(0), r).add_assign(&a);
    }
    s
}

/// The stacked contact values of an association functional as local function of the weighted
/// densities, which the tape differentiates.
pub(crate) struct ContactValues<'a, A: ?Sized>(pub(crate) &'a A);

impl<A: AssociationFunctional + ?Sized> LocalFunction for ContactValues<'_, A> {
    type Error = Infallible;

    /// Densities $\tilde\rho_i$ and square roots of the contact values $\sqrt{y_k}$ from
    /// [AssociationFunctional::contact_values], stacked (rows x points).
    fn evaluate<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> Result<Array2<N>, Infallible> {
        let (rho, y) = self.0.contact_values(temperature, weighted_densities);
        Ok(concatenate(Axis(0), &[rho.view(), y.mapv(|y| y.sqrt()).view()]).unwrap())
    }
}

/// Maximum number of Newton iterations for the mass action equations in a homogeneous system.
const BULK_MASS_ACTION_MAX_ITER: usize = 50;
/// Tolerance for the residual of the mass action equations in a homogeneous system.
const BULK_MASS_ACTION_TOL: f64 = 1e-14;
/// Maximum change of $\ln X$ in a Newton step for the mass action equations in a homogeneous
/// system.
const BULK_MASS_ACTION_MAX_STEP: f64 = 2.0;

/// Maximum number of Newton iterations for the mass action equations.
const MASS_ACTION_MAX_ITER: usize = 50;
/// Maximum number of GMRES iterations per Newton iteration for the mass action equations.
const MASS_ACTION_MAX_ITER_GMRES: usize = 50;
/// Tolerance for the residual of the mass action equations.
const MASS_ACTION_TOL: f64 = 1e-12;
/// Maximum change of $\ln X$ in a Newton step for the mass action equations.
const MASS_ACTION_MAX_STEP: f64 = 1.0;

/// Mass action equations $\ln\frac{1}{1+S}-\ln X=0$ for fixed density profiles.
pub struct MassAction<'a, A: ?Sized, D: Dimension> {
    association: &'a A,
    temperature: f64,
    contact_values: Array<f64, D::Larger>,
    n_sites: usize,
    kernel_convolver: Option<&'a dyn Convolver<f64, D>>,
}

impl<'a, A, D> MassAction<'a, A, D>
where
    A: AssociationFunctional + ?Sized,
    D: Dimension,
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    pub fn new(
        association: &'a A,
        temperature: f64,
        weighted_densities: &Array<f64, D::Larger>,
        kernel_convolver: Option<&'a dyn Convolver<f64, D>>,
    ) -> Self {
        let Ok(contact_values) =
            ContactValues(association).evaluate(temperature, flatten(weighted_densities));
        let contact_values = unflatten(contact_values, weighted_densities.raw_dim());
        Self {
            association,
            temperature,
            contact_values,
            n_sites: association.sites().n_sites(),
            kernel_convolver,
        }
    }

    /// Fractions of non-bonded sites (sites x grid) that solve the mass action equations,
    /// starting from $X=1$.
    ///
    /// The result is used as initial guess, so it is returned even if the solver does not
    /// converge.
    pub fn solve(&self) -> Array<f64, D::Larger> {
        let mut dim = self.contact_values.raw_dim();
        dim[0] = self.n_sites;
        let mut ln_x = Array::zeros(dim);
        let _ = newton_krylov(
            self,
            &mut ln_x,
            MASS_ACTION_MAX_ITER,
            MASS_ACTION_MAX_ITER_GMRES,
            MASS_ACTION_TOL,
            &mut DFTSolverLog::new(),
            Verbosity::None,
        );
        ln_x.mapv(f64::exp)
    }
}

impl<A, D> NonlinearSystem<D::Larger> for MassAction<'_, A, D>
where
    A: AssociationFunctional + ?Sized,
    D: Dimension,
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    fn residual<T: Tape>(
        &self,
        tape: &mut T,
        ln_x: &Array<f64, D::Larger>,
        _: &mut DFTSolverLog,
    ) -> Result<Array<f64, D::Larger>, T::Error<FeosError>> {
        let contact_values = tape.constant(self.contact_values.clone());
        let (mass_action_residual, _, _) = self.association.local_association(
            tape,
            self.temperature,
            &contact_values,
            None,
            ln_x,
            self.kernel_convolver,
        );
        Ok(mass_action_residual)
    }

    /// Changes of $\ln X$ limited to [MASS_ACTION_MAX_STEP].
    fn step_length(&self, _: &Array<f64, D::Larger>, delta: &Array<f64, D::Larger>) -> f64 {
        let max_step = delta.fold(0.0f64, |m, &d| m.max(d.abs()));
        (MASS_ACTION_MAX_STEP / max_step).min(1.0)
    }
}

/// Result of evaluating an association functional for given density profiles and fractions of
/// non-bonded sites.
pub struct AssociationEvaluation<N, D: Dimension> {
    /// Helmholtz energy density (Q function density).
    pub helmholtz_energy_density: Array<N, D>,
    /// Functional derivative $\frac{\delta\beta F}{\delta\rho_i}$ at fixed $X$ (segments x grid).
    pub functional_derivative: Array<N, D::Larger>,
    /// Residual of the mass action equations $\ln\frac{1}{1+S_\alpha}-\ln X_\alpha$ (sites x
    /// grid).
    pub mass_action_residual: Array<N, D::Larger>,
}

/// Association sites of an association functional and the site indices of the binary
/// association records, independent of the model records.
pub struct AssociationSites<'a> {
    /// A sites.
    pub a: &'a [AssociationSite],
    /// B sites.
    pub b: &'a [AssociationSite],
    /// C sites.
    pub c: &'a [AssociationSite],
    /// Indices of the A site and the B site of every AB record.
    pub ab: Vec<[usize; 2]>,
    /// Indices of the two C sites of every CC record.
    pub cc: Vec<[usize; 2]>,
}

impl<'a> AssociationSites<'a> {
    pub fn new<R>(parameters: &'a AssociationParameters<R>) -> Self {
        let ids =
            |records: &[BinaryParameters<R, ()>]| records.iter().map(|r| [r.id1, r.id2]).collect();
        Self {
            a: &parameters.sites_a,
            b: &parameters.sites_b,
            c: &parameters.sites_c,
            ab: ids(&parameters.binary_ab),
            cc: ids(&parameters.binary_cc),
        }
    }

    /// Total number of sites.
    pub fn n_sites(&self) -> usize {
        self.a.len() + self.b.len() + self.c.len()
    }
}

/// An association functional formulated with Michelsen's Q function.
pub trait AssociationFunctional: Sync + Send {
    /// Association sites and binary association records.
    fn sites(&self) -> AssociationSites<'_>;

    /// Weight functions for the weighted densities required in [AssociationFunctional::contact_values].
    fn weight_functions<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
    ) -> WeightFunctionInfo<N>;

    /// Contact distances $d_k$ for every record that define the association kernel as a
    /// normalized contact shell, $K_k(\mathbf{r})=\frac{\delta(d_k-|\mathbf{r}|)}{4\pi d_k^2}$.
    /// `None` for a local kernel.
    fn contact_distances<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
    ) -> Option<Array1<N>>;

    /// Densities $\tilde\rho_i$ and contact values $y_k$ of the cavity correlation function.
    fn contact_values<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
        weighted_densities: ArrayView2<N>,
    ) -> (Array2<N>, Array2<N>);

    /// Association strengths without the contact value.
    fn association_strength<N: DualNum<Primitive = f64> + Copy>(&self, temperature: N)
    -> Array1<N>;

    /// Weight functions of the association kernel for all convolved fields (`None` for a local
    /// kernel).
    fn kernel_weight_functions<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
    ) -> Option<WeightFunctionInfo<N>> {
        let d = self.contact_distances(temperature)?;
        let n_ab = self.sites().ab.len();
        let records: Vec<_> = (0..n_ab).chain(0..n_ab).chain(n_ab..d.len()).collect();
        let d = DVector::from_fn(records.len(), |i, _| d[records[i]]);
        let w = WeightFunction::new_scaled(d, WeightFunctionShape::Delta);
        Some(WeightFunctionInfo::new(records.len(), false).add(w, false))
    }

    /// Helmholtz energy density of a homogeneous system with the segment densities `density` at
    /// the equilibrium fractions of non-bonded sites.
    fn bulk_helmholtz_energy_density<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
        density: &Array1<N>,
    ) -> N {
        let wd = self
            .weight_functions(temperature)
            .weight_constants()
            .dot(density);

        let ns = self.sites().n_sites();

        // mass action residual and Helmholtz energy density for the fractions of non-bonded sites
        // x (the association kernel is the identity in the bulk)
        let Ok(contact_values) =
            ContactValues(self).evaluate(temperature, wd.view().insert_axis(Axis(1)));
        let contact_values = contact_values.remove_axis(Axis(1));
        let local = |x: &Array1<N>| {
            let ln_x = x.mapv(|x| x.ln());
            let (mass_action, helmholtz_energy_density, _) = self.local_association::<Ix0, _, _>(
                &mut Evaluate,
                temperature,
                &contact_values,
                None,
                &ln_x,
                None,
            );
            (mass_action, helmholtz_energy_density.into_scalar())
        };

        // Newton iteration for the real part in ln X starting from X=1 (the Jacobian of the mass
        // action residual is calculated column by column with Dual64)
        let t = Dual64::from(temperature.re());
        let contact_values_re = contact_values.mapv(|c| Dual64::from(c.re()));
        let residual_and_jacobian = |ln_x: &Array1<f64>| {
            let mut residual = DVector::zeros(ns);
            let mut jacobian = DMatrix::zeros(ns, ns);
            for b in 0..ns {
                let mut ln_x = ln_x.mapv(Dual64::from);
                ln_x[b].eps = 1.0;
                let (r, _, _) = self.local_association::<Ix0, _, _>(
                    &mut Evaluate,
                    t,
                    &contact_values_re,
                    None,
                    &ln_x,
                    None,
                );
                for a in 0..ns {
                    residual[a] = r[a].re;
                    jacobian[(a, b)] = r[a].eps;
                }
            }
            (residual, jacobian)
        };
        let mut ln_x = Array1::zeros(ns);
        let (mut residual, mut jacobian) = residual_and_jacobian(&ln_x);
        for _ in 0..BULK_MASS_ACTION_MAX_ITER {
            if residual.amax() < BULK_MASS_ACTION_TOL {
                break;
            }
            let Some(du) = jacobian.clone().lu().solve(&(-&residual)) else {
                break;
            };
            ln_x.iter_mut().zip(du.iter()).for_each(|(u, du)| {
                *u += du.clamp(-BULK_MASS_ACTION_MAX_STEP, BULK_MASS_ACTION_MAX_STEP)
            });
            (residual, jacobian) = residual_and_jacobian(&ln_x);
        }
        let jacobian_inv = jacobian.try_inverse().unwrap();

        // derivatives (one Newton step per order with the real Jacobian)
        let mut x = ln_x.mapv(|u| N::from(u.exp()));
        for _ in 0..N::NDERIV {
            let (r, _) = local(&x);
            x = Array1::from_shape_fn(ns, |a| {
                let du = (0..ns).fold(N::zero(), |acc, b| acc - r[b] * jacobian_inv[(a, b)]);
                x[a] * du.exp()
            });
        }

        local(&x).1
    }

    /// Evaluate the association functional including derivatives.
    fn evaluate<D, N>(
        &self,
        temperature: N,
        density: &Array<N, D::Larger>,
        x: &Array<N, D::Larger>,
        convolver: &dyn Convolver<N, D>,
        kernel_convolver: Option<&dyn Convolver<N, D>>,
    ) -> AssociationEvaluation<N, D>
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
        <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
        N: DualNumCopy<Primitive = f64>,
    {
        let wd = convolver.weighted_densities(density).remove(0);
        let ln_x = x.mapv(|x| x.ln());
        let Ok((rho_y_values, rho_y_derivatives)) =
            Evaluate.local_derivatives(temperature, &wd, &ContactValues(self));
        let (mass_action_residual, helmholtz_energy_density, partial_derivatives) = self
            .local_association(
                &mut Evaluate,
                temperature,
                &rho_y_values,
                Some(&rho_y_derivatives),
                &ln_x,
                kernel_convolver,
            );
        AssociationEvaluation {
            helmholtz_energy_density,
            functional_derivative: convolver.functional_derivative(&[partial_derivatives.unwrap()]),
            mass_action_residual,
        }
    }

    /// Local part of the association functional for the stacked contact values
    /// (see [ContactValues]) and, if given, their derivatives with respect to the
    /// weighted densities: the residual of the mass action equations, the Helmholtz energy
    /// density, and the partial derivatives of the Helmholtz energy density.
    ///
    /// The partial derivatives are taken at fixed $K*h$ of
    /// $\tilde q=\sum_\alpha\tilde\rho_\alpha\left(\ln X_\alpha-X_\alpha+1-X_\alpha S_\alpha\right)$,
    /// i.e., $q$ without the factor $\frac{1}{2}$, which accounts for the dependence of $K*h$ on
    /// the weighted densities (the association term is bilinear and symmetric).
    #[expect(clippy::type_complexity)]
    fn local_association<D, N, T>(
        &self,
        tape: &mut T,
        temperature: N,
        rho_y_values: &Array<N, D::Larger>,
        rho_y_derivatives: Option<&Array<N, D::Larger>>,
        ln_x: &Array<N, D::Larger>,
        kernel_convolver: Option<&dyn Convolver<N, D>>,
    ) -> (
        Array<N, D::Larger>,
        Array<N, D>,
        Option<Array<N, D::Larger>>,
    )
    where
        D: Dimension,
        D::Larger: Dimension<Smaller = D>,
        N: DualNum<Primitive = f64> + Copy,
        T: Tape<N>,
    {
        // segment and multiplicity of every site (A sites, B sites, C sites)
        let p = self.sites();
        let sites: Vec<_> = p.a.iter().chain(p.b).chain(p.c).collect();
        let segments: Vec<_> = sites.iter().map(|s| s.assoc_comp).collect();
        let multiplicities: Vec<_> = sites.iter().map(|s| N::from(s.n)).collect();
        let n_sites = sites.len();

        // record, partner site, and target site of every convolved field (see module
        // documentation for the order)
        let [na, nb, n_ab] = [p.a.len(), p.b.len(), p.ab.len()];
        let ab = p.ab.iter().enumerate();
        let cc = p.cc.iter().enumerate();
        let fields: Vec<_> = ab
            .clone()
            .map(|(k, &[i, j])| [k, na + j, i])
            .chain(ab.map(|(k, &[i, j])| [k, i, na + j]))
            .chain(cc.map(|(k, &[i, j])| [n_ab + k, na + nb + j, na + nb + i]))
            .collect();
        let [records, partners, targets] =
            [0, 1, 2].map(|i| fields.iter().map(|f| f[i]).collect::<Vec<_>>());
        let c = self.association_strength(temperature);
        let c: Vec<_> = records.iter().map(|&k| c[k]).collect();

        let n = rho_y_values.shape()[0];
        let n_segments = n - p.ab.len() - p.cc.len();
        let site_densities_and_sqrt_y = |v: ArrayView<N, D::Larger>| {
            let (rho, sqrt_y) = v.split_at(Axis(0), n_segments);
            let rho = scale_rows(rho.select(Axis(0), &segments), &multiplicities);
            (rho, sqrt_y.select(Axis(0), &records))
        };

        // association fields, association sums, and mass action equations
        let (rho, sqrt_y) = site_densities_and_sqrt_y(rho_y_values.view());
        let x = tape.exp(ln_x);
        let rho_x = tape.mul(&rho, &x);
        let h = tape.mul(&sqrt_y, &rho_x.select(Axis(0), &partners));
        let kh = match kernel_convolver {
            Some(kc) => kc.weighted_densities(&h).remove(0),
            None => h,
        };
        let s = scatter_rows(&tape.mul(&scale_rows(sqrt_y, &c), &kh), &targets, n_sites);
        let mass_action_residual = -tape.ln_1p(&s) - ln_x;

        // Helmholtz energy density and (if the derivatives of the contact values are given)
        // partial derivatives of q~ at fixed K*h
        let x_s = tape.mul(&x, &s);
        let one = tape.constant(Array::ones(x.raw_dim()));
        let g = ln_x - &x - &x_s + one;
        let helmholtz_energy_density = tape
            .mul(&rho, &(&g + &x_s.mapv(|x_s| x_s * 0.5)))
            .sum_axis(Axis(0));
        let partial_derivatives = rho_y_derivatives.map(|derivatives| {
            let n_wd = derivatives.shape()[0] / n;
            let partial_derivatives: Vec<_> = (0..n_wd)
                .map(|i| {
                    let block = derivatives.slice_axis(Axis(0), Slice::from(i * n..(i + 1) * n));
                    let (drho, dsqrt_y) = site_densities_and_sqrt_y(block);
                    let ds =
                        scatter_rows(&tape.mul(&scale_rows(dsqrt_y, &c), &kh), &targets, n_sites);
                    let pd = tape.mul(&drho, &g) - tape.mul(&rho_x, &ds);
                    pd.sum_axis(Axis(0))
                })
                .collect();
            let partial_derivatives: Vec<_> =
                partial_derivatives.iter().map(|pd| pd.view()).collect();
            stack(Axis(0), &partial_derivatives).unwrap()
        });
        (
            mass_action_residual,
            helmholtz_energy_density,
            partial_derivatives,
        )
    }
}

/// Placeholder for functionals without association (uninhabited, to be replaced by `!` once
/// the never type is stable).
pub enum NoAssociation {}

impl AssociationFunctional for NoAssociation {
    fn sites(&self) -> AssociationSites<'_> {
        match *self {}
    }

    fn weight_functions<N: DualNum<Primitive = f64> + Copy>(&self, _: N) -> WeightFunctionInfo<N> {
        match *self {}
    }

    fn contact_distances<N: DualNum<Primitive = f64> + Copy>(&self, _: N) -> Option<Array1<N>> {
        match *self {}
    }

    fn contact_values<N: DualNum<Primitive = f64> + Copy>(
        &self,
        _: N,
        _: ArrayView2<N>,
    ) -> (Array2<N>, Array2<N>) {
        match *self {}
    }

    fn association_strength<N: DualNum<Primitive = f64> + Copy>(&self, _: N) -> Array1<N> {
        match *self {}
    }
}
