//! The Euler-Lagrange equation of a [DFTProfile] as [NonlinearSystem].
//!
//! The unknowns are the logarithms of the density profiles and of the fractions of non-bonded
//! association sites, stacked as $x=(\ln\rho,\ln X)$. The residual consists of the logarithmic
//! Euler-Lagrange equation $\ln\rho^\mathrm{projected}-\ln\rho$ (set to 0 where the external
//! potential is overwhelming) and the mass action equations in the same form,
//! $\ln X^\mathrm{projected}-\ln X$ with $X^\mathrm{projected}=\frac{1}{1+S}$.
use super::DFTSolverLog;
use super::newton::NonlinearSystem;
use super::tape::{Evaluate, Tape};
use crate::Convolver;
use crate::association::ContactValues;
use crate::profile::MAX_POTENTIAL;
use crate::{AssociationFunctional, DFTProfile, DFTSpecification, HelmholtzEnergyFunctional};
use feos_core::{FeosError, FeosResult, ReferenceSystem};
use ndarray::{Array, Array1, ArrayView, Axis, Dimension, Slice, Zip, arr1, concatenate};
use num_dual::{DualNum, DualNumCopy};

/// Maximum change of the logarithms of the density and X in a Newton step.
const NEWTON_MAX_STEP: f64 = 10.0;

/// Residuals of the Euler-Lagrange equation and of the mass action equations.
pub(crate) struct EulerLagrangeResidual<D: Dimension> {
    /// Residual of the Euler-Lagrange equation, $\ln\rho^\mathrm{projected}-\ln\rho$ (logarithmic)
    /// or $\rho^\mathrm{projected}-\rho$ (segments x grid), 0 where the external potential is
    /// overwhelming.
    pub density: Array<f64, D::Larger>,
    /// Residual of the mass action equations $\ln X^\mathrm{projected}-\ln X$ (sites x grid).
    pub mass_action: Array<f64, D::Larger>,
    /// Norm of $\rho^\mathrm{projected}-\rho$ and of the mass action residual (convergence
    /// criterion of all solvers).
    pub norm: f64,
    /// Norm of $\rho^\mathrm{projected}-\rho$ only.
    pub density_norm: f64,
}

impl<D, F> DFTProfile<D, F>
where
    D: Dimension + 'static,
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
    F: HelmholtzEnergyFunctional,
{
    pub(crate) fn euler_lagrange_equation(
        &self,
        density: &Array<f64, D::Larger>,
        non_bonded_fractions: &Array<f64, D::Larger>,
        log: bool,
        solver_log: &mut DFTSolverLog,
    ) -> FeosResult<EulerLagrangeResidual<D>> {
        let x = Self::stack_variables(density, non_bonded_fractions);
        let (residual, _) = self.euler_lagrange(&mut Evaluate, &x, solver_log)?;
        let norm = NonlinearSystem::convergence_norm(self, &x, &residual);
        if !norm.is_finite() {
            return Err(FeosError::IterationFailed("Euler-Lagrange equation".into()));
        }

        // residual in terms of the density: rho_projected - rho
        let (res_ln_rho, mass_action) = (
            self.density_rows(&residual),
            self.association_rows(&residual),
        );
        let mut res_rho = (&self.density_rows(&x) + &res_ln_rho).mapv(f64::exp) - density;
        self.mask_external_potential(&mut res_rho);
        let density_norm = (res_rho.mapv(|r| r * r).sum() / res_rho.len() as f64).sqrt();

        Ok(EulerLagrangeResidual {
            density: if log { res_ln_rho.to_owned() } else { res_rho },
            mass_action: mass_action.to_owned(),
            norm,
            density_norm,
        })
    }

    /// Logarithms of the fugacities determined from the specification for the density `density`
    /// and the fractions of non-bonded sites `non_bonded_fractions`.
    pub(crate) fn ln_fugacity(
        &self,
        density: &Array<f64, D::Larger>,
        non_bonded_fractions: &Array<f64, D::Larger>,
    ) -> FeosResult<Array1<f64>> {
        let x = Self::stack_variables(density, non_bonded_fractions);
        let (_, ln_fugacity) = self.euler_lagrange(&mut Evaluate, &x, &mut DFTSolverLog::new())?;
        Ok(ln_fugacity)
    }

    /// Set the rows of the Euler-Lagrange equation (residual, Jacobian-vector products) to 0 at
    /// points where the external potential is overwhelming.
    pub(crate) fn mask_external_potential<N: DualNum<Primitive = f64>>(
        &self,
        rows: &mut Array<N, D::Larger>,
    ) {
        rows.iter_mut()
            .zip(self.external_potential.iter())
            .filter(|&(_, &p)| p >= MAX_POTENTIAL)
            .for_each(|(r, _)| *r = N::zero());
    }

    /// Unknowns $x=(\ln\rho,\ln X)$ for the density `density` (bounded from below by the smallest
    /// positive number) and the fractions of non-bonded sites `non_bonded_fractions`.
    pub(crate) fn stack_variables(
        density: &Array<f64, D::Larger>,
        non_bonded_fractions: &Array<f64, D::Larger>,
    ) -> Array<f64, D::Larger> {
        let ln_rho = density.mapv(|r| r.max(f64::MIN_POSITIVE).ln());
        let ln_x = non_bonded_fractions.mapv(f64::ln);
        concatenate(Axis(0), &[ln_rho.view(), ln_x.view()]).unwrap()
    }

    /// Rows of the stacked unknowns $x=(\ln\rho,\ln X)$ (or of residuals and tangents with the
    /// same layout) that belong to the density.
    pub(crate) fn density_rows<'a, N>(
        &self,
        x: &'a Array<N, D::Larger>,
    ) -> ArrayView<'a, N, D::Larger> {
        x.slice_axis(Axis(0), Slice::from(..self.density.shape()[0]))
    }

    /// Rows of the stacked unknowns $x=(\ln\rho,\ln X)$ (or of residuals and tangents with the
    /// same layout) that belong to the fractions of non-bonded association sites. Uses
    /// `slice_axis` because empty views from `split_at` cannot be copied (ndarray 0.17).
    pub(crate) fn association_rows<'a, N>(
        &self,
        x: &'a Array<N, D::Larger>,
    ) -> ArrayView<'a, N, D::Larger> {
        x.slice_axis(Axis(0), Slice::from(self.density.shape()[0]..))
    }

    /// Residual of the Euler-Lagrange equation and the mass action equations, together with the
    /// logarithms of the fugacities determined from the specification.
    #[expect(clippy::type_complexity)]
    pub(crate) fn euler_lagrange<T: Tape>(
        &self,
        tape: &mut T,
        x: &Array<f64, D::Larger>,
        log: &mut DFTSolverLog,
    ) -> Result<(Array<f64, D::Larger>, Array1<f64>), T::Error<FeosError>> {
        self.euler_lagrange_dual(
            tape,
            self.temperature.to_reduced(),
            self.convolver.as_ref(),
            self.association_convolver.as_deref(),
            self.association_kernel_convolver.as_deref(),
            x,
            log,
        )
    }

    /// Generic implementation of [Self::euler_lagrange] at the temperature `temperature`
    /// for automatic differentiation with dual numbers.
    #[expect(clippy::too_many_arguments, clippy::type_complexity)]
    pub(crate) fn euler_lagrange_dual<N, T>(
        &self,
        tape: &mut T,
        temperature: N,
        convolver: &dyn Convolver<N, D>,
        association_convolver: Option<&dyn Convolver<N, D>>,
        kernel_convolver: Option<&dyn Convolver<N, D>>,
        x: &Array<N, D::Larger>,
        log: &mut DFTSolverLog,
    ) -> Result<(Array<N, D::Larger>, Array1<N>), T::Error<FeosError>>
    where
        N: DualNumCopy<Primitive = f64>,
        T: Tape<N>,
    {
        let t = temperature;
        let ln_rho = self.density_rows(x).to_owned();
        let ln_x = self.association_rows(x).to_owned();

        // functional derivative of the functional contributions
        let rho = tape.exp(&ln_rho);
        let weighted_densities =
            log.time_function("weighted densities", || convolver.weighted_densities(&rho));
        let partial_derivatives = log.time_function("partial derivatives", || {
            self.functional
                .contributions()
                .zip(&weighted_densities)
                .map(|(c, wd)| Ok(tape.local_derivatives(t, wd, &c)?.1))
                .collect::<Result<Vec<_>, _>>()
        })?;
        let mut dfdrho = log.time_function("functional derivative", || {
            convolver.functional_derivative(&partial_derivatives)
        });

        // association functional and mass action equations ln(1/(1+S)) - ln(X)
        let mass_action_residual = if let (Some(a), Some(convolver)) =
            (self.functional.association(), association_convolver)
        {
            let wd = log.time_function("weighted densities", || {
                convolver.weighted_densities(&rho).remove(0)
            });
            let (values, derivatives) = log.time_function("contact values", || {
                T::infallible(tape.local_derivatives(t, &wd, &ContactValues(&a)))
            });
            let (mass_action_residual, _, partial_derivatives) =
                log.time_function("association", || {
                    a.local_association(
                        tape,
                        t,
                        &values,
                        Some(&derivatives),
                        &ln_x,
                        kernel_convolver,
                    )
                });
            dfdrho += &log.time_function("functional derivative", || {
                convolver.functional_derivative(&[partial_derivatives.unwrap()])
            });
            mass_action_residual
        } else {
            Array::zeros(ln_x.raw_dim())
        };

        // external potential (beta V, scaled from the temperature of the profile) and chain length
        let ratio = N::from(self.temperature.to_reduced()) / t;
        dfdrho += &tape.constant(self.external_potential.mapv(|v| ratio * v));
        for (mut df, &m) in dfdrho.outer_iter_mut().zip(self.functional.m().iter()) {
            df.mapv_inplace(|df| df / m);
        }

        // bond integrals
        let exp_dfdrho = tape.exp(&dfdrho.mapv(|df| -df));
        let bonds = log.time_function("bond integrals", || {
            self.functional
                .bond_integrals_on_tape(tape, t, &exp_dfdrho, convolver)
        });

        // fugacity from the specification
        let z = self
            .grid
            .integrate_reduced_comp(&tape.mul(&exp_dfdrho, &bonds));
        let ln_fugacity = match &self.specification {
            DFTSpecification::ChemicalPotential(ln_f) => tape.constant(ln_f.mapv(N::from)),
            DFTSpecification::Moles(n) => -tape.ln(&z) + tape.constant(n.mapv(|n| N::from(n.ln()))),
            DFTSpecification::TotalMoles(n, ln_f) => {
                let sum = ln_f.iter().zip(&z).map(|(&ln_f, &z)| z * ln_f.exp()).sum();
                let ln_sum = tape.ln(&arr1(&[sum]))[0];
                tape.constant(ln_f.mapv(|ln_f| N::from(ln_f + n.ln())))
                    .mapv(|ln_f| ln_f - ln_sum)
            }
        };

        // logarithmic Euler-Lagrange equation ln(rho_projected) - ln(rho)
        let mut res = tape.ln(&bonds) - dfdrho - &ln_rho;
        for (mut r, &f) in res.outer_iter_mut().zip(&ln_fugacity) {
            r.mapv_inplace(|r| r + f);
        }
        self.mask_external_potential(&mut res);

        Ok((
            concatenate(Axis(0), &[res.view(), mass_action_residual.view()]).unwrap(),
            ln_fugacity,
        ))
    }
}

impl<D, F> NonlinearSystem<D::Larger> for DFTProfile<D, F>
where
    D: Dimension + 'static,
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
    F: HelmholtzEnergyFunctional,
{
    fn residual<T: Tape>(
        &self,
        tape: &mut T,
        x: &Array<f64, D::Larger>,
        log: &mut DFTSolverLog,
    ) -> Result<Array<f64, D::Larger>, T::Error<FeosError>> {
        Ok(self.euler_lagrange(tape, x, log)?.0)
    }

    /// Norm of the difference between the density and the projected density, together with the
    /// mass action residual.
    fn convergence_norm(&self, x: &Array<f64, D::Larger>, residual: &Array<f64, D::Larger>) -> f64 {
        let mass_action_residual = self.association_rows(residual);
        let density_residual = Zip::from(&self.density_rows(x))
            .and(&self.density_rows(residual))
            .map_collect(|&x, &r| x.exp() * (1.0 - r.exp()));
        ((density_residual.mapv(|r| r * r).sum() + mass_action_residual.mapv(|r| r * r).sum())
            / residual.len() as f64)
            .sqrt()
    }

    /// Changes of the logarithms of the density and of X limited to [NEWTON_MAX_STEP].
    fn step_length(&self, _: &Array<f64, D::Larger>, delta: &Array<f64, D::Larger>) -> f64 {
        let max_step = delta.fold(0.0f64, |m, &d| m.max(d.abs()));
        (NEWTON_MAX_STEP / max_step).min(1.0)
    }
}
