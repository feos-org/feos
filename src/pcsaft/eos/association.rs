use super::PcSaftParameters;
use feos_core::{EosError, HelmholtzEnergyDual, StateHD};
use ndarray::*;
use num_dual::linalg::{norm, LU};
use num_dual::*;
use std::f64::consts::{FRAC_PI_3, PI};
use std::fmt;
use std::rc::Rc;

pub struct Association {
    pub parameters: Rc<PcSaftParameters>,
}

pub struct CrossAssociation {
    pub parameters: Rc<PcSaftParameters>,
    pub max_iter: usize,
    pub tol: f64,
}

fn association_strength<D: DualNum<f64>>(
    p: &PcSaftParameters,
    temperature: D,
    r: &Array1<D>,
    n2: D,
    n3i: D,
    xi: D,
    ai: usize,
    aj: usize,
) -> D {
    let k = r[ai] * r[aj] / (r[ai] + r[aj]) * (n2 * n3i * 2.0);
    n3i * (k * xi * (k / 18.0 + 0.5) + 1.0)
        * (0.5 * (p.sigma[ai] + p.sigma[aj])).powi(3)
        * p.kappa_aibj[(ai, aj)]
        * (temperature.recip() * p.epsilon_k_aibj[(ai, aj)]).exp_m1()
}

impl<D: DualNum<f64>> HelmholtzEnergyDual<D> for Association {
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let p = &self.parameters;
        let a = p.assoc_comp[0];

        // temperature dependent segment radius
        let r = p.hs_diameter(state.temperature) * 0.5;

        // auxiliary variables
        let n2 = (&state.partial_density * &p.m * &r * &r).sum() * 4.0 * PI;
        let n3 = (&state.partial_density * &p.m * &r * &r * &r).sum() * 4.0 * FRAC_PI_3;
        let n3i = (-n3 + 1.0).recip();

        // association strength
        let deltarho = association_strength(p, state.temperature, &r, n2, n3i, D::one(), a, a)
            * state.partial_density[a];

        let na = p.na[a];
        let nb = p.nb[a];
        if nb > 0.0 {
            // no cross association, two association sites
            let xa = assoc_site_frac_ab(deltarho, na, nb);
            let xb = (xa - 1.0) * (na / nb) + 1.0;

            state.moles[a] * ((xa.ln() - xa * 0.5 + 0.5) * na + (xb.ln() - xb * 0.5 + 0.5) * nb)
        } else {
            // no cross association, one association site
            let xa = assoc_site_frac_a(deltarho, na);

            state.moles[a] * (xa.ln() - xa * 0.5 + 0.5) * na
        }
    }
}

impl fmt::Display for Association {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Association")
    }
}

pub(crate) fn assoc_site_frac_ab<D: DualNum<f64>>(deltarho: D, na: f64, nb: f64) -> D {
    if deltarho.re() > f64::EPSILON.sqrt() {
        (((deltarho * (na - nb) + 1.0).powi(2) + deltarho * nb * 4.0).sqrt()
            - (deltarho * (nb - na) + 1.0))
            / (deltarho * na * 2.0)
    } else {
        D::one() + deltarho * nb * (deltarho * (nb + na) - 1.0)
    }
}

pub(crate) fn assoc_site_frac_a<D: DualNum<f64>>(deltarho: D, na: f64) -> D {
    if deltarho.re() > f64::EPSILON.sqrt() {
        ((deltarho * na * 4.0 + 1.0).powi(2) - 1.0).sqrt() / (deltarho * na * 2.0)
    } else {
        D::one() + deltarho * na * (deltarho * na * 2.0 - 1.0)
    }
}

impl<D: DualNum<f64> + ScalarOperand> HelmholtzEnergyDual<D> for CrossAssociation {
    fn helmholtz_energy(&self, state: &StateHD<D>) -> D {
        let p = &self.parameters;

        // temperature dependent segment radius
        let r = p.hs_diameter(state.temperature) * 0.5;

        // auxiliary variables
        let n2 = (&state.partial_density * &p.m * &r * &r).sum() * 4.0 * PI;
        let n3 = (&state.partial_density * &p.m * &r * &r * &r).sum() * 4.0 * FRAC_PI_3;
        let n3i = (-n3 + 1.0).recip();

        // Helmholtz energy
        helmholtz_energy_density_cross_association(
            p,
            state.temperature,
            &state.partial_density,
            &r,
            n2,
            n3i,
            D::one(),
            self.max_iter,
            self.tol,
            None,
        )
        .unwrap()
            * state.volume
    }
}

impl fmt::Display for CrossAssociation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cross-association")
    }
}

pub fn helmholtz_energy_density_cross_association<S, D: DualNum<f64> + ScalarOperand>(
    p: &PcSaftParameters,
    temperature: D,
    density: &ArrayBase<S, Ix1>,
    r: &Array1<D>,
    n2: D,
    n3i: D,
    xi: D,
    max_iter: usize,
    tol: f64,
    x0: Option<&mut Array1<f64>>,
) -> Result<D, EosError>
where
    S: Data<Elem = D>,
{
    // check if density is close to 0
    if density.sum().re() < f64::EPSILON {
        if let Some(x0) = x0 {
            x0.fill(1.0);
        }
        return Ok(D::zero());
    }

    // association strength
    let delta = Array::from_shape_fn((p.nassoc, p.nassoc), |(i, j)| {
        association_strength(
            p,
            temperature,
            r,
            n2,
            n3i,
            xi,
            p.assoc_comp[i],
            p.assoc_comp[j],
        )
    });

    // extract parameters of associating components
    let na = Array::from_shape_fn(p.nassoc, |i| p.na[p.assoc_comp[i]]);
    let nb = Array::from_shape_fn(p.nassoc, |i| p.nb[p.assoc_comp[i]]);
    let rho = Array::from_shape_fn(p.nassoc, |i| density[p.assoc_comp[i]]);

    // cross-association according to Michelsen2006
    // initialize monomer fraction
    let mut x = match &x0 {
        Some(x0) => (*x0).clone(),
        None => Array::from_elem(2 * p.nassoc, 0.2),
    };

    for k in 0..max_iter {
        if newton_step_cross_association::<f64>(
            &mut x,
            p,
            &delta.map(D::re),
            &na,
            &nb,
            &rho.map(D::re),
            tol,
        )? {
            break;
        }
        if k == max_iter - 1 {
            return Err(EosError::NotConverged("Cross association".into()));
        }
    }

    // calculate derivatives
    let mut x_dual = x.mapv(D::from);
    for _ in 0..D::NDERIV {
        newton_step_cross_association(&mut x_dual, p, &delta, &na, &nb, &rho, tol)?;
    }

    // save monomer fraction
    if let Some(x0) = x0 {
        *x0 = x;
    }

    // Helmholtz energy density
    let xa = x_dual.slice(s![..p.nassoc]);
    let xb = x_dual.slice(s![p.nassoc..]);
    let f = |x: D| x.ln() - x * 0.5 + 0.5;
    Ok((rho * (xa.mapv(f) * na + xb.mapv(f) * nb)).sum())
}

fn newton_step_cross_association<D: DualNum<f64> + ScalarOperand>(
    x: &mut Array1<D>,
    p: &PcSaftParameters,
    delta: &Array2<D>,
    na: &Array1<f64>,
    nb: &Array1<f64>,
    rho: &Array1<D>,
    tol: f64,
) -> Result<bool, EosError> {
    // gradient
    let mut g: Array1<D> = Array::zeros(2 * p.nassoc);
    // Hessian
    let mut h: Array2<D> = Array::zeros((2 * p.nassoc, 2 * p.nassoc));

    // slice arrays
    let (xa, xb) = x.multi_slice_mut((s![..p.nassoc], s![p.nassoc..]));
    let (mut ga, mut gb) = g.multi_slice_mut((s![..p.nassoc], s![p.nassoc..]));
    let (mut haa, mut hab, mut hba, mut hbb) = h.multi_slice_mut((
        s![..p.nassoc, ..p.nassoc],
        s![..p.nassoc, p.nassoc..],
        s![p.nassoc.., ..p.nassoc],
        s![p.nassoc.., p.nassoc..],
    ));

    // calculate gradients and approximate Hessian
    for i in 0..p.nassoc {
        let d = &delta.index_axis(Axis(0), i) * rho;

        let dnx = (&xb * nb * &d).sum() + 1.0;
        ga[i] = xa[i].recip() - dnx;
        hab.index_axis_mut(Axis(0), i).assign(&(&d * &(-nb)));
        haa[(i, i)] = -dnx / xa[i];

        let dnx = (&xa * na * &d).sum() + 1.0;
        gb[i] = xb[i].recip() - dnx;
        hba.index_axis_mut(Axis(0), i).assign(&(&d * &(-na)));
        hbb[(i, i)] = -dnx / xb[i];
    }

    // Newton step
    x.assign(&(&*x - &LU::new(h)?.solve(&g)));

    // check convergence
    Ok(norm(&g.map(D::re)) < tol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcsaft::parameters::utils::water_parameters;
    use approx::assert_relative_eq;

    #[test]
    fn helmholtz_energy() {
        let assoc = Association {
            parameters: Rc::new(water_parameters()),
        };
        let t = 350.0;
        let v = 41.248289328513216;
        let n = 1.23;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a_rust = assoc.helmholtz_energy(&s) / n;
        assert_relative_eq!(a_rust, -4.229878997054543, epsilon = 1e-10);
    }

    #[test]
    fn helmholtz_energy_cross() {
        let assoc = CrossAssociation {
            parameters: Rc::new(water_parameters()),
            max_iter: 50,
            tol: 1e-10,
        };
        let t = 350.0;
        let v = 41.248289328513216;
        let n = 1.23;
        let s = StateHD::new(t, v, arr1(&[n]));
        let a_rust = assoc.helmholtz_energy(&s) / n;
        assert_relative_eq!(a_rust, -4.229878997054543, epsilon = 1e-10);
    }
}
