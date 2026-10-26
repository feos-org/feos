//! Tests of the generic association functional with a toy model: one component with an A site, a
//! B site, and two C sites (C2 with multiplicity 2), weighted densities and contact values as in
//! a WDA functional, and either a local or a contact-shell association kernel.
use feos_core::ReferenceSystem;
use feos_core::parameter::{AssociationParameters, AssociationSite, BinaryParameters};
use feos_dft::{
    AssociationFunctional, AssociationSites, Axis, ConvolverFFT, Grid, WeightFunction,
    WeightFunctionInfo, WeightFunctionShape,
};
use nalgebra::dvector;
use ndarray::{Array1, Array2, ArrayView2, Ix1, arr1};
use num_dual::{Dual2_64, Dual64, DualNum};
use quantity::Length;
use std::f64::consts::PI;

/// Hard-sphere diameter.
const D: f64 = 2.9;
/// Association strengths for the records AB, CC(0,0), CC(0,1), CC(1,0), CC(1,1).
const C: [f64; 5] = [3.6, 0.5, 0.3, 0.3, 0.8];
/// Contact distances of the shell kernel for every record.
const CONTACT_DISTANCES: [f64; 5] = [2.9, 3.0, 3.1, 3.1, 3.2];
/// Multiplicities of the sites A, B, C1, C2.
const N_SITE: [f64; 4] = [1.0, 1.0, 1.0, 2.0];
/// Temperature (not used by the toy model).
const T: f64 = 300.0;
/// Bulk density.
const RHO: f64 = 0.0333;

struct Toy {
    parameters: AssociationParameters<()>,
    shell: bool,
}

impl Toy {
    fn new(shell: bool) -> Self {
        let site = |n| AssociationSite {
            assoc_comp: 0,
            id: "".into(),
            n,
        };
        let parameters = AssociationParameters {
            component_index: dvector![0],
            sites_a: vec![site(N_SITE[0])],
            sites_b: vec![site(N_SITE[1])],
            sites_c: vec![site(N_SITE[2]), site(N_SITE[3])],
            binary_ab: vec![BinaryParameters::new(0, 0, (), ())],
            binary_cc: vec![
                BinaryParameters::new(0, 0, (), ()),
                BinaryParameters::new(0, 1, (), ()),
                BinaryParameters::new(1, 0, (), ()),
                BinaryParameters::new(1, 1, (), ()),
            ],
        };
        Self { parameters, shell }
    }
}

/// BMCSL contact value of a pure hard-sphere fluid.
fn contact_value<N: DualNum<Primitive = f64> + Copy>(rho: N) -> N {
    let z2 = rho * (PI / 6.0 * D * D);
    let z3 = rho * (PI / 6.0 * D * D * D);
    let z3i = (-z3 + 1.0).recip();
    z3i + z2 * z3i * z3i * (1.5 * D) + (z2 * D).powi(2) * z3i.powi(3) * 0.5
}

impl AssociationFunctional for Toy {
    fn sites(&self) -> AssociationSites<'_> {
        AssociationSites::new(&self.parameters)
    }

    fn weight_functions<N: DualNum<Primitive = f64> + Copy>(&self, _: N) -> WeightFunctionInfo<N> {
        WeightFunctionInfo::new(1, false).add(
            WeightFunction::new_scaled(dvector![N::from(D)], WeightFunctionShape::Theta),
            false,
        )
    }

    fn contact_distances<N: DualNum<Primitive = f64> + Copy>(&self, _: N) -> Option<Array1<N>> {
        self.shell
            .then(|| Array1::from_iter(CONTACT_DISTANCES.map(N::from)))
    }

    fn contact_values<N: DualNum<Primitive = f64> + Copy>(
        &self,
        _: N,
        weighted_densities: ArrayView2<N>,
    ) -> (Array2<N>, Array2<N>) {
        let rho = weighted_densities.row(0);
        let y = rho.mapv(contact_value);
        let y = Array2::from_shape_fn((C.len(), rho.len()), |(_, j)| y[j]);
        (rho.to_owned().insert_axis(ndarray::Axis(0)), y)
    }

    fn association_strength<N: DualNum<Primitive = f64> + Copy>(&self, _: N) -> Array1<N> {
        Array1::from_iter(C.map(N::from))
    }
}

/// Independent bulk solution: fractions of non-bonded sites and Helmholtz energy density.
fn bulk_reference(rho: f64) -> ([f64; 4], f64) {
    let y = contact_value(rho);
    let r = N_SITE.map(|n| n * rho);
    let dab = C[0] * y;
    let dcc = [[C[1] * y, C[2] * y], [C[3] * y, C[4] * y]];
    let mut x = [0.5; 4];
    for _ in 0..10000 {
        let new = [
            1.0 / (1.0 + dab * r[1] * x[1]),
            1.0 / (1.0 + dab * r[0] * x[0]),
            1.0 / (1.0 + dcc[0][0] * r[2] * x[2] + dcc[0][1] * r[3] * x[3]),
            1.0 / (1.0 + dcc[1][0] * r[2] * x[2] + dcc[1][1] * r[3] * x[3]),
        ];
        for i in 0..4 {
            x[i] = 0.5 * (x[i] + new[i]);
        }
    }
    let a = (0..4).map(|i| r[i] * (x[i].ln() - 0.5 * x[i] + 0.5)).sum();
    (x, a)
}

/// Derivative of the reference bulk Helmholtz energy density with respect to the density.
fn bulk_reference_derivative(rho: f64) -> f64 {
    let h = 1e-6 * rho;
    (bulk_reference(rho + h).1 - bulk_reference(rho - h).1) / (2.0 * h)
}

fn grid(n: usize) -> (Grid, Array1<f64>, f64) {
    let axis = Axis::new_cartesian(n, Length::from_reduced(20.0), None);
    let z = axis.grid.clone();
    (Grid::Cartesian1(axis), z, 20.0 / n as f64)
}

fn rel(a: f64, b: f64) -> f64 {
    ((a - b) / b).abs()
}

#[test]
fn bulk_evaluation() {
    let (x_ref, a_ref) = bulk_reference(RHO);
    let mu_ref = bulk_reference_derivative(RHO);
    for shell in [false, true] {
        let f = Toy::new(shell);
        let n = 64;
        let (grid, _, _) = grid(n);
        let convolver = ConvolverFFT::<f64, Ix1>::plan(&grid, &[f.weight_functions(T)]);
        let kernel_convolver = f
            .kernel_weight_functions(T)
            .map(|k| ConvolverFFT::<f64, Ix1>::plan(&grid, &[k]));
        let density = Array2::from_elem((1, n), RHO);
        let x = Array2::from_shape_fn((4, n), |(i, _)| x_ref[i]);
        let e = f.evaluate(
            T,
            &density,
            &x,
            convolver.as_ref(),
            kernel_convolver.as_deref(),
        );
        for j in 0..n {
            assert!(rel(e.helmholtz_energy_density[j], a_ref) < 1e-12);
            assert!(rel(e.functional_derivative[[0, j]], mu_ref) < 1e-7);
        }
        assert!(e.mass_action_residual.iter().all(|r| r.abs() < 1e-12));
    }
}

#[test]
fn bulk_methods() {
    let f = Toy::new(true);
    let (_, a_ref) = bulk_reference(RHO);

    // Helmholtz energy density at the equilibrium fractions of non-bonded sites
    assert!(rel(f.bulk_helmholtz_energy_density(T, &arr1(&[RHO])), a_ref) < 1e-12);

    // first derivative with respect to the density
    let a =
        f.bulk_helmholtz_energy_density(Dual64::from(T), &arr1(&[Dual64::from(RHO).derivative()]));
    assert!(rel(a.eps, bulk_reference_derivative(RHO)) < 1e-7);

    // second derivative with respect to the density (requires the derivative of the fractions
    // of non-bonded sites with respect to the density)
    let a = f.bulk_helmholtz_energy_density(
        Dual2_64::from(T),
        &arr1(&[Dual2_64::from(RHO).derivative()]),
    );
    let h = 1e-4 * RHO;
    let d2a = (bulk_reference_derivative(RHO + h) - bulk_reference_derivative(RHO - h)) / (2.0 * h);
    assert!(rel(a.v2, d2a) < 1e-5);
}

#[test]
fn functional_derivative() {
    for shell in [false, true] {
        let f = Toy::new(shell);
        let n = 256;
        let (grid, z, dz) = grid(n);
        let convolver = ConvolverFFT::<f64, Ix1>::plan(&grid, &[f.weight_functions(T)]);
        let kernel_convolver = f
            .kernel_weight_functions(T)
            .map(|k| ConvolverFFT::<f64, Ix1>::plan(&grid, &[k]));

        // inhomogeneous density and fractions of non-bonded sites (held fixed)
        let density = Array2::from_shape_fn((1, n), |(_, j)| {
            RHO * (1.0 + 0.4 * (z[j] / 2.0).cos() * (-z[j] / 8.0).exp())
        });
        let x = Array2::from_shape_fn((4, n), |(i, j)| {
            0.05 + 0.1 * i as f64 + 0.03 * (z[j] / 3.0).sin()
        });
        let evaluate = |density: &Array2<f64>| {
            f.evaluate(
                T,
                density,
                &x,
                convolver.as_ref(),
                kernel_convolver.as_deref(),
            )
        };
        let functional_derivative = evaluate(&density).functional_derivative;
        let free_energy =
            |density: &Array2<f64>| evaluate(density).helmholtz_energy_density.sum() * dz;

        // central finite differences
        for j in [0, 10, 40, 100, 200, 255] {
            let h = 1e-4 * RHO;
            let (mut rho_p, mut rho_m) = (density.clone(), density.clone());
            rho_p[[0, j]] += h;
            rho_m[[0, j]] -= h;
            let fd = (free_energy(&rho_p) - free_energy(&rho_m)) / (2.0 * h * dz);
            assert!(rel(functional_derivative[[0, j]], fd) < 1e-7);
        }
    }
}
