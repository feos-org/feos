use feos_core::ReferenceSystem;
use ndarray::{
    Array, Array1, Array2, Array3, ArrayBase, ArrayD, Axis as Axis_nd, Data, Dimension, RemoveAxis,
};
use num_dual::DualNum;
use quantity::{Angle, DEGREES, Length, Quantity};
use std::f64::consts::{FRAC_PI_3, PI};
use std::ops::MulAssign;

/// Grids with up to three dimensions.
#[derive(Clone)]
pub enum Grid {
    Bulk,
    Cartesian1(Axis),
    Periodical1(Axis),
    Cartesian2(Axis, Axis),
    Periodical2(Axis, Axis, Angle),
    Cartesian3(Axis, Axis, Axis),
    Periodical3(Axis, Axis, Axis, [Angle; 3]),
    Spherical(Axis),
    Polar(Axis),
    Cylindrical { r: Axis, z: Axis },
}

impl Grid {
    pub fn new_1d(axis: Axis) -> Self {
        match axis.geometry {
            Geometry::Cartesian => Self::Cartesian1(axis),
            Geometry::Cylindrical => Self::Polar(axis),
            Geometry::Spherical => Self::Spherical(axis),
        }
    }

    pub fn axes(&self) -> Vec<&Axis> {
        match self {
            Self::Bulk => vec![],
            Self::Cartesian1(x) | Self::Periodical1(x) => vec![x],
            Self::Cartesian2(x, y) | Self::Periodical2(x, y, _) => vec![x, y],
            Self::Cartesian3(x, y, z) | Self::Periodical3(x, y, z, _) => vec![x, y, z],
            Self::Spherical(r) | Self::Polar(r) => vec![r],
            Self::Cylindrical { r, z } => vec![r, z],
        }
    }

    pub fn axes_mut(&mut self) -> Vec<&mut Axis> {
        match self {
            Self::Bulk => vec![],
            Self::Cartesian1(x) | Self::Periodical1(x) => vec![x],
            Self::Cartesian2(x, y) | Self::Periodical2(x, y, _) => vec![x, y],
            Self::Cartesian3(x, y, z) | Self::Periodical3(x, y, z, _) => vec![x, y, z],
            Self::Spherical(r) | Self::Polar(r) => vec![r],
            Self::Cylindrical { r, z } => vec![r, z],
        }
    }

    pub fn grids(&self) -> Vec<&Array1<f64>> {
        self.axes().iter().map(|ax| &ax.grid).collect()
    }

    pub(crate) fn integration_weights(&self) -> (Vec<&Array1<f64>>, f64) {
        (
            self.axes()
                .iter()
                .map(|ax| &ax.integration_weights)
                .collect(),
            self.functional_determinant(),
        )
    }

    pub(crate) fn functional_determinant(&self) -> f64 {
        match &self {
            Self::Periodical2(_, _, alpha) => alpha.sin(),
            Self::Periodical3(_, _, _, [alpha, beta, gamma]) => {
                let xi = (alpha.cos() - gamma.cos() * beta.cos()) / gamma.sin();
                gamma.sin() * (1.0 - beta.cos().powi(2) - xi * xi).sqrt()
            }
            _ => 1.0,
        }
    }

    pub fn mesh(&self) -> Vec<Length<ArrayD<f64>>> {
        match self {
            Self::Bulk => vec![],
            Grid::Cartesian1(ax)
            | Self::Periodical1(ax)
            | Grid::Spherical(ax)
            | Grid::Polar(ax) => {
                vec![Length::from_reduced(ax.grid.clone()).into_dyn()]
            }
            Grid::Cartesian2(u, v) => mesh_2d(u, v, 90.0 * DEGREES),
            Grid::Periodical2(u, v, alpha) => mesh_2d(u, v, *alpha),
            Grid::Cylindrical { r, z } => mesh_2d(r, z, 90.0 * DEGREES),
            Grid::Cartesian3(u, v, w) => mesh_3d(u, v, w, [90.0 * DEGREES; 3]),
            Grid::Periodical3(u, v, w, angles) => mesh_3d(u, v, w, *angles),
        }
    }

    pub fn integrate_reduced<D: Dimension, N: DualNum<Primitive = f64> + Copy>(
        &self,
        mut profile: Array<N, D>,
    ) -> N {
        let (integration_weights, functional_determinant) = self.integration_weights();

        for (i, w) in integration_weights.into_iter().enumerate() {
            for mut l in profile.lanes_mut(Axis_nd(i)) {
                l.mul_assign(&w.mapv(N::from));
            }
        }
        profile.sum() * functional_determinant
    }

    pub fn integrate_reduced_comp<
        D: Dimension + RemoveAxis,
        S: Data<Elem = N>,
        N: DualNum<Primitive = f64> + Copy,
    >(
        &self,
        profile: &ArrayBase<S, D>,
    ) -> Array1<N> {
        Array1::from_shape_fn(profile.shape()[0], |i| {
            self.integrate_reduced(profile.index_axis(Axis_nd(0), i).to_owned())
        })
    }
}

fn mesh_2d(u: &Axis, v: &Axis, alpha: Angle) -> Vec<Length<ArrayD<f64>>> {
    let u_grid = Array2::from_shape_fn([u.grid.len(), v.grid.len()], |(i, _)| u.grid[i]);
    let v_grid = Array2::from_shape_fn([u.grid.len(), v.grid.len()], |(_, j)| v.grid[j]);
    let x = Length::from_reduced(u_grid + &v_grid * alpha.cos());
    let y = Length::from_reduced(v_grid * alpha.sin());
    vec![x.into_dyn(), y.into_dyn()]
}

fn mesh_3d(
    u: &Axis,
    v: &Axis,
    w: &Axis,
    [alpha, beta, gamma]: [Angle; 3],
) -> Vec<Length<ArrayD<f64>>> {
    let shape = [u.grid.len(), v.grid.len(), w.grid.len()];
    let u_grid = Array3::from_shape_fn(shape, |(i, _, _)| u.grid[i]);
    let v_grid = Array3::from_shape_fn(shape, |(_, j, _)| v.grid[j]);
    let w_grid = Array3::from_shape_fn(shape, |(_, _, k)| w.grid[k]);
    let xi = (alpha.cos() - gamma.cos() * beta.cos()) / gamma.sin();
    let zeta = (1.0_f64 - beta.cos().powi(2) - xi * xi).sqrt();
    let x = Length::from_reduced(u_grid + &v_grid * gamma.cos() + &w_grid * beta.cos());
    let y = Length::from_reduced(v_grid * gamma.sin() + &w_grid * xi);
    let z = Length::from_reduced(w_grid * zeta);
    vec![x.into_dyn(), y.into_dyn(), z.into_dyn()]
}

/// Geometries of individual axes.
#[derive(Copy, Clone, PartialEq)]
pub enum Geometry {
    Cartesian,
    Cylindrical,
    Spherical,
}

impl Geometry {
    /// Return the number of spatial dimensions for this geometry.
    pub fn dimension(&self) -> i32 {
        match self {
            Self::Cartesian => 1,
            Self::Cylindrical => 2,
            Self::Spherical => 3,
        }
    }
}

/// An individual discretized axis.
#[derive(Clone)]
pub struct Axis {
    pub geometry: Geometry,
    pub grid: Array1<f64>,
    pub edges: Array1<f64>,
    integration_weights: Array1<f64>,
    potential_offset: f64,
}

impl Axis {
    /// Create a new (equidistant) cartesian axis.
    ///
    /// The potential_offset is required to make sure that particles
    /// can not interact through walls.
    pub fn new_cartesian(points: usize, length: Length, potential_offset: Option<f64>) -> Self {
        let potential_offset = potential_offset.unwrap_or(0.0);
        let l = length.to_reduced() + potential_offset;
        let cell_size = l / points as f64;
        let grid = Array1::linspace(0.5 * cell_size, l - 0.5 * cell_size, points);
        let edges = Array1::linspace(0.0, l, points + 1);
        let integration_weights = Array1::from_elem(points, cell_size);
        Self {
            geometry: Geometry::Cartesian,
            grid,
            edges,
            integration_weights,
            potential_offset,
        }
    }

    /// Create a new (equidistant) spherical axis.
    pub fn new_spherical(points: usize, length: Length) -> Self {
        let l = length.to_reduced();
        let cell_size = l / points as f64;
        let grid = Array1::linspace(0.5 * cell_size, l - 0.5 * cell_size, points);
        let edges = Array1::linspace(0.0, l, points + 1);
        let integration_weights = Array1::from_shape_fn(points, |k| {
            4.0 * FRAC_PI_3 * cell_size.powi(3) * (3 * k * k + 3 * k + 1) as f64
        });
        Self {
            geometry: Geometry::Spherical,
            grid,
            edges,
            integration_weights,
            potential_offset: 0.0,
        }
    }

    /// Create a new logarithmically scaled cylindrical axis.
    pub fn new_polar(points: usize, length: Length) -> Self {
        let l = length.to_reduced();

        let mut alpha = 0.002_f64;
        for _ in 0..20 {
            alpha = -(1.0 - (-alpha).exp()).ln() / (points - 1) as f64;
        }
        let x0 = 0.5 * ((-alpha * points as f64).exp() + (-alpha * (points - 1) as f64).exp());
        let grid = (0..points)
            .map(|i| l * x0 * (alpha * i as f64).exp())
            .collect();
        let edges = (0..=points)
            .map(|i| {
                if i == 0 {
                    0.0
                } else {
                    l * (-alpha * (points - i) as f64).exp()
                }
            })
            .collect();

        let k0 = (2.0 * alpha).exp() * (2.0 * alpha.exp() + (2.0 * alpha).exp() - 1.0)
            / ((1.0 + alpha.exp()).powi(2) * ((2.0 * alpha).exp() - 1.0));
        let integration_weights = (0..points)
            .map(|i| {
                (match i {
                    0 => k0 * (2.0 * alpha).exp(),
                    1 => ((2.0 * alpha).exp() - k0) * (2.0 * alpha).exp(),
                    _ => (2.0 * alpha * i as f64).exp() * ((2.0 * alpha).exp() - 1.0),
                }) * ((-2.0 * alpha * points as f64).exp() * PI * l * l)
            })
            .collect();

        Self {
            geometry: Geometry::Cylindrical,
            grid,
            edges,
            integration_weights,
            potential_offset: 0.0,
        }
    }

    /// Returns the total length of the axis.
    ///
    /// This includes the `potential_offset` and used e.g.
    /// to determine the correct frequency vector in FFT.
    pub fn length(&self) -> f64 {
        self.edges[self.grid.len()] - self.edges[0]
    }

    /// Returns the volume of the axis.
    ///
    /// Depending on the geometry, the result is in m, m² or m³.
    /// The `potential_offset` is not included in the volume, as
    /// it is mainly used to calculate excess properties.
    pub fn volume(&self) -> f64 {
        let length = self.edges[self.grid.len()] - self.potential_offset - self.edges[0];
        (match self.geometry {
            Geometry::Cartesian => 1.0,
            Geometry::Cylindrical => 4.0 * PI,
            Geometry::Spherical => 4.0 * FRAC_PI_3,
        }) * length.powi(self.geometry.dimension())
    }

    /// Interpolate a function on the given axis.
    pub fn interpolate<U>(
        &self,
        x: f64,
        y: &Quantity<Array2<f64>, U>,
        i: usize,
    ) -> Quantity<f64, U> {
        let n = self.grid.len();
        y.get((
            i,
            if x >= self.edges[n] {
                n - 1
            } else {
                match self.geometry {
                    Geometry::Cartesian | Geometry::Spherical => (x / self.edges[1]) as usize,
                    Geometry::Cylindrical => {
                        if x < self.edges[1] {
                            0
                        } else {
                            (n as f64
                                - (n - 1) as f64 * (x / self.edges[n]).ln()
                                    / (self.edges[1] / self.edges[n]).ln())
                                as usize
                        }
                    }
                }
            },
        ))
    }
}
