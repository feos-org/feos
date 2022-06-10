use feos_core::{EosResult, EosUnit};
use ndarray::Array1;
use quantity::{QuantityArray1, QuantityArray2, QuantityScalar};
use std::f64::consts::{FRAC_PI_3, PI};

/// Grids with up to three dimensions.
#[derive(Clone)]
pub enum Grid {
    Cartesian1(Axis),
    Cartesian2(Axis, Axis),
    Periodical2(Axis, Axis),
    Cartesian3(Axis, Axis, Axis),
    Periodical3(Axis, Axis, Axis),
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
            Self::Cartesian1(x) => vec![x],
            Self::Cartesian2(x, y) | Self::Periodical2(x, y) => vec![x, y],
            Self::Cartesian3(x, y, z) | Self::Periodical3(x, y, z) => vec![x, y, z],
            Self::Spherical(r) | Self::Polar(r) => vec![r],
            Self::Cylindrical { r, z } => vec![r, z],
        }
    }

    pub fn axes_mut(&mut self) -> Vec<&mut Axis> {
        match self {
            Self::Cartesian1(x) => vec![x],
            Self::Cartesian2(x, y) | Self::Periodical2(x, y) => vec![x, y],
            Self::Cartesian3(x, y, z) | Self::Periodical3(x, y, z) => vec![x, y, z],
            Self::Spherical(r) | Self::Polar(r) => vec![r],
            Self::Cylindrical { r, z } => vec![r, z],
        }
    }

    pub fn grids(&self) -> Vec<&Array1<f64>> {
        self.axes().iter().map(|ax| &ax.grid).collect()
    }

    pub(crate) fn integration_weights(&self) -> Vec<&Array1<f64>> {
        self.axes()
            .iter()
            .map(|ax| &ax.integration_weights)
            .collect()
    }

    pub(crate) fn integration_weights_unit<U: EosUnit>(&self) -> Vec<QuantityArray1<U>> {
        self.axes()
            .iter()
            .map(|ax| &ax.integration_weights * U::reference_length().powi(ax.geometry.dimension()))
            .collect()
    }
}

/// Geometries of individual axes.
#[derive(Copy, Clone)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
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
    pub fn new_cartesian<U: EosUnit>(
        points: usize,
        length: QuantityScalar<U>,
        potential_offset: Option<f64>,
    ) -> EosResult<Self> {
        let potential_offset = potential_offset.unwrap_or(0.0);
        let l = length.to_reduced(U::reference_length())? + potential_offset;
        let cell_size = l / points as f64;
        let grid = Array1::linspace(0.5 * cell_size, l - 0.5 * cell_size, points);
        let edges = Array1::linspace(0.0, l, points + 1);
        let integration_weights = Array1::from_elem(points, cell_size);
        Ok(Self {
            geometry: Geometry::Cartesian,
            grid,
            edges,
            integration_weights,
            potential_offset,
        })
    }

    /// Create a new (equidistant) spherical axis.
    pub fn new_spherical<U: EosUnit>(points: usize, length: QuantityScalar<U>) -> EosResult<Self> {
        let l = length.to_reduced(U::reference_length())?;
        let cell_size = l / points as f64;
        let grid = Array1::linspace(0.5 * cell_size, l - 0.5 * cell_size, points);
        let edges = Array1::linspace(0.0, l, points + 1);
        let integration_weights = Array1::from_shape_fn(points, |k| {
            4.0 * FRAC_PI_3 * cell_size.powi(3) * (3 * k * k + 3 * k + 1) as f64
        });
        Ok(Self {
            geometry: Geometry::Spherical,
            grid,
            edges,
            integration_weights,
            potential_offset: 0.0,
        })
    }

    /// Create a new logarithmically scaled cylindrical axis.
    pub fn new_polar<U: EosUnit>(points: usize, length: QuantityScalar<U>) -> EosResult<Self> {
        let l = length.to_reduced(U::reference_length())?;

        let mut alpha = 0.002_f64;
        for _ in 0..20 {
            alpha = -(1.0 - (-alpha).exp()).ln() / (points - 1) as f64;
        }
        let x0 = 0.5 * ((-alpha * points as f64).exp() + (-alpha * (points - 1) as f64).exp());
        let grid = (0..points)
            .into_iter()
            .map(|i| l * x0 * (alpha * i as f64).exp())
            .collect();
        let edges = (0..=points)
            .into_iter()
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
            .into_iter()
            .map(|i| {
                (match i {
                    0 => k0 * (2.0 * alpha).exp(),
                    1 => ((2.0 * alpha).exp() - k0) * (2.0 * alpha).exp(),
                    _ => (2.0 * alpha * i as f64).exp() * ((2.0 * alpha).exp() - 1.0),
                }) * ((-2.0 * alpha * points as f64).exp() * PI * l * l)
            })
            .collect();

        Ok(Self {
            geometry: Geometry::Cylindrical,
            grid,
            edges,
            integration_weights,
            potential_offset: 0.0,
        })
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
    pub fn volume<U: EosUnit>(&self) -> QuantityScalar<U> {
        let length = (self.edges[self.grid.len()] - self.potential_offset - self.edges[0])
            * U::reference_length();
        (match self.geometry {
            Geometry::Cartesian => 1.0,
            Geometry::Cylindrical => 4.0 * PI,
            Geometry::Spherical => 4.0 * FRAC_PI_3,
        }) * length.powi(self.geometry.dimension())
    }

    /// Interpolate a function on the given axis.
    pub fn interpolate<U: EosUnit>(
        &self,
        x: f64,
        y: &QuantityArray2<U>,
        i: usize,
    ) -> QuantityScalar<U> {
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
