use ndarray::*;
use num_dual::DualNum;
// use rustfft::num_complex::Complex;
use std::f64::consts::{FRAC_PI_3, PI};
use std::ops::Mul;

/// A weight function corresponding to a single weighted density.
#[derive(Clone)]
pub struct WeightFunction<T> {
    /// Factor in front of normalized weight function
    pub prefactor: Array1<T>,
    /// Kernel radius of the convolution
    pub kernel_radius: Array1<T>,
    /// Shape of the weight function (Dirac delta, Heaviside, etc.)
    pub shape: WeightFunctionShape,
}

impl<T: DualNum<f64>> WeightFunction<T> {
    /// Create a new weight function without prefactor
    pub fn new_unscaled(kernel_radius: Array1<T>, shape: WeightFunctionShape) -> Self {
        Self {
            prefactor: Array::ones(kernel_radius.raw_dim()),
            kernel_radius,
            shape,
        }
    }

    /// Create a new weight function with weight constant = 1
    pub fn new_scaled(kernel_radius: Array1<T>, shape: WeightFunctionShape) -> Self {
        let unscaled = Self::new_unscaled(kernel_radius, shape);
        let weight_constants = unscaled.scalar_weight_constants(T::zero());
        Self {
            prefactor: weight_constants.mapv(|w| w.recip()),
            kernel_radius: unscaled.kernel_radius,
            shape,
        }
    }

    /// Calculates the value of the scalar weight function depending on its shape
    /// `k_abs` describes the absolute value of the Fourier variable
    pub(crate) fn fft_scalar_weight_functions<D: Dimension, T2: DualNum<f64>>(
        &self,
        k_abs: &Array<T2, D>,
        lanczos: &Option<Array<T2, D>>,
    ) -> Array<T, D::Larger>
    where
        T: Mul<T2, Output = T>,
        D::Larger: Dimension<Smaller = D>,
    {
        // Allocate vector for weight functions
        let mut d = vec![self.prefactor.len()];
        k_abs.shape().iter().for_each(|&x| d.push(x));
        let mut w = Array::zeros(d.into_dimension())
            .into_dimensionality()
            .unwrap();

        // Calculate weight function for each component
        for i in 0..self.prefactor.len() {
            let radius = self.kernel_radius[i];
            let p = self.prefactor[i];
            let rik = k_abs.mapv(|k| radius * k);
            let mut w_i = w.index_axis_mut(Axis(0), i);
            w_i.assign(&match self.shape {
                WeightFunctionShape::Theta => rik.mapv(|rik| {
                    (rik.sph_j0() + rik.sph_j2()) * 4.0 * FRAC_PI_3 * radius.powi(3) * p
                }),
                WeightFunctionShape::Delta => {
                    rik.mapv(|rik| rik.sph_j0() * 4.0 * PI * radius.powi(2) * p)
                }
                WeightFunctionShape::KR1 => {
                    rik.mapv(|rik| (rik.sph_j0() + rik.cos()) * 0.5 * radius * p)
                }
                WeightFunctionShape::KR0 => rik.mapv(|rik| (rik * rik.sin() * 0.5 + rik.cos()) * p),
                WeightFunctionShape::DeltaVec => unreachable!(),
            });

            // Apply Lanczos sigma factor
            if let Some(l) = lanczos {
                w_i.assign(&(&w_i * l));
            }
        }

        // Return real part of weight function
        w
    }

    /// Calculates the value of the vector weight function depending on its shape
    /// `k_abs` describes the absolute value of the Fourier variable
    /// `k` describes the (potentially multi-dimensional) Fourier variable
    pub(crate) fn fft_vector_weight_functions<D: Dimension, T2: DualNum<f64>>(
        &self,
        k_abs: &Array<T2, D>,
        k: &Array<T2, D::Larger>,
        lanczos: &Option<Array<T2, D>>,
    ) -> Array<T, <D::Larger as Dimension>::Larger>
    where
        D::Larger: Dimension<Smaller = D>,
        <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
        T: Mul<T2, Output = T>,
    {
        // Allocate vector for weight functions
        let mut d = vec![k.shape()[0], self.prefactor.len()];
        k.shape().iter().skip(1).for_each(|&x| d.push(x));
        let mut w = Array::zeros(d.into_dimension())
            .into_dimensionality()
            .unwrap();

        // Iterate all dimensions
        for (k_x, mut w_x) in k.outer_iter().zip(w.outer_iter_mut()) {
            // Calculate weight function for each component
            for i in 0..self.prefactor.len() {
                let radius = self.kernel_radius[i];
                let p = self.prefactor[i];
                let rik = k_abs.mapv(|k| radius * k);
                let mut w_i = w_x.index_axis_mut(Axis(0), i);
                w_i.assign(&match self.shape {
                    WeightFunctionShape::DeltaVec => {
                        &rik.mapv(|rik| {
                            (rik.sph_j0() + rik.sph_j2()) * (-radius.powi(3) * 4.0 * FRAC_PI_3 * p)
                        }) * &k_x
                    }
                    _ => unreachable!(),
                });

                // Apply Lanczos sigma factor
                if let Some(l) = lanczos {
                    w_i.assign(&(&w_i * l));
                }
            }
        }
        // Return imaginary part of weight function
        w
    }

    /// Scalar weights for the bulk convolver (for the bulk convolver only the prefactor
    /// of the normalized weight functions are required)
    pub fn scalar_weight_constants(&self, k: T) -> Array1<T> {
        let k = arr0(k);
        self.fft_scalar_weight_functions(&k, &None)
    }

    /// Vector weights for the bulk convolver (for the bulk convolver only the prefactor
    /// of the normalized weight functions are required)
    pub fn vector_weight_constants(&self, k: T) -> Array1<T> {
        let k_abs = arr0(k);
        let k = arr1(&[k]);
        self.fft_vector_weight_functions(&k_abs, &k, &None)
            .index_axis_move(Axis(0), 0)
    }
}

/// Possible weight function shapes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum WeightFunctionShape {
    /// Heaviside step function
    Theta,
    /// Dirac delta function
    Delta,
    /// Combination of first & second derivative of Dirac delta function (with different prefactor; only in Kierlik-Rosinberg functional)
    KR0,
    /// First derivative of Dirac delta function (with different prefactor; only in Kierlik-Rosinberg functional)
    KR1,
    /// Vector-shape as combination of Dirac delta and outward normal
    DeltaVec,
}

/// Defining `type` for information about weight functions based on
/// `WeightFunctionBase<TScal, TVec>`.
// pub type WeightFunctionInfo<T> = WeightFunctionBase<WeightFunction<T>, WeightFunction<T>>;

pub struct WeightFunctionInfo<T> {
    /// Index of the component that each individual segment belongs to.
    pub(crate) component_index: Array1<usize>,
    /// Flag if local density is required in the functional
    pub(crate) local_density: bool,
    /// Container for scalar component-wise weighted densities
    pub(crate) scalar_component_weighted_densities: Vec<WeightFunction<T>>,
    /// Container for vector component-wise weighted densities
    pub(crate) vector_component_weighted_densities: Vec<WeightFunction<T>>,
    /// Container for scalar FMT weighted densities
    pub(crate) scalar_fmt_weighted_densities: Vec<WeightFunction<T>>,
    /// Container for vector FMT weighted densities
    pub(crate) vector_fmt_weighted_densities: Vec<WeightFunction<T>>,
}

impl<T> WeightFunctionInfo<T> {
    /// Calculates the total number of weighted densities for each functional
    /// from multiple weight functions depending on dimension.
    pub fn n_weighted_densities(&self, dimensions: usize) -> usize {
        let segments = self.component_index.len();
        (if self.local_density { segments } else { 0 })
            + self.scalar_component_weighted_densities.len() * segments
            + self.vector_component_weighted_densities.len() * segments * dimensions
            + self.scalar_fmt_weighted_densities.len()
            + self.vector_fmt_weighted_densities.len() * dimensions
    }
}

impl<T> WeightFunctionInfo<T> {
    /// Initializing empty `WeightFunctionInfo`.
    pub fn new(component_index: Array1<usize>, local_density: bool) -> Self {
        Self {
            component_index,
            local_density,
            scalar_component_weighted_densities: Vec::new(),
            vector_component_weighted_densities: Vec::new(),
            scalar_fmt_weighted_densities: Vec::new(),
            vector_fmt_weighted_densities: Vec::new(),
        }
    }

    /// Adds and sorts [WeightFunction] depending on information
    /// about {FMT, component} & {scalar-valued, vector-valued}.
    pub fn add(mut self, weight_function: WeightFunction<T>, fmt: bool) -> Self {
        let segments = self.component_index.len();

        // Check size of `kernel_radius`
        if segments != weight_function.kernel_radius.len() {
            panic!(
                "Number of segments is fixed to {}; `kernel_radius` has {} entries.",
                segments,
                weight_function.kernel_radius.len()
            );
        }

        // Check size of `prefactor`
        if segments != weight_function.prefactor.len() {
            panic!(
                "Number of segments is fixed to {}; `prefactor` has {} entries.",
                segments,
                weight_function.prefactor.len()
            );
        }

        // Add new `WeightFunction`
        match (fmt, weight_function.shape) {
            // {Component, vector}
            (false, WeightFunctionShape::DeltaVec) => self
                .vector_component_weighted_densities
                .push(weight_function),
            // {Component, scalar}
            (false, _) => self
                .scalar_component_weighted_densities
                .push(weight_function),
            // {FMT, vector}
            (true, WeightFunctionShape::DeltaVec) => {
                self.vector_fmt_weighted_densities.push(weight_function)
            }
            // {FMT, scalar}
            (true, _) => self.scalar_fmt_weighted_densities.push(weight_function),
        };

        // Return
        self
    }

    /// Adds and sorts multiple [WeightFunction]s.
    pub fn extend(mut self, weight_functions: Vec<WeightFunction<T>>, fmt: bool) -> Self {
        // Add each element of vector
        for wf in weight_functions {
            self = self.add(wf, fmt);
        }

        // Return
        self
    }

    /// Expose weight functions outside of this crate
    pub fn as_slice(&self) -> [&Vec<WeightFunction<T>>; 4] {
        [
            &self.scalar_component_weighted_densities,
            &self.vector_component_weighted_densities,
            &self.scalar_fmt_weighted_densities,
            &self.vector_fmt_weighted_densities,
        ]
    }
}

impl<T: DualNum<f64>> WeightFunctionInfo<T> {
    /// calculates the matrix of weight constants for this set of weighted densities
    pub fn weight_constants(&self, k: T, dimensions: usize) -> Array2<T> {
        let segments = self.component_index.len();
        let n_wd = self.n_weighted_densities(dimensions);
        let mut weight_constants = Array::zeros([n_wd, segments]);
        let mut j = 0;
        if self.local_density {
            weight_constants
                .slice_mut(s![j..j + segments, ..])
                .into_diag()
                .assign(&Array::ones(segments));
            j += segments;
        }
        for w in &self.scalar_component_weighted_densities {
            weight_constants
                .slice_mut(s![j..j + segments, ..])
                .into_diag()
                .assign(&w.scalar_weight_constants(k));
            j += segments;
        }
        if dimensions == 1 {
            for w in &self.vector_component_weighted_densities {
                weight_constants
                    .slice_mut(s![j..j + segments, ..])
                    .into_diag()
                    .assign(&w.vector_weight_constants(k));
                j += segments;
            }
        }
        for w in &self.scalar_fmt_weighted_densities {
            weight_constants
                .slice_mut(s![j, ..])
                .assign(&w.scalar_weight_constants(k));
            j += 1;
        }
        if dimensions == 1 {
            for w in &self.vector_fmt_weighted_densities {
                weight_constants
                    .slice_mut(s![j, ..])
                    .assign(&w.vector_weight_constants(k));
                j += 1;
            }
        }
        weight_constants
    }
}
