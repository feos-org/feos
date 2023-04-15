use super::{Convolver, FFTWeightFunctions};
use crate::geometry::Axis;
use crate::weight_functions::{WeightFunction, WeightFunctionInfo};
use ang::Angle;
use ndarray::Axis as Axis_nd;
use ndarray::*;
use num_dual::DualNum;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftDirection, FftNum, FftPlanner};
use std::f64::consts::PI;
use std::ops::AddAssign;
use std::sync::Arc;

#[derive(Clone)]
pub struct PeriodicConvolver<T, D: Dimension> {
    /// k vectors
    k_abs: Array<f64, D>,
    /// Vector of weight functions for each component in multiple dimensions.
    weight_functions: Vec<FFTWeightFunctions<T, D>>,
    /// Lanczos sigma factor
    lanczos_sigma: Option<Array<f64, D>>,
    /// Vector of forward Fourier transforms in each dimensions
    forward_transforms: Vec<Arc<dyn Fft<T>>>,
    /// Vector of inverse Fourier transforms in each dimensions
    inverse_transforms: Vec<Arc<dyn Fft<T>>>,
}

impl<T, D: Dimension + 'static> PeriodicConvolver<T, D>
where
    T: FftNum + DualNum<f64>,
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    pub fn new_2d(
        axes: &[&Axis],
        angle: Angle,
        weight_functions: &[WeightFunctionInfo<T>],
        lanczos: Option<i32>,
    ) -> Arc<dyn Convolver<T, D>> {
        let f = |k: &mut Array<f64, D::Larger>| {
            let k_y =
                (&k.index_axis(Axis(0), 1) - &k.index_axis(Axis(0), 0) * angle.cos()) / angle.sin();
            k.index_axis_mut(Axis(0), 1).assign(&k_y);
        };
        Self::new(axes, f, weight_functions, lanczos)
    }

    pub fn new_3d(
        axes: &[&Axis],
        angles: [Angle; 3],
        weight_functions: &[WeightFunctionInfo<T>],
        lanczos: Option<i32>,
    ) -> Arc<dyn Convolver<T, D>> {
        let f = |k: &mut Array<f64, D::Larger>| {
            let [alpha, beta, gamma] = angles;
            let [k_u, k_v, k_w] = [0, 1, 2].map(|i| k.index_axis(Axis(0), i));
            let k_y = (&k_v - &k_u * gamma.cos()) / gamma.sin();
            let xi = (alpha.cos() - gamma.cos() * beta.cos()) / gamma.sin();
            let zeta = (1.0 - beta.cos().powi(2) - xi * xi).sqrt();
            let k_z = ((gamma.cos() * xi / gamma.sin() - beta.cos()) * &k_u
                - xi / gamma.sin() * &k_v
                + &k_w)
                / zeta;
            k.index_axis_mut(Axis(0), 1).assign(&k_y);
            k.index_axis_mut(Axis(0), 2).assign(&k_z);
        };
        Self::new(axes, f, weight_functions, lanczos)
    }

    pub fn new<F: Fn(&mut Array<f64, D::Larger>)>(
        axes: &[&Axis],
        non_orthogonal_correction: F,
        weight_functions: &[WeightFunctionInfo<T>],
        lanczos: Option<i32>,
    ) -> Arc<dyn Convolver<T, D>> {
        // initialize the Fourier transform
        let mut planner = FftPlanner::new();
        let mut forward_transforms = Vec::with_capacity(axes.len());
        let mut inverse_transforms = Vec::with_capacity(axes.len());
        let mut k_vec = Vec::with_capacity(axes.len());
        let mut lengths = Vec::with_capacity(axes.len());
        for ax in axes {
            let points = ax.grid.len();
            forward_transforms.push(planner.plan_fft_forward(points));
            inverse_transforms.push(planner.plan_fft_inverse(points));
            let (min, max) = (-(points as isize / 2), (points as isize - 1) / 2);
            let k_x: Array1<_> = (0..=max)
                .chain(min..0)
                .map(|i| 2.0 * PI * i as f64 / ax.length())
                .collect();

            k_vec.push(k_x);
            lengths.push(ax.length());
        }

        // Calculate the full k vectors
        let mut dim = vec![k_vec.len()];
        k_vec.iter().for_each(|k_x| dim.push(k_x.len()));
        let mut k: Array<_, D::Larger> = Array::zeros(dim).into_dimensionality().unwrap();
        for (i, (mut k_i, k_x)) in k.outer_iter_mut().zip(k_vec.iter()).enumerate() {
            k_i.lanes_mut(Axis_nd(i))
                .into_iter()
                .for_each(|mut l| l.assign(k_x));
        }

        // Correction for non-orthogonal coordinate systems
        non_orthogonal_correction(&mut k);

        // Calculate the absolute value of the k vector
        let mut k_abs = Array::zeros(k.raw_dim().remove_axis(Axis_nd(0)));
        for k_i in k.outer_iter() {
            k_abs.add_assign(&k_i.mapv(|k| k.powi(2)));
        }
        k_abs.map_inplace(|k| *k = k.sqrt());

        // Lanczos sigma factor
        let lanczos_sigma = lanczos.map(|exp| {
            let mut lanczos = Array::ones(k_abs.raw_dim());
            for (i, (k_x, &l)) in k_vec.iter().zip(lengths.iter()).enumerate() {
                let points = k_x.len();
                let m2 = if points % 2 == 0 {
                    points as f64 + 2.0
                } else {
                    points as f64 + 1.0
                };
                let l_x = k_x.mapv(|k| (k * l / m2).sph_j0().powi(exp));
                for mut l in lanczos.lanes_mut(Axis_nd(i)) {
                    l *= &l_x;
                }
            }
            lanczos
        });

        // calculate weight functions in Fourier space and weight constants
        let mut fft_weight_functions = Vec::with_capacity(weight_functions.len());
        for wf in weight_functions {
            // Calculates the weight functions values from `k_abs`
            // Pre-allocation of empty `Vec`
            let mut scal_comp = Vec::with_capacity(wf.scalar_component_weighted_densities.len());
            // Filling array with scalar component-wise weight functions
            for wf_i in &wf.scalar_component_weighted_densities {
                scal_comp.push(wf_i.fft_scalar_weight_functions(&k_abs, &lanczos_sigma));
            }

            // Pre-allocation of empty `Vec`
            let mut vec_comp = Vec::with_capacity(wf.vector_component_weighted_densities.len());
            // Filling array with vector-valued component-wise weight functions
            for wf_i in &wf.vector_component_weighted_densities {
                vec_comp.push(wf_i.fft_vector_weight_functions(&k_abs, &k, &lanczos_sigma));
            }

            // Pre-allocation of empty `Vec`
            let mut scal_fmt = Vec::with_capacity(wf.scalar_fmt_weighted_densities.len());
            // Filling array with scalar FMT weight functions
            for wf_i in &wf.scalar_fmt_weighted_densities {
                scal_fmt.push(wf_i.fft_scalar_weight_functions(&k_abs, &lanczos_sigma));
            }

            // Pre-allocation of empty `Vec`
            let mut vec_fmt = Vec::with_capacity(wf.vector_fmt_weighted_densities.len());
            // Filling array with vector-valued FMT weight functions
            for wf_i in &wf.vector_fmt_weighted_densities {
                vec_fmt.push(wf_i.fft_vector_weight_functions(&k_abs, &k, &lanczos_sigma));
            }

            // Initializing `FFTWeightFunctions` structure
            fft_weight_functions.push(FFTWeightFunctions::<_, D> {
                segments: wf.component_index.len(),
                local_density: wf.local_density,
                scalar_component_weighted_densities: scal_comp,
                vector_component_weighted_densities: vec_comp,
                scalar_fmt_weighted_densities: scal_fmt,
                vector_fmt_weighted_densities: vec_fmt,
            });
        }

        Arc::new(Self {
            k_abs,
            weight_functions: fft_weight_functions,
            lanczos_sigma,
            forward_transforms,
            inverse_transforms,
        })
    }
}

impl<T: FftNum, D: Dimension> PeriodicConvolver<T, D> {
    fn transform(&self, transform: &Arc<dyn Fft<T>>, mut f: ArrayViewMut1<Complex<T>>) {
        if let Some(f) = f.as_slice_mut() {
            transform.process(f);
        } else {
            let mut f_cont = f.to_owned();
            transform.process(f_cont.as_slice_mut().unwrap());
            f.assign(&f_cont);
        }
        if let FftDirection::Inverse = transform.fft_direction() {
            let points = T::from_usize(transform.len()).unwrap();
            f.mapv_inplace(|x| x / points);
        }
    }

    fn forward_transform<D2: Dimension>(&self, f: ArrayView<T, D2>) -> Array<Complex<T>, D2> {
        let offset = D2::NDIM.unwrap() - D::NDIM.unwrap();
        let mut result = f.mapv(Complex::from);
        for (i, transform) in self.forward_transforms.iter().enumerate() {
            for r in result.lanes_mut(Axis_nd(i + offset)).into_iter() {
                self.transform(transform, r);
            }
        }
        result
    }

    fn inverse_transform<D2: Dimension>(&self, mut f: Array<Complex<T>, D2>) -> Array<T, D2> {
        let offset = D2::NDIM.unwrap() - D::NDIM.unwrap();
        for (i, transform) in self.inverse_transforms.iter().enumerate() {
            for r in f.lanes_mut(Axis_nd(i + offset)).into_iter() {
                self.transform(transform, r);
            }
        }
        f.mapv(|x| x.re)
    }
}

impl<T, D: Dimension> Convolver<T, D> for PeriodicConvolver<T, D>
where
    T: FftNum + DualNum<f64>,
    D::Larger: Dimension<Smaller = D>,
    <D::Larger as Dimension>::Larger: Dimension<Smaller = D::Larger>,
{
    fn convolve(&self, profile: Array<T, D>, weight_function: &WeightFunction<T>) -> Array<T, D> {
        // Forward transform
        let f_k = self.forward_transform(profile.view());

        // calculate weight function
        let w = weight_function
            .fft_scalar_weight_functions(&self.k_abs, &self.lanczos_sigma)
            .index_axis_move(Axis_nd(0), 0);

        // Inverse transform
        self.inverse_transform(f_k * w)
    }

    fn weighted_densities(&self, density: &Array<T, D::Larger>) -> Vec<Array<T, D::Larger>> {
        // Applying FFT to each row of the matrix `rho` saving the result in `rho_k`
        let rho_k = self.forward_transform(density.view());

        // Iterate over all contributions
        let mut weighted_densities_vec = Vec::with_capacity(self.weight_functions.len());
        for wf in &self.weight_functions {
            // number of weighted densities
            let n_wd = wf.n_weighted_densities(density.ndim() - 1);

            // Allocating new array for intended weighted densities
            let mut dim = vec![n_wd];
            density.shape().iter().skip(1).for_each(|&d| dim.push(d));
            let mut weighted_densities = Array::zeros(dim).into_dimensionality().unwrap();

            // Initilaizing row index for non-local weighted densities
            let mut k = 0;

            // Assigning possible local densities to the front of the array
            if wf.local_density {
                weighted_densities
                    .slice_axis_mut(Axis_nd(0), Slice::from(0..wf.segments))
                    .assign(density);
                k += wf.segments;
            }

            // Calculating weighted densities {scalar, component}
            for wf_i in &wf.scalar_component_weighted_densities {
                weighted_densities
                    .slice_axis_mut(Axis_nd(0), Slice::from(k..k + wf.segments))
                    .assign(&self.inverse_transform(&rho_k * wf_i));
                k += wf.segments;
            }

            // Calculating weighted densities {vector, component}
            for wf_i in &wf.vector_component_weighted_densities {
                for wf_i in wf_i.outer_iter() {
                    weighted_densities
                        .slice_axis_mut(Axis_nd(0), Slice::from(k..k + wf.segments))
                        .assign(
                            &self.inverse_transform((&rho_k * &wf_i).mapv(|x| x * Complex::i())),
                        );
                    k += wf.segments;
                }
            }

            // Calculating weighted densities {scalar, FMT}
            for wf_i in &wf.scalar_fmt_weighted_densities {
                weighted_densities
                    .index_axis_mut(Axis_nd(0), k)
                    .assign(&self.inverse_transform((&rho_k * wf_i).sum_axis(Axis_nd(0))));
                k += 1;
            }

            // Calculating weighted densities {vector, FMT}
            for wf_i in &wf.vector_fmt_weighted_densities {
                for wf_i in wf_i.outer_iter() {
                    weighted_densities.index_axis_mut(Axis_nd(0), k).assign(
                        &self.inverse_transform(
                            (&rho_k * &wf_i)
                                .sum_axis(Axis_nd(0))
                                .mapv(|x| x * Complex::i()),
                        ),
                    );
                    k += 1;
                }
            }

            // add weighted densities for this contribution to the result
            weighted_densities_vec.push(weighted_densities);
        }
        // Return
        weighted_densities_vec
    }

    fn functional_derivative(
        &self,
        partial_derivatives: &[Array<T, D::Larger>],
    ) -> Array<T, D::Larger> {
        // Allocate arrays for the the local contribution to the functional derivative
        // and the functional derivative in Fourier space
        let mut dim = vec![self.weight_functions[0].segments];
        partial_derivatives[0]
            .shape()
            .iter()
            .skip(1)
            .for_each(|&d| dim.push(d));
        let mut functional_deriv_local: Array<_, D::Larger> =
            Array::zeros(dim).into_dimensionality().unwrap();
        let mut functional_deriv_k = Array::zeros(functional_deriv_local.raw_dim());

        // Iterate over all contributions
        for (pd, wf) in partial_derivatives.iter().zip(&self.weight_functions) {
            // Multiplication of `partial_derivatives` with the weight functions in
            // Fourier space (convolution in real space); summation leads to
            // functional derivative: the rows in the array are selected from the
            // running variable `k` with the number of rows needed for this
            // particular contribution
            let mut k = 0;

            // If local densities are present, their contributions are added directly
            if wf.local_density {
                functional_deriv_local += &pd.slice_axis(Axis_nd(0), Slice::from(..wf.segments));
                k += wf.segments;
            }

            // Convolution of functional derivatives {scalar, component}
            for wf_i in &wf.scalar_component_weighted_densities {
                let pd_k = self
                    .forward_transform(pd.slice_axis(Axis_nd(0), Slice::from(k..k + wf.segments)));
                functional_deriv_k += &(&pd_k * wf_i);
                k += wf.segments;
            }

            // Convolution of functional derivatives {vector, component}
            for wf_i in &wf.vector_component_weighted_densities {
                for wf_i in wf_i.outer_iter() {
                    let pd_k = self.forward_transform(
                        pd.slice_axis(Axis_nd(0), Slice::from(k..k + wf.segments)),
                    );
                    functional_deriv_k -= &(pd_k * &wf_i).mapv(|x| x * Complex::i());
                    k += wf.segments;
                }
            }

            // Convolution of functional derivatives {scalar, FMT}
            for wf_i in &wf.scalar_fmt_weighted_densities {
                let pd_k = self.forward_transform(pd.index_axis(Axis_nd(0), k));
                functional_deriv_k += &(pd_k * wf_i);
                k += 1;
            }

            // Convolution of functional derivatives {vector, FMT}
            for wf_i in &wf.vector_fmt_weighted_densities {
                for wf_i in wf_i.outer_iter() {
                    let pd_k = self.forward_transform(pd.index_axis(Axis_nd(0), k));
                    functional_deriv_k -= &(pd_k * wf_i).mapv(|x| x * Complex::i());
                    k += 1;
                }
            }
        }

        // Return sum over non-local and local contributions
        self.inverse_transform(functional_deriv_k) + functional_deriv_local
    }
}
