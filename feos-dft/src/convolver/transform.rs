use crate::geometry::Axis;
use ndarray::prelude::*;
use ndarray::*;
use num_dual::*;
use rustdct::{DctNum, DctPlanner, TransformType2And3};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::f64::consts::PI;
use std::ops::{DivAssign, SubAssign};
use std::rc::Rc;
use std::sync::Arc;

#[derive(Clone, Copy)]
enum SinCosTransform {
    SinForward,
    SinReverse,
    CosForward,
    CosReverse,
}

impl SinCosTransform {
    fn is_reverse(&self) -> bool {
        match self {
            Self::CosForward | Self::SinForward => false,
            Self::CosReverse | Self::SinReverse => true,
        }
    }
}

pub(super) trait FourierTransform<T: DualNum<f64>> {
    fn forward_transform(&self, f_r: ArrayView1<T>, f_k: ArrayViewMut1<T>, scalar: bool);

    fn back_transform(&self, f_k: ArrayViewMut1<T>, f_r: ArrayViewMut1<T>, scalar: bool);
}

pub(super) struct CartesianTransform<T> {
    dct: Arc<dyn TransformType2And3<T>>,
}

impl<T: DualNum<f64> + DctNum + ScalarOperand> CartesianTransform<T> {
    pub(super) fn new(axis: &Axis) -> (Rc<dyn FourierTransform<T>>, Array1<f64>) {
        let (s, k) = Self::init(axis);
        (Rc::new(s), k)
    }

    pub(super) fn new_cartesian(axis: &Axis) -> (Rc<Self>, Array1<f64>) {
        let (s, k) = Self::init(axis);
        (Rc::new(s), k)
    }

    fn init(axis: &Axis) -> (Self, Array1<f64>) {
        let points = axis.grid.len();
        let length = axis.length();
        let k_grid = (0..=points).map(|v| PI * v as f64 / length).collect();
        (
            Self {
                dct: DctPlanner::new().plan_dct2(points),
            },
            k_grid,
        )
    }

    fn calculate_transform(&self, slice: &mut [T], transform: SinCosTransform) {
        match transform {
            SinCosTransform::CosForward => self.dct.process_dct2(slice),
            SinCosTransform::CosReverse => self.dct.process_dct3(slice),
            SinCosTransform::SinForward => self.dct.process_dst2(slice),
            SinCosTransform::SinReverse => self.dct.process_dst3(slice),
        }
    }

    fn transform(&self, mut f: ArrayViewMut1<T>, transform: SinCosTransform) {
        let mut f_slice = match transform {
            SinCosTransform::CosForward | SinCosTransform::CosReverse => f.slice_mut(s![..-1]),
            SinCosTransform::SinForward | SinCosTransform::SinReverse => f.slice_mut(s![1..]),
        };
        match f_slice.as_slice_mut() {
            Some(slice) => self.calculate_transform(slice, transform),
            None => {
                let mut slice = f_slice.to_owned();
                self.calculate_transform(slice.as_slice_mut().unwrap(), transform);
                f_slice.assign(&slice);
            }
        }
        if transform.is_reverse() {
            f.div_assign(T::from_f64(0.5).unwrap() * T::from_usize(self.dct.len()).unwrap())
        }
    }

    pub(super) fn forward_transform_inplace(&self, f: ArrayViewMut1<T>, scalar: bool) {
        if scalar {
            self.transform(f, SinCosTransform::CosForward);
        } else {
            self.transform(f, SinCosTransform::SinForward);
        }
    }

    pub(super) fn back_transform_inplace(&self, f: ArrayViewMut1<T>, scalar: bool) {
        if scalar {
            self.transform(f, SinCosTransform::CosReverse);
        } else {
            self.transform(f, SinCosTransform::SinReverse);
        }
    }
}

impl<T: DualNum<f64> + DctNum + ScalarOperand> FourierTransform<T> for CartesianTransform<T> {
    fn forward_transform(&self, f_r: ArrayView1<T>, mut f_k: ArrayViewMut1<T>, scalar: bool) {
        if scalar {
            f_k.slice_mut(s![..-1]).assign(&f_r);
        } else {
            f_k.slice_mut(s![1..]).assign(&f_r);
        }
        self.forward_transform_inplace(f_k, scalar);
    }

    fn back_transform(&self, mut f_k: ArrayViewMut1<T>, mut f_r: ArrayViewMut1<T>, scalar: bool) {
        self.back_transform_inplace(f_k.view_mut(), scalar);
        if scalar {
            f_r.assign(&f_k.slice(s![..-1]));
        } else {
            f_r.assign(&f_k.slice(s![1..]));
        }
    }
}

pub(super) struct SphericalTransform<T> {
    r_grid: Array1<f64>,
    k_grid: Array1<f64>,
    dct: Arc<dyn TransformType2And3<T>>,
}

impl<T: DualNum<f64> + DctNum + ScalarOperand> SphericalTransform<T> {
    pub(super) fn new(axis: &Axis) -> (Rc<dyn FourierTransform<T>>, Array1<f64>) {
        let points = axis.grid.len();
        let length = axis.length();
        let k_grid: Array1<_> = (0..=points).map(|v| PI * v as f64 / length).collect();
        (
            Rc::new(Self {
                r_grid: axis.grid.clone(),
                k_grid: k_grid.clone(),
                dct: DctPlanner::new().plan_dct2(points),
            }),
            k_grid,
        )
    }

    fn sine_transform<S1, S2>(
        &self,
        f_in: ArrayBase<S1, Ix1>,
        mut f_out: ArrayBase<S2, Ix1>,
        reverse: bool,
    ) where
        S1: Data<Elem = T>,
        S2: RawData<Elem = T> + DataMut,
    {
        if reverse {
            f_out.assign(&f_in.slice(s![1..]));
            self.dct.process_dst3(f_out.as_slice_mut().unwrap());
            f_out.div_assign(T::from_f64(0.5).unwrap() * T::from_usize(f_out.len()).unwrap());
        } else {
            let mut f_slice = f_out.slice_mut(s![1..]);
            f_slice.assign(&f_in);
            self.dct.process_dst2(f_slice.as_slice_mut().unwrap());
        }
    }

    fn cosine_transform<S1, S2>(
        &self,
        f_in: ArrayBase<S1, Ix1>,
        mut f_out: ArrayBase<S2, Ix1>,
        reverse: bool,
    ) where
        S1: Data<Elem = T>,
        S2: RawData<Elem = T> + DataMut,
    {
        if reverse {
            f_out.assign(&f_in.slice(s![..-1]));
            self.dct.process_dct3(f_out.as_slice_mut().unwrap());
            f_out.div_assign(T::from_f64(0.5).unwrap() * T::from_usize(f_out.len()).unwrap());
        } else {
            let mut f_slice = f_out.slice_mut(s![..-1]);
            f_slice.assign(&f_in);
            self.dct.process_dct2(f_slice.as_slice_mut().unwrap());
        }
    }
}

impl<T: DualNum<f64> + DctNum + ScalarOperand> FourierTransform<T> for SphericalTransform<T> {
    fn forward_transform(&self, f_r: ArrayView1<T>, mut f_k: ArrayViewMut1<T>, scalar: bool) {
        if scalar {
            self.sine_transform(&f_r * &self.r_grid, f_k.view_mut(), false);
        } else {
            self.cosine_transform(&f_r * &self.r_grid, f_k.view_mut(), false);
            let mut f_aux = Array::zeros(f_k.raw_dim());
            self.sine_transform(f_r, f_aux.view_mut(), false);
            f_k.sub_assign(&(&f_aux / &self.k_grid));
        }
        f_k.assign(&(&f_k / &self.k_grid));
        f_k[0] = T::zero();
    }

    fn back_transform(&self, f_k: ArrayViewMut1<T>, mut f_r: ArrayViewMut1<T>, scalar: bool) {
        if scalar {
            self.sine_transform(&f_k * &self.k_grid, f_r.view_mut(), true);
        } else {
            self.cosine_transform(&f_k * &self.k_grid, f_r.view_mut(), true);
            let mut f_aux = Array::zeros(f_r.raw_dim());
            self.sine_transform(f_k, f_aux.view_mut(), true);
            f_r.sub_assign(&(&f_aux / &self.r_grid));
        }
        f_r.assign(&(&f_r / &self.r_grid));
    }
}

pub(super) struct PolarTransform<T: DctNum> {
    r_grid: Array1<f64>,
    k_grid: Array1<f64>,
    fft: Arc<dyn Fft<T>>,
    j: [Array1<Complex<T>>; 2],
    k0: [f64; 2],
    alpha: f64,
    gamma: f64,
    l: f64,
}

impl<T: DualNum<f64> + DctNum + ScalarOperand> PolarTransform<T> {
    pub(super) fn new(axis: &Axis) -> (Rc<dyn FourierTransform<T>>, Array1<f64>) {
        let points = axis.grid.len();

        let mut alpha = 0.002_f64;
        for _ in 0..20 {
            alpha = -(1.0 - (-alpha).exp()).ln() / (points - 1) as f64;
        }
        let x0 = 0.5 * ((-alpha * points as f64).exp() + (-alpha * (points - 1) as f64).exp());
        let gamma = (alpha * (points - 1) as f64).exp();

        let l = axis.length();
        let k_grid: Array1<_> = (0..points)
            .map(|i| x0 * (alpha * i as f64).exp() * gamma / l)
            .collect();

        let k0 = (2.0 * alpha).exp() * (2.0 * alpha.exp() + (2.0 * alpha).exp() - 1.0)
            / ((1.0 + alpha.exp()).powi(2) * ((2.0 * alpha).exp() - 1.0));
        let k0v = (2.0 * alpha).exp() * (2.0 * alpha.exp() + (2.0 * alpha).exp() - 5.0 / 3.0)
            / ((1.0 + alpha.exp()).powi(2) * ((2.0 * alpha).exp() - 1.0));

        let fft = FftPlanner::new().plan_fft_forward(2 * points);
        let ifft = FftPlanner::new().plan_fft_inverse(2 * points);

        let mut j = Array1::from_shape_fn(2 * points, |i| {
            Complex::from(T::from(
                (gamma * x0 * (alpha * ((i + 1) as f64 - points as f64)).exp()).bessel_j1()
                    / ((2 * points) as f64),
            ))
        });
        ifft.process(j.as_slice_mut().unwrap());
        let mut jv = Array1::from_shape_fn(2 * points, |i| {
            Complex::from(T::from(
                (gamma * x0 * (alpha * ((i + 1) as f64 - points as f64)).exp()).bessel_j2()
                    / ((2 * points) as f64),
            ))
        });
        ifft.process(jv.as_slice_mut().unwrap());

        (
            Rc::new(Self {
                r_grid: axis.grid.clone(),
                k_grid: k_grid.clone(),
                fft,
                j: [j, jv],
                k0: [k0, k0v],
                alpha,
                gamma,
                l,
            }),
            k_grid,
        )
    }

    fn transform(
        &self,
        f_in: ArrayView1<T>,
        mut f_out: ArrayViewMut1<T>,
        scalar: bool,
        x_in: &Array1<f64>,
        x_out: &Array1<f64>,
        mut factor: f64,
    ) {
        let n = f_in.len();
        let (f_in, alpha, k0, j) = if scalar {
            (f_in.to_owned(), self.alpha, self.k0[0], &self.j[0])
        } else {
            factor *= factor;
            (&f_in / x_in, 2.0 * self.alpha, self.k0[1], &self.j[1])
        };
        let mut phi = Array1::from_shape_fn(2 * n, |i| {
            if i < n - 1 {
                (f_in[i] - f_in[i + 1]) * (-alpha * (n - i - 1) as f64).exp()
            } else {
                T::zero()
            }
        });
        phi[0] *= k0;
        let mut phi = phi.mapv(Complex::from);
        self.fft.process(phi.as_slice_mut().unwrap());
        phi *= j;
        self.fft.process(phi.as_slice_mut().unwrap());
        f_out.assign(&(phi.slice(s![..n]).map(|phi| phi.re * factor) / x_out));
    }
}

impl<T: DualNum<f64> + DctNum + ScalarOperand> FourierTransform<T> for PolarTransform<T> {
    fn forward_transform(&self, f_r: ArrayView1<T>, f_k: ArrayViewMut1<T>, scalar: bool) {
        self.transform(f_r, f_k, scalar, &self.r_grid, &self.k_grid, self.l);
    }

    fn back_transform(&self, f_k: ArrayViewMut1<T>, f_r: ArrayViewMut1<T>, scalar: bool) {
        self.transform(
            f_k.view(),
            f_r,
            scalar,
            &self.k_grid,
            &self.r_grid,
            self.gamma / self.l,
        );
    }
}

pub(super) struct NoTransform();

impl NoTransform {
    pub(super) fn new<T: DualNum<f64>>() -> (Rc<dyn FourierTransform<T>>, Array1<f64>) {
        (Rc::new(Self()), arr1(&[0.0]))
    }
}

impl<T: DualNum<f64>> FourierTransform<T> for NoTransform {
    fn forward_transform(&self, f: ArrayView1<T>, mut f_k: ArrayViewMut1<T>, _: bool) {
        f_k.assign(&f);
    }

    fn back_transform(&self, f_k: ArrayViewMut1<T>, mut f_r: ArrayViewMut1<T>, _: bool) {
        f_r.assign(&f_k);
    }
}
