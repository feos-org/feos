//! Tapes for writing a function once and obtaining both its value and its Jacobian-vector
//! product (JVP).
//!
//! The function is written against the [Tape] trait and called with one of three passes:
//! - [Evaluate]: values only.
//! - [Record]: values; every nonlinear operation stores what its linear map needs.
//! - [Replay]: the same code runs on tangents; every nonlinear operation takes its entry from the
//!   recording in the same order and applies the linear map.
//!
//! Linear operations (sums, multiplication with constants, convolutions, integration) act the
//! same way on values and tangents and need no entry. Constants that are added to quantities
//! depending on the unknowns have to be created with [Tape::constant] (zero tangent). Local
//! functions of the temperature and the weighted densities ([LocalFunction]) use
//! [Tape::local_derivatives], which differentiates them with dual numbers.
#[cfg(feature = "rayon")]
use ndarray::parallel::prelude::*;
use ndarray::{
    Array, Array2, Array3, ArrayD, ArrayView2, Axis, Dimension, RemoveAxis, Slice, Zip,
    concatenate, s,
};
use num_dual::{Dual, DualNum, DualNumCopy, HyperDual64};
use std::convert::Infallible;

/// Linear map of a nonlinear operation: one factor per input, interpreted by the operation.
/// - `unary`: (f'(a), None), dy = f'(a) * da
/// - `mul`: (b, Some(a)), dy = b * da + a * db
/// - `local_derivatives`: (Jacobian (outputs x inputs x grid), None), dy = J dx
type Entry = (ArrayD<f64>, Option<ArrayD<f64>>);

// The chunk size is specifically chosen to be the default amount of grid points in a 1D density
// profile to switch to parallel evaluation only for 2D and 3D profiles.
#[cfg(feature = "rayon")]
const CHUNK_SIZE: usize = 2048;

/// A local function f(x) of the temperature and of variables x (variables x points, outputs x
/// points), generic in the number type so that the tape can differentiate it.
pub trait LocalFunction: Sync {
    type Error: Send;

    fn evaluate<N: DualNum<Primitive = f64> + Copy>(
        &self,
        temperature: N,
        x: ArrayView2<N>,
    ) -> Result<Array2<N>, Self::Error>;

    /// [Self::evaluate] for parallel evaluation with rayon.
    fn evaluate_parallel<N: DualNumCopy<Primitive = f64>>(
        &self,
        temperature: N,
        x: ArrayView2<N>,
    ) -> Result<Array2<N>, Self::Error> {
        #[cfg(feature = "rayon")]
        if x.shape()[1] > CHUNK_SIZE {
            let chunks: Vec<_> = x
                .axis_chunks_iter(Axis(1), CHUNK_SIZE)
                .into_par_iter()
                .map(|x| self.evaluate(temperature, x))
                .collect::<Result<_, _>>()?;
            let mut y = Array2::zeros((chunks[0].shape()[0], x.shape()[1]));
            for (mut y, c) in y.axis_chunks_iter_mut(Axis(1), CHUNK_SIZE).zip(&chunks) {
                y.assign(c);
            }
            return Ok(y);
        }
        self.evaluate(temperature, x)
    }
}

/// Value of a local function and its first derivatives (variables x grid).
pub type LocalDerivatives<N, D> = (Array<N, D>, Array<N, D>);

pub trait Tape<N: DualNum<Primitive = f64> + Copy = f64> {
    /// Error of an operation that can fail with the error `E`: `E` if it is evaluated, impossible
    /// in a replay.
    type Error<E>;

    /// The value of a result that cannot fail.
    fn infallible<A>(result: Result<A, Self::Error<Infallible>>) -> A;

    /// A constant (zero tangent).
    fn constant<D: Dimension>(&mut self, a: Array<N, D>) -> Array<N, D>;

    /// Elementwise function f with its derivative df(a, f(a)) (non-finite derivatives are set
    /// to 0).
    fn unary<D: Dimension>(
        &mut self,
        a: &Array<N, D>,
        f: impl Fn(N) -> N,
        df: impl Fn(N, N) -> N,
    ) -> Array<N, D>;

    /// a * b where both depend on the unknowns.
    fn mul<D: Dimension>(&mut self, a: &Array<N, D>, b: &Array<N, D>) -> Array<N, D>;

    /// The local function `f` of the variables `x` (variables x grid) and its first derivatives
    /// with respect to x, stacked as (df/dx_0, df/dx_1, ...).
    fn local_derivatives<D, L>(
        &mut self,
        temperature: N,
        x: &Array<N, D>,
        f: &L,
    ) -> Result<LocalDerivatives<N, D>, Self::Error<L::Error>>
    where
        N: Send + Sync,
        D: RemoveAxis,
        D::Larger: Dimension<Smaller = D>,
        L: LocalFunction;

    fn exp<D: Dimension>(&mut self, a: &Array<N, D>) -> Array<N, D> {
        self.unary(a, |a| a.exp(), |_, y| y)
    }

    fn ln<D: Dimension>(&mut self, a: &Array<N, D>) -> Array<N, D> {
        self.unary(a, |a| a.ln(), |a, _| a.recip())
    }

    fn ln_1p<D: Dimension>(&mut self, a: &Array<N, D>) -> Array<N, D> {
        self.unary(a, |a| a.ln_1p(), |a, _| (a + 1.0).recip())
    }
}

/// Variables x grid flattened to variables x points.
pub(crate) fn flatten<N, D: Dimension>(x: &Array<N, D>) -> ArrayView2<'_, N> {
    let n = x.shape()[0];
    x.view().into_shape_with_order((n, x.len() / n)).unwrap()
}

/// Rows x points restored to rows x grid (with the grid of `dim`).
pub(crate) fn unflatten<N, D: Dimension>(a: Array2<N>, mut dim: D) -> Array<N, D> {
    dim[0] = a.shape()[0];
    a.into_shape_with_order(dim).unwrap()
}

/// Values only.
pub struct Evaluate;

impl<N: DualNum<Primitive = f64> + Copy> Tape<N> for Evaluate {
    type Error<E> = E;

    fn infallible<A>(result: Result<A, Infallible>) -> A {
        let Ok(a) = result;
        a
    }

    fn constant<D: Dimension>(&mut self, a: Array<N, D>) -> Array<N, D> {
        a
    }

    fn unary<D: Dimension>(
        &mut self,
        a: &Array<N, D>,
        f: impl Fn(N) -> N,
        _: impl Fn(N, N) -> N,
    ) -> Array<N, D> {
        a.mapv(f)
    }

    fn mul<D: Dimension>(&mut self, a: &Array<N, D>, b: &Array<N, D>) -> Array<N, D> {
        a * b
    }

    /// First derivatives with dual numbers.
    fn local_derivatives<D, L>(
        &mut self,
        temperature: N,
        x: &Array<N, D>,
        f: &L,
    ) -> Result<LocalDerivatives<N, D>, L::Error>
    where
        N: Send + Sync,
        D: RemoveAxis,
        D::Larger: Dimension<Smaller = D>,
        L: LocalFunction,
    {
        let n_x = x.shape()[0];
        let t = Dual::from_re(temperature);
        let mut x_dual = flatten(x).mapv(Dual::from_re);
        let mut value = None;
        let mut derivatives = Vec::with_capacity(n_x);
        for i in 0..n_x {
            x_dual.row_mut(i).map_inplace(|x| x.eps = N::one());
            let y = f.evaluate_parallel(t, x_dual.view())?;
            value.get_or_insert_with(|| y.mapv(|y| y.re));
            derivatives.push(y.mapv(|y| y.eps));
            x_dual.row_mut(i).map_inplace(|x| x.eps = N::zero());
        }
        let derivatives: Vec<_> = derivatives.iter().map(|d| d.view()).collect();
        let derivatives = concatenate(Axis(0), &derivatives).unwrap();
        Ok((
            unflatten(value.unwrap(), x.raw_dim()),
            unflatten(derivatives, x.raw_dim()),
        ))
    }
}

/// Values and the linear maps of all nonlinear operations.
#[derive(Default)]
pub struct Record {
    entries: Vec<Entry>,
}

impl Record {
    pub fn new() -> Self {
        Self::default()
    }

    /// A replay of the recording (one per JVP).
    pub fn replay(&self) -> Replay<'_> {
        Replay {
            entries: self.entries.iter(),
        }
    }
}

impl Tape for Record {
    type Error<E> = E;

    fn infallible<A>(result: Result<A, Infallible>) -> A {
        let Ok(a) = result;
        a
    }

    fn constant<D: Dimension>(&mut self, a: Array<f64, D>) -> Array<f64, D> {
        a
    }

    fn unary<D: Dimension>(
        &mut self,
        a: &Array<f64, D>,
        f: impl Fn(f64) -> f64,
        df: impl Fn(f64, f64) -> f64,
    ) -> Array<f64, D> {
        let y = a.mapv(f);
        let factor = Zip::from(a).and(&y).map_collect(|&a, &y| {
            let d = df(a, y);
            if d.is_finite() { d } else { 0.0 }
        });
        self.entries.push((factor.into_dyn(), None));
        y
    }

    fn mul<D: Dimension>(&mut self, a: &Array<f64, D>, b: &Array<f64, D>) -> Array<f64, D> {
        self.entries
            .push((b.clone().into_dyn(), Some(a.clone().into_dyn())));
        a * b
    }

    /// First derivatives and their Jacobian with hyper-dual numbers ($\varepsilon_1$ on x_i,
    /// $\varepsilon_2$ on x_j, $j\leq i$).
    fn local_derivatives<D, L>(
        &mut self,
        temperature: f64,
        x: &Array<f64, D>,
        f: &L,
    ) -> Result<LocalDerivatives<f64, D>, L::Error>
    where
        D: RemoveAxis,
        D::Larger: Dimension<Smaller = D>,
        L: LocalFunction,
    {
        let n_x = x.shape()[0];
        let n_points = x.len() / n_x;
        let t = HyperDual64::from(temperature);
        let mut x_dual = flatten(x).mapv(HyperDual64::from);
        let mut value = None;
        let mut derivatives = Vec::with_capacity(n_x);
        let mut jacobian = None;
        for i in 0..n_x {
            x_dual.row_mut(i).map_inplace(|x| x.eps1 = 1.0);
            for j in 0..=i {
                x_dual.row_mut(j).map_inplace(|x| x.eps2 = 1.0);
                let y = f.evaluate_parallel(t, x_dual.view())?;
                let n = y.shape()[0];
                let jacobian =
                    jacobian.get_or_insert_with(|| Array3::zeros((n * (n_x + 1), n_x, n_points)));
                let mut block = |b: usize, j: usize, y: Array2<f64>| {
                    jacobian.slice_mut(s![b * n..(b + 1) * n, j, ..]).assign(&y)
                };
                if i == j {
                    value.get_or_insert_with(|| y.mapv(|y| y.re));
                    derivatives.push(y.mapv(|y| y.eps1));
                    block(0, i, y.mapv(|y| y.eps1));
                }
                block(1 + i, j, y.mapv(|y| y.eps1eps2));
                block(1 + j, i, y.mapv(|y| y.eps1eps2));
                x_dual.row_mut(j).map_inplace(|x| x.eps2 = 0.0);
            }
            x_dual.row_mut(i).map_inplace(|x| x.eps1 = 0.0);
        }

        // Jacobian of (f, df/dx) (outputs x inputs x grid)
        let jacobian = jacobian.unwrap();
        let mut dim = x.raw_dim().insert_axis(Axis(1));
        dim[0] = jacobian.shape()[0];
        dim[1] = n_x;
        self.entries.push((
            jacobian.into_shape_with_order(dim).unwrap().into_dyn(),
            None,
        ));

        let derivatives: Vec<_> = derivatives.iter().map(|d| d.view()).collect();
        let derivatives = concatenate(Axis(0), &derivatives).unwrap();
        Ok((
            unflatten(value.unwrap(), x.raw_dim()),
            unflatten(derivatives, x.raw_dim()),
        ))
    }
}

/// Tangents: applies the recorded linear maps in order.
pub struct Replay<'a> {
    entries: std::slice::Iter<'a, Entry>,
}

impl Replay<'_> {
    fn next(&mut self) -> &Entry {
        self.entries
            .next()
            .expect("replay is longer than the recording")
    }
}

impl Tape for Replay<'_> {
    type Error<E> = Infallible;

    fn infallible<A>(result: Result<A, Infallible>) -> A {
        let Ok(a) = result;
        a
    }

    fn constant<D: Dimension>(&mut self, a: Array<f64, D>) -> Array<f64, D> {
        Array::zeros(a.raw_dim())
    }

    fn unary<D: Dimension>(
        &mut self,
        a: &Array<f64, D>,
        _: impl Fn(f64) -> f64,
        _: impl Fn(f64, f64) -> f64,
    ) -> Array<f64, D> {
        let (factor, _) = self.next();
        a * &factor.view().into_dimensionality::<D>().unwrap()
    }

    fn mul<D: Dimension>(&mut self, a: &Array<f64, D>, b: &Array<f64, D>) -> Array<f64, D> {
        let (b0, Some(a0)) = self.next() else {
            panic!("replay does not match the recording")
        };
        let a0 = a0.view().into_dimensionality::<D>().unwrap();
        let b0 = b0.view().into_dimensionality::<D>().unwrap();
        a * &b0 + &a0 * b
    }

    fn local_derivatives<D, L>(
        &mut self,
        _: f64,
        x: &Array<f64, D>,
        _: &L,
    ) -> Result<LocalDerivatives<f64, D>, Infallible>
    where
        D: RemoveAxis,
        D::Larger: Dimension<Smaller = D>,
        L: LocalFunction,
    {
        // product of the recorded Jacobian (outputs x inputs x grid) with x (inputs x grid)
        let (jacobian, _) = self.next();
        let jacobian = jacobian.view().into_dimensionality::<D::Larger>().unwrap();
        let mut dim = x.raw_dim();
        dim[0] = jacobian.shape()[0];
        let mut y = Array::zeros(dim);
        for (mut y, j) in y.outer_iter_mut().zip(jacobian.outer_iter()) {
            for (j, x) in j.outer_iter().zip(x.outer_iter()) {
                y += &(&j * &x);
            }
        }

        // split into the tangents of f and of df/dx
        let n = y.shape()[0] / (x.shape()[0] + 1);
        Ok((
            y.slice_axis(Axis(0), Slice::from(..n)).to_owned(),
            y.slice_axis(Axis(0), Slice::from(n..)).to_owned(),
        ))
    }
}
