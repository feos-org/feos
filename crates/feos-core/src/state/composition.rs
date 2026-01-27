use super::State;
use crate::equation_of_state::Residual;
use crate::{FeosError, FeosResult};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, Dyn, OVector, U1, U2, dvector, vector};
use num_dual::{DualNum, DualStruct};
use quantity::{Density, Moles, Quantity, SIUnit};

pub trait Composition<D: DualNum<f64> + Copy, N: Dim>
where
    DefaultAllocator: Allocator<N>,
{
    #[expect(clippy::type_complexity)]
    fn into_molefracs<E: Residual<N, D>>(
        self,
        eos: &E,
    ) -> FeosResult<(OVector<D, N>, Option<Moles<D>>)>;
    fn density(&self) -> Option<Density<D>> {
        None
    }
}

pub trait FullComposition<D: DualNum<f64> + Copy, N: Dim>: Composition<D, N>
where
    DefaultAllocator: Allocator<N>,
{
    fn into_moles<E: Residual<N, D>>(self, eos: &E) -> FeosResult<(OVector<D, N>, Moles<D>)>;
}

// trivial implementations
impl<D: DualNum<f64> + Copy, N: Dim> Composition<D, N> for (OVector<D, N>, Moles<D>)
where
    DefaultAllocator: Allocator<N>,
{
    fn into_molefracs<E: Residual<N, D>>(
        self,
        _: &E,
    ) -> FeosResult<(OVector<D, N>, Option<Moles<D>>)> {
        Ok((self.0, Some(self.1)))
    }
}

impl<D: DualNum<f64> + Copy, N: Dim> FullComposition<D, N> for (OVector<D, N>, Moles<D>)
where
    DefaultAllocator: Allocator<N>,
{
    fn into_moles<E: Residual<N, D>>(self, _: &E) -> FeosResult<(OVector<D, N>, Moles<D>)> {
        Ok((self.0, self.1))
    }
}

impl<D: DualNum<f64> + Copy, N: Dim> Composition<D, N> for (OVector<D, N>, Option<Moles<D>>)
where
    DefaultAllocator: Allocator<N>,
{
    fn into_molefracs<E: Residual<N, D>>(
        self,
        _: &E,
    ) -> FeosResult<(OVector<D, N>, Option<Moles<D>>)> {
        Ok((self.0, self.1))
    }
}

// copy the composition from a given state
impl<E, D: DualNum<f64> + Copy, N: Dim> Composition<D, N> for &State<E, N, D>
where
    DefaultAllocator: Allocator<N>,
{
    fn into_molefracs<E1: Residual<N, D>>(
        self,
        _: &E1,
    ) -> FeosResult<(OVector<D, N>, Option<Moles<D>>)> {
        Ok(((self.molefracs.clone()), self.total_moles))
    }
}

// a pure component needs no specification
impl<D: DualNum<f64> + Copy> Composition<D, U1> for () {
    fn into_molefracs<E: Residual<U1, D>>(
        self,
        _: &E,
    ) -> FeosResult<(OVector<D, U1>, Option<Moles<D>>)> {
        Ok(((vector![D::one()]), None))
    }
}
impl<D: DualNum<f64> + Copy> Composition<D, Dyn> for () {
    fn into_molefracs<E: Residual<Dyn, D>>(
        self,
        eos: &E,
    ) -> FeosResult<(OVector<D, Dyn>, Option<Moles<D>>)> {
        if eos.components() == 1 {
            Ok(((dvector![D::one()]), None))
        } else {
            Err(FeosError::UndeterminedState(
                "The composition needs to be specified for a system with more than one component."
                    .into(),
            ))
        }
    }
}

// a binary mixture can be specified by a scalar (x1)
impl<D: DualNum<f64> + Copy> Composition<D, U2> for D {
    fn into_molefracs<E: Residual<U2, D>>(
        self,
        _: &E,
    ) -> FeosResult<(OVector<D, U2>, Option<Moles<D>>)> {
        Ok(((vector![self, -self + 1.0]), None))
    }
}

// this cannot be implemented generically for D due to mising specialization
impl Composition<f64, Dyn> for f64 {
    fn into_molefracs<E: Residual>(
        self,
        eos: &E,
    ) -> FeosResult<(OVector<f64, Dyn>, Option<Moles>)> {
        if eos.components() == 2 {
            Ok(((dvector![self, 1.0 - self]), None))
        } else {
            Err(FeosError::UndeterminedState(format!(
                "A scalar ({}) can only be used to specify a binary mixture!",
                self
            )))
        }
    }
}

// a pure component can be specified by the total mole number
impl<D: DualNum<f64> + Copy> Composition<D, U1> for Moles<D> {
    fn into_molefracs<E: Residual<U1, D>>(
        self,
        _: &E,
    ) -> FeosResult<(OVector<D, U1>, Option<Moles<D>>)> {
        Ok(((vector![D::one()]), Some(self)))
    }
}

impl<D: DualNum<f64> + Copy> FullComposition<D, U1> for Moles<D> {
    fn into_moles<E: Residual<U1, D>>(self, _: &E) -> FeosResult<(OVector<D, U1>, Moles<D>)> {
        Ok(((vector![D::one()]), self))
    }
}

impl<D: DualNum<f64> + Copy> Composition<D, Dyn> for Moles<D> {
    fn into_molefracs<E: Residual<Dyn, D>>(
        self,
        eos: &E,
    ) -> FeosResult<(OVector<D, Dyn>, Option<Moles<D>>)> {
        if eos.components() == 1 {
            Ok(((dvector![D::one()]), Some(self)))
        } else {
            Err(FeosError::UndeterminedState(format!(
                "A single mole number ({}) can only be used to specify a pure component!",
                self.re()
            )))
        }
    }
}

impl<D: DualNum<f64> + Copy> FullComposition<D, Dyn> for Moles<D> {
    fn into_moles<E: Residual<Dyn, D>>(self, eos: &E) -> FeosResult<(OVector<D, Dyn>, Moles<D>)> {
        if eos.components() == 1 {
            Ok(((dvector![D::one()]), self))
        } else {
            Err(FeosError::UndeterminedState(format!(
                "A single mole number ({}) can only be used to specify a pure component!",
                self.re()
            )))
        }
    }
}

// the mixture can be specified by its molefractions
//
// for a dynamic number of components, it is also possible to specify only the
// N-1 first components
impl<D: DualNum<f64> + Copy, N: Dim> Composition<D, N> for OVector<D, N>
where
    DefaultAllocator: Allocator<N>,
{
    fn into_molefracs<E: Residual<N, D>>(
        self,
        eos: &E,
    ) -> FeosResult<(OVector<D, N>, Option<Moles<D>>)> {
        (&self).into_molefracs(eos)
    }
}

impl<D: DualNum<f64> + Copy, N: Dim> Composition<D, N> for &OVector<D, N>
where
    DefaultAllocator: Allocator<N>,
{
    fn into_molefracs<E: Residual<N, D>>(
        self,
        eos: &E,
    ) -> FeosResult<(OVector<D, N>, Option<Moles<D>>)> {
        let sum = self.sum();
        if eos.components() == self.len() {
            Ok(((self.clone() / sum), None))
        } else if eos.components() == self.len() + 1 {
            let mut x = OVector::zeros_generic(N::from_usize(eos.components()), U1);
            for i in 0..self.len() {
                x[i] = self[i];
            }
            x[self.len()] = -sum + 1.0;
            Ok(((x), None))
        } else {
            Err(FeosError::UndeterminedState(format!(
                "The length of the composition vector ({}) does not match the number of components ({})!",
                self.len(),
                eos.components()
            )))
        }
    }
}

// the mixture can be specified by its moles
impl<D: DualNum<f64> + Copy, N: Dim> Composition<D, N> for Moles<OVector<D, N>>
where
    DefaultAllocator: Allocator<N>,
{
    fn into_molefracs<E: Residual<N, D>>(
        self,
        eos: &E,
    ) -> FeosResult<(OVector<D, N>, Option<Moles<D>>)> {
        (&self).into_molefracs(eos)
    }
}

impl<D: DualNum<f64> + Copy, N: Dim> FullComposition<D, N> for Moles<OVector<D, N>>
where
    DefaultAllocator: Allocator<N>,
{
    fn into_moles<E: Residual<N, D>>(self, eos: &E) -> FeosResult<(OVector<D, N>, Moles<D>)> {
        (&self).into_moles(eos)
    }
}

impl<D: DualNum<f64> + Copy, N: Dim> Composition<D, N> for &Moles<OVector<D, N>>
where
    DefaultAllocator: Allocator<N>,
{
    fn into_molefracs<E: Residual<N, D>>(
        self,
        eos: &E,
    ) -> FeosResult<(OVector<D, N>, Option<Moles<D>>)> {
        if eos.components() == self.len() {
            let total_moles = self.sum();
            Ok(((self.convert_to(total_moles)), Some(total_moles)))
        } else {
            Err(FeosError::UndeterminedState(format!(
                "The length of the composition vector ({}) does not match the number of components ({})!",
                self.len(),
                eos.components()
            )))
        }
    }
}

impl<D: DualNum<f64> + Copy, N: Dim> FullComposition<D, N> for &Moles<OVector<D, N>>
where
    DefaultAllocator: Allocator<N>,
{
    fn into_moles<E: Residual<N, D>>(self, eos: &E) -> FeosResult<(OVector<D, N>, Moles<D>)> {
        if eos.components() == self.len() {
            let total_moles = self.sum();
            Ok(((self.convert_to(total_moles)), total_moles))
        } else {
            Err(FeosError::UndeterminedState(format!(
                "The length of the composition vector ({}) does not match the number of components ({})!",
                self.len(),
                eos.components()
            )))
        }
    }
}

// the mixture can be specified with the partial density
impl<D: DualNum<f64> + Copy, N: Dim> Composition<D, N>
    for Quantity<OVector<D, N>, SIUnit<0, -3, 0, 0, 0, 1, 0>>
where
    DefaultAllocator: Allocator<N>,
{
    fn into_molefracs<E: Residual<N, D>>(
        self,
        eos: &E,
    ) -> FeosResult<(OVector<D, N>, Option<Moles<D>>)> {
        if eos.components() == self.len() {
            let density = self.sum();
            Ok(((self.convert_to(density)), None))
        } else {
            panic!(
                "The length of the composition vector ({}) does not match the number of components ({})!",
                self.len(),
                eos.components()
            )
        }
    }

    fn density(&self) -> Option<Density<D>> {
        Some(self.sum())
    }
}
