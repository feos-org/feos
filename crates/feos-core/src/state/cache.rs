use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, OVector, Scalar};
use quantity::*;
use std::ops::Sub;
use std::sync::OnceLock;

type Diff<T1, T2> = <T1 as Sub<T2>>::Output;

#[derive(Clone, Debug)]
#[expect(clippy::type_complexity)]
pub struct Cache<D: Scalar, N: Dim>
where
    DefaultAllocator: Allocator<N>,
{
    pub a: OnceLock<Energy<D>>,
    pub da_dt: OnceLock<Entropy<D>>,
    pub da_dv: OnceLock<Pressure<D>>,
    pub da_dn: OnceLock<MolarEnergy<OVector<D, N>>>,
    pub d2a_dt2: OnceLock<Quantity<D, Diff<_Entropy, _Temperature>>>,
    pub d2a_dv2: OnceLock<Quantity<D, Diff<_Pressure, _Volume>>>,
    pub d2a_dtdv: OnceLock<Quantity<D, Diff<_Pressure, _Temperature>>>,
    pub d2a_dndt: OnceLock<Quantity<OVector<D, N>, Diff<_MolarEnergy, _Temperature>>>,
    pub d2a_dndv: OnceLock<Quantity<OVector<D, N>, Diff<_MolarEnergy, _Volume>>>,
    pub d3a_dt3: OnceLock<Quantity<D, Diff<Diff<_Entropy, _Temperature>, _Temperature>>>,
    pub d3a_dv3: OnceLock<Quantity<D, Diff<Diff<_Pressure, _Volume>, _Volume>>>,
}

impl<D: Scalar + Copy, N: Dim> Cache<D, N>
where
    DefaultAllocator: Allocator<N>,
{
    pub fn new() -> Self {
        Self {
            a: OnceLock::new(),
            da_dt: OnceLock::new(),
            da_dv: OnceLock::new(),
            da_dn: OnceLock::new(),
            d2a_dt2: OnceLock::new(),
            d2a_dv2: OnceLock::new(),
            d2a_dtdv: OnceLock::new(),
            d2a_dndt: OnceLock::new(),
            d2a_dndv: OnceLock::new(),
            d3a_dt3: OnceLock::new(),
            d3a_dv3: OnceLock::new(),
        }
    }
}
