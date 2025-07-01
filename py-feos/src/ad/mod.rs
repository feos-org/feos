#![cfg(feature = "pcsaft")]
use feos_ad::eos::{PcSaftBinary, PcSaftPure};
use feos_ad::{
    bubble_point_pressure, dew_point_pressure, equilibrium_liquid_density, liquid_density,
    vapor_pressure, ParametersAD,
};
use nalgebra::{Const, SVector, U1};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, Zip};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;
use quantity::{KELVIN, KILO, METER, MOL, PASCAL};
use typenum::P3;

#[pyclass(name = "PcSaft")]
pub struct PyPcSaft;

#[pymethods]
#[expect(clippy::type_complexity)]
impl PyPcSaft {
    #[staticmethod]
    fn vapor_pressure<'py>(
        parameters: PyReadonlyArray2<f64>,
        temperature: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<bool>>,
    ) {
        let (pressure, grad, status) =
            vapor_pressure_(parameters.as_array(), temperature.as_array());
        (
            pressure.to_pyarray(py),
            grad.to_pyarray(py),
            status.to_pyarray(py),
        )
    }

    #[staticmethod]
    fn liquid_density<'py>(
        parameters: PyReadonlyArray2<f64>,
        temperature: PyReadonlyArray1<f64>,
        pressure: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<bool>>,
    ) {
        let (rho, grad, status) = liquid_density_(
            parameters.as_array(),
            temperature.as_array(),
            pressure.as_array(),
        );
        (
            rho.to_pyarray(py),
            grad.to_pyarray(py),
            status.to_pyarray(py),
        )
    }

    #[staticmethod]
    fn equilibrium_liquid_density<'py>(
        parameters: PyReadonlyArray2<f64>,
        temperature: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<bool>>,
    ) {
        let (rho, grad, status) =
            equilibrium_liquid_density_(parameters.as_array(), temperature.as_array());
        (
            rho.to_pyarray(py),
            grad.to_pyarray(py),
            status.to_pyarray(py),
        )
    }

    #[staticmethod]
    #[expect(clippy::type_complexity)]
    fn bubble_point<'py>(
        parameters: PyReadonlyArray3<f64>,
        kij: PyReadonlyArray1<f64>,
        temperature: PyReadonlyArray1<f64>,
        liquid_molefracs: PyReadonlyArray1<f64>,
        pressure: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<bool>>,
    ) {
        let (rho, grad, status) = bubble_point_(
            parameters.as_array(),
            kij.as_array(),
            temperature.as_array(),
            liquid_molefracs.as_array(),
            pressure.as_array(),
        );
        (
            rho.to_pyarray(py),
            grad.to_pyarray(py),
            status.to_pyarray(py),
        )
    }

    #[staticmethod]
    #[expect(clippy::type_complexity)]
    fn dew_point<'py>(
        parameters: PyReadonlyArray3<f64>,
        kij: PyReadonlyArray1<f64>,
        temperature: PyReadonlyArray1<f64>,
        vapor_molefracs: PyReadonlyArray1<f64>,
        pressure: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<bool>>,
    ) {
        let (rho, grad, status) = dew_point_(
            parameters.as_array(),
            kij.as_array(),
            temperature.as_array(),
            vapor_molefracs.as_array(),
            pressure.as_array(),
        );
        (
            rho.to_pyarray(py),
            grad.to_pyarray(py),
            status.to_pyarray(py),
        )
    }
}

const PCSAFT_PARAMS: [&str; 8] = [
    "m",
    "sigma",
    "epsilon_k",
    "mu",
    "kappa_ab",
    "epsilon_k_ab",
    "na",
    "nb",
];

fn vapor_pressure_(
    parameters: ArrayView2<f64>,
    temperature: ArrayView1<f64>,
) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
    let pressure_dual = Zip::from(parameters.rows())
        .and(&temperature)
        .par_map_collect(|par, &t| {
            let par = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| par[i]);
            let pcsaft = PcSaftPure(par).wrap::<1>();
            let pcsaft = pcsaft.named_derivatives(PCSAFT_PARAMS);
            vapor_pressure(&pcsaft, t * KELVIN).map(|p| p.convert_into(PASCAL))
        });
    let status = pressure_dual.iter().map(|p| p.is_ok()).collect();
    let pressure_dual: Array1<_> = pressure_dual.into_iter().flatten().collect();
    let mut pressure = Array1::zeros(pressure_dual.len());
    let mut grad = Array2::zeros([pressure_dual.len(), 8]);
    Zip::from(grad.rows_mut())
        .and(&mut pressure)
        .and(&pressure_dual)
        .for_each(|mut grad, p, p_dual| {
            *p = p_dual.re;
            let eps = p_dual.eps.unwrap_generic(Const::<8>, U1).data.0[0].to_vec();
            grad.assign(&Array1::from(eps));
        });
    (pressure, grad, status)
}

fn liquid_density_(
    parameters: ArrayView2<f64>,
    temperature: ArrayView1<f64>,
    pressure: ArrayView1<f64>,
) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
    let density_dual = Zip::from(parameters.rows())
        .and(&temperature)
        .and(&pressure)
        .par_map_collect(|par, &t, &p| {
            let par = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| par[i]);
            let pcsaft = PcSaftPure(par).wrap::<1>();
            let pcsaft = pcsaft.named_derivatives(PCSAFT_PARAMS);
            liquid_density(&pcsaft, t * KELVIN, p * PASCAL)
                .map(|r| r.convert_into(KILO * MOL / METER.powi::<P3>()))
        });
    let status = density_dual.iter().map(|p| p.is_ok()).collect();
    let density_dual: Array1<_> = density_dual.into_iter().flatten().collect();
    let mut density = Array1::zeros(density_dual.len());
    let mut grad = Array2::zeros([density_dual.len(), 8]);
    Zip::from(grad.rows_mut())
        .and(&mut density)
        .and(&density_dual)
        .for_each(|mut grad, p, p_dual| {
            *p = p_dual.re;
            let eps = p_dual.eps.unwrap_generic(Const::<8>, U1).data.0[0].to_vec();
            grad.assign(&Array1::from(eps));
        });
    (density, grad, status)
}

fn equilibrium_liquid_density_(
    parameters: ArrayView2<f64>,
    temperature: ArrayView1<f64>,
) -> (Array1<f64>, Array2<f64>, Array1<bool>) {
    let pressure_dual = Zip::from(parameters.rows())
        .and(&temperature)
        .par_map_collect(|par, &t| {
            let par = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| par[i]);
            let pcsaft = PcSaftPure(par).wrap::<1>();
            let pcsaft = pcsaft.named_derivatives(PCSAFT_PARAMS);
            equilibrium_liquid_density(&pcsaft, t * KELVIN)
                .map(|(_, r)| r.convert_into(KILO * MOL / METER.powi::<P3>()))
        });
    let status = pressure_dual.iter().map(|p| p.is_ok()).collect();
    let pressure_dual: Array1<_> = pressure_dual.into_iter().flatten().collect();
    let mut pressure = Array1::zeros(pressure_dual.len());
    let mut grad = Array2::zeros([pressure_dual.len(), 8]);
    Zip::from(grad.rows_mut())
        .and(&mut pressure)
        .and(&pressure_dual)
        .for_each(|mut grad, p, p_dual| {
            *p = p_dual.re;
            let eps = p_dual.eps.unwrap_generic(Const::<8>, U1).data.0[0].to_vec();
            grad.assign(&Array1::from(eps));
        });
    (pressure, grad, status)
}

fn bubble_point_(
    parameters: ArrayView3<f64>,
    kij: ArrayView1<f64>,
    temperature: ArrayView1<f64>,
    liquid_molefracs: ArrayView1<f64>,
    pressure: ArrayView1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<bool>) {
    let pressure_dual = Zip::from(parameters.outer_iter())
        .and(kij)
        .and(temperature)
        .and(liquid_molefracs)
        .and(pressure)
        .par_map_collect(|par, &kij, &t, &x, &p| {
            let parameters = [0, 1].map(|j| [0, 1, 2, 3, 4, 5, 6, 7].map(|i| par[[j, i]]));
            let pcsaft = PcSaftBinary::new(parameters, kij).wrap();
            let pcsaft = pcsaft.named_derivatives(["k_ij"]);
            bubble_point_pressure(
                &pcsaft,
                t * KELVIN,
                Some(p * PASCAL),
                SVector::from([x, 1.0 - x]),
            )
            .map(|p| p.convert_into(PASCAL))
        });
    let status = pressure_dual.iter().map(|p| p.is_ok()).collect();
    let pressure_dual: Array1<_> = pressure_dual.into_iter().flatten().collect();
    let mut pressure = Array1::zeros(pressure_dual.len());
    let mut grad = Array1::zeros(pressure_dual.len());
    Zip::from(&mut grad)
        .and(&mut pressure)
        .and(&pressure_dual)
        .for_each(|grad, p, p_dual| {
            *p = p_dual.re;
            *grad = p_dual.eps.unwrap();
        });
    (pressure, grad, status)
}

fn dew_point_(
    parameters: ArrayView3<f64>,
    kij: ArrayView1<f64>,
    temperature: ArrayView1<f64>,
    vapor_molefracs: ArrayView1<f64>,
    pressure: ArrayView1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<bool>) {
    let pressure_dual = Zip::from(parameters.outer_iter())
        .and(kij)
        .and(temperature)
        .and(vapor_molefracs)
        .and(pressure)
        .par_map_collect(|par, &kij, &t, &y, &p| {
            let parameters = [0, 1].map(|j| [0, 1, 2, 3, 4, 5, 6, 7].map(|i| par[[j, i]]));
            let pcsaft = PcSaftBinary::new(parameters, kij).wrap();
            let pcsaft = pcsaft.named_derivatives(["k_ij"]);
            dew_point_pressure(
                &pcsaft,
                t * KELVIN,
                Some(p * PASCAL),
                SVector::from([y, 1.0 - y]),
            )
            .map(|p| p.convert_into(PASCAL))
        });
    let status = pressure_dual.iter().map(|p| p.is_ok()).collect();
    let pressure_dual: Array1<_> = pressure_dual.into_iter().flatten().collect();
    let mut pressure = Array1::zeros(pressure_dual.len());
    let mut grad = Array1::zeros(pressure_dual.len());
    Zip::from(&mut grad)
        .and(&mut pressure)
        .and(&pressure_dual)
        .for_each(|grad, p, p_dual| {
            *p = p_dual.re;
            *grad = p_dual.eps.unwrap();
        });
    (pressure, grad, status)
}
