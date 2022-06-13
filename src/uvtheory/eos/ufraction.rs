use feos_core::StateHD;
use num_dual::DualNum;

trait UFraction<D: DualNum<f64>> {
    fn u_fraction(&self, state: &StateHD<D>) -> D;
}