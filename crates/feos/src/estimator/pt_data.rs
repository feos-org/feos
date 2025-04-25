use feos_core::StateVec;

trait BulkPropertyCalculation {
    fn name(&self) -> String;

    fn calcualte(&self, states: StateVec) -> Array1<f64>;
}

pub struct PTData {}
