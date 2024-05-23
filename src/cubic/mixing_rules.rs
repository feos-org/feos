use feos_core::parameter::Parameter;
use ndarray::Array1;
use num_dual::DualNum;

trait MixingRuleFunction {
    fn mixing_rule<D: DualNum<f64>, P: Parameter>(parameters: &P, molefracs: &Array1<D>) -> (D, D);
}
