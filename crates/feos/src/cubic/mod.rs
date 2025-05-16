use feos_core::parameter::Parameter;
use feos_core::{Components, Residual, StateHD}; // Whereis EosResult?
use ndarray::{Array1, ScalarOperand, Zip};
use num_dual::DualNum;
use quantity::{GRAM, MOL, MolarWeight};
use std::f64::consts::SQRT_2;
use std::fmt;
use std::sync::Arc;

mod parameters;
use parameters::CubicParameters;

/// To learn some testing setup.
pub fn dumb_sum(a: f64, b: f64) -> f64 {
    // This is a dumb function that just adds two numbers together.
    println!("we are in the dumb function");
    println!("Adding {} and {}", a, b);
    let c = a + b;
    println!("Result: {}", c);
    c
}

#[cfg(test)]
#[cfg(feature = "cubic")]
mod tests {
    use crate::cubic; // Import from parent module
    use approx::assert_relative_eq;

    #[test]
    fn test_dumb_equation() {
        let a = 1.0;
        let b = 2.0;

        let result = cubic::dumb_sum(a, b);
        assert_relative_eq!(result, a + b, epsilon = 1e-10);
    }

    #[test]
    fn fail_dumb_equation() {
        let a = 1.0;
        let b = 2.0;

        let deviation = 1.0;

        let result = cubic::dumb_sum(a, b);
        assert_ne!(result, a + b + deviation);
    }
}
