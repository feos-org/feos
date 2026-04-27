/// Loss function applied to the residual.
///
/// Functions are not directly applied to a given residual.
/// Instead, they define the IRLS weight which is multiplied
/// by the residual to obtain the loss.
#[derive(Debug, Clone)]
pub enum LossFunction {
    /// L2 loss (all weights are 1.0)
    L2,
    /// Huber loss with dimensionless threshold `delta`.
    Huber { delta: f64 },
}

impl Default for LossFunction {
    fn default() -> Self {
        Self::L2
    }
}

impl LossFunction {
    /// Iteratively reweighted least squares weight (inverse prefactor of the residual).
    ///
    /// - L2: 1.0
    /// - Huber: `min(delta / |r|, 1.0)`
    #[inline(always)]
    pub fn irls_weight(&self, r: f64) -> f64 {
        match self {
            // L2: weight is constant (1.0).
            LossFunction::L2 => 1.0,
            // Huber: weight depends on current residual magnitude and delta.
            LossFunction::Huber { delta } => (delta / r.abs()).min(1.0),
        }
    }

    /// Square root of weight function.
    #[inline(always)]
    pub fn irls_weight_sqrt(&self, r: f64) -> f64 {
        self.irls_weight(r).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huber_within_threshold() {
        // inside threshold, huber is L2 and weight is 1.0
        let loss = LossFunction::Huber { delta: 0.1 };
        assert_eq!(loss.irls_weight(0.05), 1.0);
        assert_eq!(loss.irls_weight(0.1), 1.0);
        assert_eq!(loss.irls_weight_sqrt(0.1), 1.0);
    }

    #[test]
    fn huber_beyond_threshold() {
        // above threshold, weight is decreased.
        let loss = LossFunction::Huber { delta: 0.1 };
        // |r| = 0.2 → w = 0.1/0.2 = 0.5
        let w = loss.irls_weight(0.2);
        assert!((w - 0.5).abs() < 1e-12);
        // sqrt variant
        let ws = loss.irls_weight_sqrt(0.2);
        assert!((ws - 0.5f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn huber_negative_residual() {
        // w(-r) has to be w(r)
        let loss = LossFunction::Huber { delta: 0.1 };
        assert_eq!(loss.irls_weight(-0.2), loss.irls_weight(0.2));
    }
}
