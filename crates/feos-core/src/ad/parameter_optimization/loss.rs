/// Loss function applied to the dimensionless relative residual
/// `r = (calc − exp) / exp` during LM fitting.
#[derive(Debug, Clone)]
pub enum LossFunction {
    /// L2 loss (all weights are 1.0)
    L2,
    /// Huber loss with dimensionless threshold `delta`.
    ///
    /// Points with `|r_rel| ≤ delta` are treated as L2.
    /// Beyond that, the effective residual is `sqrt(delta * |r_rel|)` (linear loss).
    Huber { delta: f64 },
}

impl Default for LossFunction {
    fn default() -> Self {
        Self::L2
    }
}

impl LossFunction {
    /// IRLS weight `w = ρ'(r) / r`.
    ///
    /// - L2: 1.0
    /// - Huber: `min(delta / |r|, 1.0)`
    pub fn irls_weight(&self, r: f64) -> f64 {
        match self {
            LossFunction::L2 => 1.0,
            LossFunction::Huber { delta } => (delta / r.abs()).min(1.0),
        }
    }

    /// Square root of weight function.
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
