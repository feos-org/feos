use ndarray::ArrayViewMut1;

#[derive(Clone, Debug, Copy)]
pub enum Loss {
    Linear,
    SoftL1(f64),
    Huber(f64),
    Cauchy(f64),
    Arctan(f64),
}

impl Loss {
    pub fn softl1(scaling_factor: f64) -> Self {
        Self::SoftL1(scaling_factor)
    }
    pub fn huber(scaling_factor: f64) -> Self {
        Self::Huber(scaling_factor)
    }
    pub fn cauchy(scaling_factor: f64) -> Self {
        Self::Cauchy(scaling_factor)
    }
    pub fn arctan(scaling_factor: f64) -> Self {
        Self::Arctan(scaling_factor)
    }

    pub fn apply(&self, res: &mut ArrayViewMut1<f64>) {
        match self {
            Self::Linear => (),
            Self::SoftL1(s) => {
                let s2 = s * s;
                let s2_inv = 1.0 / s2;
                res.mapv_inplace(|ri| {
                    (s2 * (2.0 * (((ri * ri * s2_inv) + 1.0).sqrt() - 1.0))).sqrt()
                })
            }
            Self::Huber(s) => {
                let s2 = s * s;
                let s2_inv = 1.0 / s2;
                res.mapv_inplace(|ri| {
                    if ri * ri * s2_inv <= 1.0 {
                        ri
                    } else {
                        (s2 * (2.0 * (ri / s).abs() - 1.0)).sqrt()
                    }
                })
            }
            Self::Cauchy(s) => {
                let s2 = s * s;
                let s2_inv = 1.0 / s2;
                res.mapv_inplace(|ri| (s2 * (1.0 + (ri * ri * s2_inv)).ln()).sqrt())
            }
            Self::Arctan(s) => {
                let s2 = s * s;
                let s2_inv = 1.0 / s2;
                res.mapv_inplace(|ri| (s2 * (ri * ri * s2_inv).atan()).sqrt())
            }
        }
    }
}
