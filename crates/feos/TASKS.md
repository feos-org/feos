# get the imports and interconnect the files
## src/cubic/mod.rs

[x] use feos_core::parameter::Parameter;
[x] use feos_core::si::{MolarWeight, GRAM, MOL};
[X] use feos_core::{Components, Residual, StateHD};

[x] use ndarray::{Array1, ScalarOperand, Zip};
[x] use num_dual::DualNum;
[x] use std::f64::consts::SQRT_2;
[x] use std::fmt;
[x] use std::sync::Arc;
[x] mod parameters;


### Finding Problems:
[ ] use feos_core::{EoSresult}

### Still to implement:
[ ] mod alpha;
[ ] use alpha::{Alpha, AlphaFunction, PengRobinson1976, RedlichKwong1972};
[ ] mod mixing_rules;
[ ] use mixing_rules::{MixingRule, MixingRuleFunction, MixtureParameters, Quadratic};
[ ] use parameters::CubicParameters;


## src/cubic/mod.rs
[x] use feos_core::parameter::{Identifier, Parameter, PureRecord};
[x] use ndarray::{Array1, Array2};
[x] use num_traits::Zero;
[x] use serde::{Deserialize, Serialize};

### Finding Problems:
[ ] use feos_core::parameter::ParameterError


## src/cubic/alpha
