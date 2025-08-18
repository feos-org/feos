#[cfg(feature = "pcsaft")]
mod pcsaft_binary;
#[cfg(feature = "pcsaft")]
mod pcsaft_pure;
#[cfg(feature = "pcsaft")]
pub use pcsaft_binary::PcSaftBinary;
#[cfg(feature = "pcsaft")]
pub use pcsaft_pure::PcSaftPure;

pub const MAX_ETA: f64 = 0.5;

pub const A0: [f64; 7] = [
    0.91056314451539,
    0.63612814494991,
    2.68613478913903,
    -26.5473624914884,
    97.7592087835073,
    -159.591540865600,
    91.2977740839123,
];
pub const A1: [f64; 7] = [
    -0.30840169182720,
    0.18605311591713,
    -2.50300472586548,
    21.4197936296668,
    -65.2558853303492,
    83.3186804808856,
    -33.7469229297323,
];
pub const A2: [f64; 7] = [
    -0.09061483509767,
    0.45278428063920,
    0.59627007280101,
    -1.72418291311787,
    -4.13021125311661,
    13.7766318697211,
    -8.67284703679646,
];
pub const B0: [f64; 7] = [
    0.72409469413165,
    2.23827918609380,
    -4.00258494846342,
    -21.00357681484648,
    26.8556413626615,
    206.5513384066188,
    -355.60235612207947,
];
pub const B1: [f64; 7] = [
    -0.57554980753450,
    0.69950955214436,
    3.89256733895307,
    -17.21547164777212,
    192.6722644652495,
    -161.8264616487648,
    -165.2076934555607,
];
pub const B2: [f64; 7] = [
    0.09768831158356,
    -0.25575749816100,
    -9.15585615297321,
    20.64207597439724,
    -38.80443005206285,
    93.6267740770146,
    -29.66690558514725,
];

// Dipole parameters
pub const AD: [[f64; 3]; 5] = [
    [0.30435038064, 0.95346405973, -1.16100802773],
    [-0.13585877707, -1.83963831920, 4.52586067320],
    [1.44933285154, 2.01311801180, 0.97512223853],
    [0.35569769252, -7.37249576667, -12.2810377713],
    [-2.06533084541, 8.23741345333, 5.93975747420],
];

pub const BD: [[f64; 3]; 5] = [
    [0.21879385627, -0.58731641193, 3.48695755800],
    [-1.18964307357, 1.24891317047, -14.9159739347],
    [1.16268885692, -0.50852797392, 15.3720218600],
    [0.0; 3],
    [0.0; 3],
];

pub const CD: [[f64; 3]; 4] = [
    [-0.06467735252, -0.95208758351, -0.62609792333],
    [0.19758818347, 2.99242575222, 1.29246858189],
    [-0.80875619458, -2.38026356489, 1.65427830900],
    [0.69028490492, -0.27012609786, -3.43967436378],
];

#[cfg(test)]
#[cfg(feature = "pcsaft")]
pub mod test {
    use super::{PcSaftBinary, PcSaftPure};
    use feos::pcsaft::{
        PcSaft, PcSaftAssociationRecord, PcSaftBinaryRecord, PcSaftParameters, PcSaftRecord,
    };
    use feos_core::FeosResult;
    use feos_core::parameter::{AssociationRecord, PureRecord};
    use std::sync::Arc;

    pub fn pcsaft() -> FeosResult<(PcSaftPure<f64, 8>, Arc<PcSaft>)> {
        let m = 1.5;
        let sigma = 3.4;
        let epsilon_k = 180.0;
        let mu = 2.2;
        let kappa_ab = 0.03;
        let epsilon_k_ab = 2500.;
        let na = 2.0;
        let nb = 1.0;
        let params = PcSaftParameters::new_pure(PureRecord::with_association(
            Default::default(),
            0.0,
            PcSaftRecord::new(m, sigma, epsilon_k, mu, 0.0, None, None, None),
            vec![AssociationRecord::new(
                Some(PcSaftAssociationRecord::new(kappa_ab, epsilon_k_ab)),
                na,
                nb,
                0.0,
            )],
        ))?;
        let eos = Arc::new(PcSaft::new(params));
        let params = [m, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab, na, nb];
        Ok((PcSaftPure(params), eos))
    }

    pub fn pcsaft_binary() -> FeosResult<(PcSaftBinary<f64, 8>, Arc<PcSaft>)> {
        let params = [
            [1.5, 3.4, 180.0, 2.2, 0.03, 2500., 2.0, 1.0],
            [2.5, 3.6, 250.0, 1.2, 0.015, 1500., 1.0, 2.0],
        ];
        let kij = 0.15;
        let records = params.map(|p| {
            PureRecord::with_association(
                Default::default(),
                0.0,
                PcSaftRecord::new(p[0], p[1], p[2], p[3], 0.0, None, None, None),
                vec![AssociationRecord::new(
                    Some(PcSaftAssociationRecord::new(p[4], p[5])),
                    p[6],
                    p[7],
                    0.0,
                )],
            )
        });
        let params_feos =
            PcSaftParameters::new_binary(records, Some(PcSaftBinaryRecord::new(kij)), vec![])?;
        let eos = Arc::new(PcSaft::new(params_feos));
        Ok((PcSaftBinary::new(params, kij), eos))
    }

    #[cfg(feature = "parameter_fit")]
    pub fn pcsaft_non_assoc() -> FeosResult<(PcSaftPure<4>, Arc<PcSaft>)> {
        let m = 1.5;
        let sigma = 3.4;
        let epsilon_k = 180.0;
        let mu = 2.2;
        let params = PcSaftParameters::new_pure(PureRecord::new(
            Default::default(),
            0.0,
            PcSaftRecord::new(m, sigma, epsilon_k, mu, 0.0, None, None, None),
        ))?;
        let eos = Arc::new(PcSaft::new(params));
        let params = [m, sigma, epsilon_k, mu];
        Ok((PcSaftPure(params), eos))
    }
}
