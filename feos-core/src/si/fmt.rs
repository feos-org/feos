use super::*;
use ndarray::{Array, Dimension};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::fmt;
use typenum::{Quot, N1, N2, N3, P2, P3, P4};

const UNIT_SYMBOLS: [&str; 7] = ["m", "kg", "s", "A", "mol", "K", "cd"];

impl<
        Inner: fmt::Debug,
        T: Integer,
        L: Integer,
        M: Integer,
        I: Integer,
        THETA: Integer,
        N: Integer,
        J: Integer,
    > fmt::Debug for Quantity<Inner, SIUnit<T, L, M, I, THETA, N, J>>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)?;
        let unit = [T::I8, L::I8, M::I8, I::I8, THETA::I8, N::I8, J::I8]
            .iter()
            .zip(UNIT_SYMBOLS.iter())
            .filter_map(|(&u, &s)| match u {
                0 => None,
                1 => Some(s.to_owned()),
                _ => Some(format!("{s}^{u}")),
            })
            .collect::<Vec<String>>()
            .join(" ");

        write!(f, " {}", unit)
    }
}

macro_rules! impl_fmt {
    ($t:ident, $l:ident, $m:ident, $i:ident, $theta:ident, $n:ident, $unit:expr, $symbol:expr, $has_prefix:expr) => {
        impl<T> fmt::LowerExp for Quantity<T, SIUnit<$t, $l, $m, $i, $theta, $n, Z0>>
        where
            for<'a> &'a T: Div<f64>,
            for<'a> Quot<&'a T, f64>: fmt::LowerExp,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                (self / $unit).into_value().fmt(f)?;
                write!(f, " {}", $symbol)
            }
        }

        impl<T> fmt::UpperExp for Quantity<T, SIUnit<$t, $l, $m, $i, $theta, $n, Z0>>
        where
            for<'a> &'a T: Div<f64>,
            for<'a> Quot<&'a T, f64>: fmt::UpperExp,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                (self / $unit).into_value().fmt(f)?;
                write!(f, " {}", $symbol)
            }
        }

        impl<D: Dimension> fmt::Display
            for Quantity<Array<f64, D>, SIUnit<$t, $l, $m, $i, $theta, $n, Z0>>
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                (self / $unit).into_value().fmt(f)?;
                write!(f, " {}", $symbol)
            }
        }

        impl fmt::Display for Quantity<f64, SIUnit<$t, $l, $m, $i, $theta, $n, Z0>> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let (value, prefix) = get_prefix((self / $unit).into_value(), $has_prefix);
                if !((1e-2..1e4).contains(&value.abs()) || value == 0.0) {
                    write!(f, "{:e} {}{}", value, prefix, $symbol)
                } else {
                    value.fmt(f)?;
                    write!(f, " {}{}", prefix, $symbol)
                }
            }
        }
    };
}

impl_fmt!(P1, Z0, Z0, Z0, Z0, Z0, SECOND, "s", Some(KILO));
impl_fmt!(Z0, P1, Z0, Z0, Z0, Z0, METER, "m", Some(MEGA));
impl_fmt!(Z0, Z0, P1, Z0, Z0, Z0, GRAM, "g", Some(MEGA));
impl_fmt!(Z0, Z0, Z0, Z0, Z0, P1, MOL, "mol", Some(MEGA));
impl_fmt!(Z0, Z0, Z0, Z0, P1, Z0, KELVIN, "K", None);
impl_fmt!(N1, Z0, Z0, Z0, Z0, Z0, HERTZ, "Hz", Some(PETA));
impl_fmt!(N2, P1, P1, Z0, Z0, Z0, NEWTON, "N", Some(PETA));
impl_fmt!(N2, N1, P1, Z0, Z0, Z0, PASCAL, "Pa", Some(PETA));
impl_fmt!(N2, P2, P1, Z0, Z0, Z0, JOULE, "J", Some(PETA));
impl_fmt!(N3, P2, P1, Z0, Z0, Z0, WATT, "W", Some(PETA));
impl_fmt!(P1, Z0, Z0, P1, Z0, Z0, COULOMB, "C", None);
impl_fmt!(N3, P2, P1, N1, Z0, Z0, VOLT, "V", Some(PETA));
impl_fmt!(P4, N2, N1, P2, Z0, Z0, FARAD, "F", Some(PETA));
impl_fmt!(N3, P2, P1, N2, Z0, Z0, OHM, "Ω", Some(PETA));
impl_fmt!(P3, N2, N1, P2, Z0, Z0, SIEMENS, "S", Some(PETA));
impl_fmt!(N2, P2, P1, N1, Z0, Z0, WEBER, "Wb", Some(PETA));
impl_fmt!(N2, Z0, P1, N1, Z0, Z0, TESLA, "T", Some(PETA));
impl_fmt!(N2, P2, P1, N2, Z0, Z0, HENRY, "H", Some(PETA));

const M2: Area = Quantity(1.0, PhantomData);
const M3: Volume = Quantity(1.0, PhantomData);
const KG: Mass = KILOGRAM;
const JMK: MolarEntropy = Quantity(1.0, PhantomData);
const JKGK: SpecificEntropy = Quantity(1.0, PhantomData);
const WMK: ThermalConductivity = Quantity(1.0, PhantomData);

impl_fmt!(Z0, N3, Z0, Z0, Z0, P1, MOL / M3, "mol/m³", Some(MEGA));
impl_fmt!(Z0, N2, Z0, Z0, Z0, P1, MOL / M2, "mol/m²", Some(MEGA));
impl_fmt!(Z0, N1, Z0, Z0, Z0, P1, MOL / METER, "mol/m", Some(MEGA));
impl_fmt!(Z0, P3, Z0, Z0, Z0, N1, M3 / MOL, "m³/mol", None);
impl_fmt!(Z0, P3, Z0, Z0, N1, N1, M3 / MOL / KELVIN, "m³/mol/K", None);
impl_fmt!(Z0, N3, P1, Z0, Z0, Z0, GRAM / M3, "g/m³", Some(MEGA));
impl_fmt!(N2, Z0, P1, Z0, Z0, Z0, NEWTON / METER, "N/m", Some(PETA));
impl_fmt!(N1, P2, P1, Z0, Z0, Z0, JOULE * SECOND, "J*s", Some(PETA));
impl_fmt!(N2, P2, P1, Z0, Z0, N1, JOULE / MOL, "J/mol", Some(PETA));
impl_fmt!(N2, P2, P1, Z0, N1, Z0, JOULE / KELVIN, "J/K", Some(PETA));
impl_fmt!(N2, P2, P1, Z0, N1, N1, JMK, "J/mol/K", Some(PETA));
impl_fmt!(N2, P2, Z0, Z0, Z0, Z0, JOULE / KG, "J/kg", Some(PETA));
impl_fmt!(N2, P2, Z0, Z0, N1, Z0, JKGK, "J/kg/K", Some(PETA));
impl_fmt!(N1, N1, P1, Z0, Z0, Z0, PASCAL * SECOND, "Pa*s", Some(PETA));
impl_fmt!(N1, P1, Z0, Z0, Z0, Z0, METER / SECOND, "m/s", Some(MEGA));
impl_fmt!(N1, P2, Z0, Z0, Z0, Z0, M2 / SECOND, "m²/s", None);
impl_fmt!(N3, P1, P1, Z0, N1, Z0, WMK, "W/m/K", Some(PETA));
impl_fmt!(Z0, Z0, P1, Z0, Z0, N1, GRAM / MOL, "g/mol", Some(MEGA));
impl_fmt!(Z0, P2, Z0, Z0, Z0, Z0, M2, "m²", None);
impl_fmt!(Z0, P3, Z0, Z0, Z0, Z0, M3, "m³", None);
impl_fmt!(N1, P3, N1, Z0, Z0, Z0, M3 / KG / SECOND, "m³/kg/s²", None);

fn get_prefix(value: f64, has_prefix: Option<f64>) -> (f64, &'static str) {
    if let Some(p) = has_prefix {
        let abs_value = value.abs();
        let e: i8 = if abs_value > PICO && abs_value < p {
            (abs_value.log10().floor() as i8).div_euclid(3) * 3
        } else {
            0
        };
        let prefix = 10.0f64.powi(e as i32);
        return (value / prefix, PREFIX_SYMBOLS.get(&e).unwrap());
    }
    (value, "")
}

static PREFIX_SYMBOLS: Lazy<HashMap<i8, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert(0, " ");
    m.insert(-24, "y");
    m.insert(-21, "z");
    m.insert(-18, "a");
    m.insert(-15, "f");
    m.insert(-12, "p");
    m.insert(-9, "n");
    m.insert(-6, "µ");
    m.insert(-3, "m");
    m.insert(3, "k");
    m.insert(6, "M");
    m.insert(9, "G");
    m.insert(12, "T");
    m.insert(15, "P");
    m.insert(18, "E");
    m.insert(21, "Z");
    m.insert(24, "Y");
    m
});

#[cfg(test)]
mod tests {
    use crate::si::*;
    use ndarray::arr1;

    #[test]
    fn test_fmt_si() {
        assert_eq!(format!("{:.3}", RGAS), "8.314  J/mol/K");
    }

    #[test]
    fn test_fmt_exp() {
        assert_eq!(format!("{:e}", PICO * METER), "1e-12 m");
        assert_eq!(format!("{:E}", 50.0 * KILO * GRAM), "5E4 g");
    }

    #[test]
    fn test_fmt_arr() {
        assert_eq!(
            format!("{}", arr1(&[273.15, 323.15]) * KELVIN),
            "[273.15, 323.15] K"
        );
        assert_eq!(format!("{:e}", arr1(&[3.0, 5.0]) * BAR), "[3e5, 5e5] Pa");
    }

    #[test]
    fn test_fmt_zero() {
        assert_eq!(format!("{}", 0.0 * KELVIN), "0 K");
        assert_eq!(format!("{:.2}", 0.0 * PASCAL), "0.00  Pa");
    }
}
