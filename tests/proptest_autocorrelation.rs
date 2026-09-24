//! Exact integer and metamorphic oracles for scalar autocorrelation.

use markov_chain_monte_carlo::Autocorrelation;
use proptest::prelude::*;

/// Expand the covariance into integer raw moments, without floating-point
/// centering or scaling. Inputs have 2..=32 elements in -16..=17, so every
/// intermediate fits in `i32` and converts exactly to `f64` before division.
fn exact_acf(samples: &[i16]) -> Vec<f64> {
    let count = i32::try_from(samples.len()).unwrap();
    let sum: i32 = samples.iter().map(|&value| i32::from(value)).sum();
    let squares: i32 = samples.iter().map(|&value| i32::from(value).pow(2)).sum();
    let denominator = count * count * squares - count * sum * sum;
    assert!(denominator > 0, "the generator guarantees distinct samples");

    (0..samples.len())
        .map(|lag| {
            let pairs = samples.len() - lag;
            let products: i32 = samples[..pairs]
                .iter()
                .zip(&samples[lag..])
                .map(|(&left, &right)| i32::from(left) * i32::from(right))
                .sum();
            let endpoints: i32 = samples[..pairs]
                .iter()
                .chain(&samples[lag..])
                .map(|&value| i32::from(value))
                .sum();
            let numerator = count * count * products - count * sum * endpoints
                + i32::try_from(pairs).unwrap() * sum * sum;
            f64::from(numerator) / f64::from(denominator)
        })
        .collect()
}

proptest! {
    /// Check affine transforms, time reversal, and bounded prefixes against exact ratios.
    #[test]
    fn acf_matches_exact_covariances(
        first in -16i16..=16,
        rest in prop::collection::vec(-16i16..=16, 0..31),
        lag_seed in 0usize..32,
        exponent in -900i32..=900,
        offset in -128i16..=128,
    ) {
        // Every generated input is admitted independently of production:
        // at least two finite samples, with the first two always distinct.
        let mut raw = vec![first, first + 1];
        raw.extend(rest);
        let expected = exact_acf(&raw);
        let samples: Vec<_> = raw.iter().map(|&value| f64::from(value)).collect();
        let reversed: Vec<_> = samples.iter().rev().copied().collect();
        let scale = -2.0_f64.powi(exponent);
        let affine: Vec<_> = raw
            .iter()
            .map(|&value| f64::from(value + offset) * scale)
            .collect();
        let max_lag = lag_seed % raw.len();

        for (label, input, budget) in [
            ("original", &samples, raw.len() - 1),
            ("reversed", &reversed, raw.len() - 1),
            ("affine", &affine, raw.len() - 1),
            ("bounded prefix", &samples, max_lag),
        ] {
            let acf = Autocorrelation::estimate(input, budget).map_err(|error| {
                TestCaseError::fail(format!(
                    "{label}: raw={raw:?}, lag={budget}, exponent={exponent}, \
                     offset={offset}, unexpected error={error:?}"
                ))
            })?;
            prop_assert_eq!(acf.sample_count(), raw.len());
            prop_assert_eq!(acf.values().len(), budget + 1);
            prop_assert_eq!(acf.values()[0].to_bits(), 1.0_f64.to_bits());
            for (lag, &actual) in acf.values().iter().enumerate() {
                prop_assert!(
                    (actual - expected[lag]).abs() <= 1e-13,
                    "{}: raw={:?}, lag={}, actual={}, exact={}",
                    label, raw, lag, actual, expected[lag],
                );
            }
        }
    }
}
