//! Seeded rational properties independent of both ACF implementations.

use diagnostic_backend_comparison::{arima_acf, assert_acf};
use markov_chain_monte_carlo::Autocorrelation;
use num_rational::BigRational;
use num_traits::ToPrimitive;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[test]
fn full_range_binary64_values_match_exact_rational_acf() {
    let mut rng = StdRng::seed_from_u64(73);
    let mut raw_arima_failures = 0;
    let mut worst_native = 0.0_f64;
    let mut worst_adapter = 0.0_f64;
    for case in 0..500 {
        let n = rng.random_range(8..=32);
        let base = rng.random::<u64>() & 0x7fef_ffff_ffff_ff00;
        let samples: Vec<_> = (0..n)
            .map(|_| match case % 5 {
                0 => f64::from_bits(rng.random::<u64>() & 0xffef_ffff_ffff_ffff),
                1 => f64::from_bits(rng.random::<u64>() & 0x800f_ffff_ffff_ffff),
                2 => f64::from_bits(base + rng.random_range(0..32)),
                3 => {
                    f64::from(rng.random_range(-16..=17))
                        * 2.0_f64.powi(rng.random_range(-1000..=1000))
                }
                _ => f64::from(rng.random_range(-16..=17)),
            })
            .collect();
        assert!(samples.iter().all(|x| x.is_finite()));
        assert!(samples.iter().any(|x| *x != samples[0]));
        let rational: Vec<_> = samples
            .iter()
            .map(|&x| {
                BigRational::from_float(x).unwrap_or_else(|| {
                    panic!("finite binary64 fixture could not be converted: {x}")
                })
            })
            .collect();
        let mean =
            rational.iter().cloned().sum::<BigRational>() / BigRational::from_integer(n.into());
        let centered: Vec<_> = rational.iter().map(|x| x - &mean).collect();
        let variance = centered.iter().map(|x| x * x).sum::<BigRational>();
        let expected: Vec<_> = (0..n)
            .map(|lag| {
                (centered
                    .iter()
                    .zip(&centered[lag..])
                    .map(|(a, b)| a * b)
                    .sum::<BigRational>()
                    / &variance)
                    .to_f64()
                    .unwrap_or_else(|| {
                        panic!("bounded rational ACF could not be converted at lag {lag}")
                    })
            })
            .collect();
        let native = Autocorrelation::estimate(&samples, n - 1)
            .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"));
        let adapted = arima_acf(&samples, n - 1)
            .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"));
        assert_acf(native.values(), &expected, 1e-13);
        assert_acf(&adapted, &expected, 1e-13);
        for ((&a, &b), &e) in native.values().iter().zip(&adapted).zip(&expected) {
            worst_native = worst_native.max((a - e).abs());
            worst_adapter = worst_adapter.max((b - e).abs());
        }
        let raw = arima::acf::acf(&samples, Some(n - 1), false)
            .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"));
        if raw
            .iter()
            .zip(&expected)
            .any(|(&a, &e)| !a.is_finite() || (a - e).abs() > 1e-13)
        {
            raw_arima_failures += 1;
        }
    }
    assert!(raw_arima_failures > 0);
    eprintln!(
        "500 full-range rational fixtures: raw arima failures={raw_arima_failures}; max absolute ACF error native={worst_native:e}, adapted arima={worst_adapter:e}"
    );
}
