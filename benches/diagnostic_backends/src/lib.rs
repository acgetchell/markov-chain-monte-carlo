//! Experimental adapters and fixtures for the diagnostic backend decision.
//!
//! These are comparison code, not supported library APIs.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Candidate adapter: retain the public validation and range-safe normalization,
/// then delegate every covariance calculation to the actual published arima crate.
///
/// arima casts N to u32 internally; this experimental adapter rejects that
/// additional unsupported domain rather than silently truncating it.
pub fn arima_acf(samples: &[f64], max_lag: usize) -> Result<Vec<f64>, String> {
    let n = samples.len();
    if n < 2 {
        return Err(format!("insufficient samples: {n}"));
    }
    if max_lag >= n {
        return Err(format!("invalid max_lag {max_lag} for {n} samples"));
    }
    if let Some(index) = samples.iter().position(|x| !x.is_finite()) {
        return Err(format!("non-finite sample at {index}"));
    }
    let origin = samples[0];
    let mut normalized: Vec<_> = samples.iter().map(|x| x - origin).collect();
    if normalized.iter().any(|x| !x.is_finite()) {
        for (delta, &x) in normalized.iter_mut().zip(samples) {
            *delta = origin.mul_add(-0.5, x * 0.5);
        }
    }
    let scale = normalized.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err("constant trace".into());
    }
    u32::try_from(n).map_err(|_| "arima's sample-count conversion overflows u32")?;
    for x in &mut normalized {
        *x /= scale;
    }
    let values =
        arima::acf::acf(&normalized, Some(max_lag), false).map_err(|error| error.to_string())?;
    if values.iter().any(|x| !x.is_finite()) {
        return Err("arima returned non-finite correlations".into());
    }
    Ok(values)
}

/// Reproduce the existing benchmark's AR(1) workload, outside timing.
pub fn ar1(count: usize, coefficient: f64, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = 0.0;
    (0..count + 2_000)
        .filter_map(|index| {
            state = coefficient.mul_add(state, rng.random_range(-1.0..1.0));
            (index >= 2_000).then_some(state)
        })
        .collect()
}

/// Integer raw-moment oracle; bounded fixtures avoid any floating centering.
/// Returns exact rational numerators and the common positive denominator.
pub fn exact_covariances(samples: &[i16]) -> (Vec<i64>, i64) {
    assert!((2..=32).contains(&samples.len()));
    assert!(samples.iter().all(|x| (-16..=17).contains(x)));
    let n = i64::try_from(samples.len())
        .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"));
    let sum = samples.iter().map(|&x| i64::from(x)).sum::<i64>();
    let squares = samples.iter().map(|&x| i64::from(x).pow(2)).sum::<i64>();
    let denominator = n * n * squares - n * sum * sum;
    assert!(denominator > 0);
    let numerators = (0..samples.len())
        .map(|lag| {
            let left = &samples[..samples.len() - lag];
            let right = &samples[lag..];
            let products = left
                .iter()
                .zip(right)
                .map(|(&a, &b)| i64::from(a) * i64::from(b))
                .sum::<i64>();
            let endpoints = left.iter().chain(right).map(|&x| i64::from(x)).sum::<i64>();
            n * n * products - n * sum * endpoints
                + i64::try_from(left.len())
                    .unwrap_or_else(|error| panic!("diagnostic fixture failed: {error:?}"))
                    * sum
                    * sum
        })
        .collect();
    (numerators, denominator)
}

/// Exact integer pair selection; returns numerator, window, and cap flag.
pub fn exact_time(
    numerators: &[i64],
    denominator: i64,
    max_lag: usize,
    monotone: bool,
) -> (i64, usize, bool) {
    let mut sum = 0;
    let mut minimum = i64::MAX;
    let mut window = 0;
    let mut capped = true;
    for (index, pair) in numerators[..=max_lag].as_chunks::<2>().0.iter().enumerate() {
        let value = pair[0] + pair[1];
        if value <= 0 {
            capped = false;
            break;
        }
        minimum = minimum.min(value);
        sum += if monotone { minimum } else { value };
        window = 2 * index + 1;
    }
    (-denominator + 2 * sum, window, capped)
}

/// Independent raw-moment preflight for the bounded AR(1) benchmark fixtures.
pub fn reference_acf(samples: &[f64], max_lag: usize) -> Vec<f64> {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let squares = samples.iter().map(|x| x * x).sum::<f64>();
    assert!(n * mean * mean <= squares / 2.0);
    let variance = squares - n * mean * mean;
    assert!(variance > 0.0 && variance.is_finite());
    (0..=max_lag)
        .map(|lag| {
            let left = &samples[..samples.len() - lag];
            let right = &samples[lag..];
            let products = left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>();
            let endpoints = left.iter().chain(right).sum::<f64>();
            (left.len() as f64 * mean).mul_add(mean, mean.mul_add(-endpoints, products)) / variance
        })
        .collect()
}

/// Compare normalized correlations using a fixture-specific roundoff allowance.
pub fn assert_acf(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (lag, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= tolerance, "lag {lag}: {a} versus {e}");
    }
}
